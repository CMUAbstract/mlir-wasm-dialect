//===- PreprocessIntermittentToStd.cpp - Preprocess Intermittent to Std
//-----------------*- C++ -*-===//

#define DEBUG_TYPE "preprocessintermittenttostd"

#include "Intermittent/IntermittentPasses.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_PREPROCESSINTERMITTENTTOSTD
#include "Intermittent/IntermittentPasses.h.inc"

namespace {
/// Gathers symbol names from all IdempotentTaskOp in the module.
static SmallVector<StringRef> collectTaskSymbols(ModuleOp module) {
  SmallVector<StringRef> taskNames;
  for (auto taskOp : module.getOps</*intermittent::*/ IdempotentTaskOp>()) {
    taskNames.push_back(taskOp.getSymName());
  }
  return taskNames;
}

/// Utility to get or insert an LLVM function with the given name and type.
static LLVM::LLVMFuncOp
getOrInsertFunction(ModuleOp module, StringRef funcName,
                    LLVM::LLVMFunctionType functionType) {
  // Check if function already exists
  if (auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    return funcOp;
  }

  // If not, create it
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto llvmFuncOp = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), funcName, functionType, LLVM::Linkage::External,
      /*dsoLocal=*/false);
  return llvmFuncOp;
}

/// Looks up or creates an i32 global named `globalName`. Returns a pointer to
/// it.
static Value lookupOrCreateGlobalI32(ModuleOp module, OpBuilder &builder,
                                     StringRef globalName) {
  MLIRContext *ctx = builder.getContext();
  // Check if it already exists.
  if (auto globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
    // Use AddressOfOp to get a pointer to it.
    return builder.create<LLVM::AddressOfOp>(module.getLoc(),
                                             LLVM::LLVMPointerType::get(ctx),
                                             globalOp.getSymName());
  }

  // Otherwise, create a new global with an initial value of 0.
  auto i32Ty = IntegerType::get(ctx, 32);
  auto newGlobal =
      OpBuilder::atBlockBegin(module.getBody())
          .create<LLVM::GlobalOp>(module.getLoc(), i32Ty,
                                  /*isConstant=*/false, LLVM::Linkage::External,
                                  globalName, builder.getI32IntegerAttr(0),
                                  /*alignment=*/0,
                                  /*addr_space=*/0);

  // And return a pointer to it.
  return builder.create<LLVM::AddressOfOp>(
      module.getLoc(), LLVM::LLVMPointerType::get(ctx), newGlobal.getSymName());
}

/// Builds the body of the `main` function:
///  - Allocates an array of pointers for each task handle.
///  - Calls each task, storing its coroutine handle in the array.
///  - Enters an infinite loop, using `@nextTaskIdx` to select which handle to
///  resume.
static void buildMainFunctionBody(ModuleOp module, LLVM::LLVMFuncOp mainFunc,
                                  ArrayRef<StringRef> taskNames) {
  // Common LLVM types.
  MLIRContext *ctx = module.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);

  // Create an entry block.
  OpBuilder builder(ctx);
  Block *entryBlock = mainFunc.addEntryBlock(builder);
  builder.setInsertionPointToStart(entryBlock);

  // 1) alloca array [numTasks x ptr]
  auto numTasks = builder.create<LLVM::ConstantOp>(
      mainFunc.getLoc(), i32Ty, builder.getI32IntegerAttr(taskNames.size()));
  // Create array type: [numTasks x ptr]
  auto arrayType = LLVM::LLVMArrayType::get(ptrTy, taskNames.size());

  auto arrayAlloca = builder.create<LLVM::AllocaOp>(
      mainFunc.getLoc(),
      ptrTy,     // result type
      arrayType, // element type (array of pointers)
      numTasks,  // array size (should be 1 since we're allocating one array)
      /*alignment=*/0);

  // 2) For each task, call it and store its handle
  for (unsigned i = 0; i < taskNames.size(); ++i) {
    // Insert or retrieve the task function
    auto taskFnType =
        LLVM::LLVMFunctionType::get(ptrTy, {}, /*isVarArg=*/false);
    auto taskFn = getOrInsertFunction(module, taskNames[i], taskFnType);

    // call ptr @taskN()
    auto callOp =
        builder.create<LLVM::CallOp>(mainFunc.getLoc(), TypeRange{ptrTy},
                                     SymbolRefAttr::get(taskFn), ValueRange{});

    // store the handle in arrayAlloca at index i
    auto zeroI32 = builder.create<LLVM::ConstantOp>(
        mainFunc.getLoc(), i32Ty, builder.getI32IntegerAttr(0));
    auto idxI = builder.create<LLVM::ConstantOp>(mainFunc.getLoc(), i32Ty,
                                                 builder.getI32IntegerAttr(i));

    // GEP => pointer to arrayAlloca[0][i]
    auto elementPtr =
        builder.create<LLVM::GEPOp>(mainFunc.getLoc(),
                                    ptrTy,       // result type
                                    arrayType,   // element type (array)
                                    arrayAlloca, // base pointer
                                    ArrayRef<Value>{zeroI32, idxI}); // indices
    // store handle
    builder.create<LLVM::StoreOp>(mainFunc.getLoc(), callOp.getResult(),
                                  elementPtr);
  }

  // 3) Create a loop block
  auto *loopBlock = builder.createBlock(&mainFunc.getBody());
  builder.create<LLVM::BrOp>(mainFunc.getLoc(), ValueRange{}, loopBlock);
  builder.setInsertionPointToStart(loopBlock);

  //  - load @nextTaskIdx
  auto globalPtr = lookupOrCreateGlobalI32(module, builder, "nextTaskIdx");
  auto nextIdx =
      builder.create<LLVM::LoadOp>(mainFunc.getLoc(), i32Ty, globalPtr);

  //  - GEP arrayAlloca at nextIdx
  auto zeroI32 = builder.create<LLVM::ConstantOp>(mainFunc.getLoc(), i32Ty,
                                                  builder.getI32IntegerAttr(0));
  auto elementPtr = builder.create<LLVM::GEPOp>(
      mainFunc.getLoc(), ptrTy, arrayType, arrayAlloca,
      ArrayRef<Value>{zeroI32, nextIdx});

  //  - load handle
  auto handle =
      builder.create<LLVM::LoadOp>(mainFunc.getLoc(), ptrTy, elementPtr);

  //  - call llvm.coro.resume(ptr)
  auto coroResumeTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                  {ptrTy}, /*isVarArg=*/false);
  auto coroResumeFn =
      getOrInsertFunction(module, "llvm.coro.resume", coroResumeTy);

  builder.create<LLVM::CallOp>(mainFunc.getLoc(), TypeRange{},
                               SymbolRefAttr::get(coroResumeFn),
                               ValueRange{handle});

  //  - branch back to loop
  builder.create<LLVM::BrOp>(mainFunc.getLoc(), ValueRange{}, loopBlock);

  // Create a return block if needed. We never branch to it here.
  auto *returnBlock = builder.createBlock(&mainFunc.getBody());
  builder.setInsertionPointToStart(returnBlock);
  builder.create<LLVM::ReturnOp>(mainFunc.getLoc(), ValueRange{});
}

} // namespace

struct PreprocessIntermittentToStd
    : public impl::PreprocessIntermittentToStdBase<
          PreprocessIntermittentToStd> {
  using impl::PreprocessIntermittentToStdBase<
      PreprocessIntermittentToStd>::PreprocessIntermittentToStdBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    // 1) Collect all IdempotentTaskOps
    auto taskNames = collectTaskSymbols(module);
    if (taskNames.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[CreateMainForTasksPass] No IdempotentTaskOp found.\n");
      return;
    }

    // 2) Create () -> () function type for main
    auto mainFnType =
        LLVM::LLVMFunctionType::get({LLVM::LLVMVoidType::get(ctx)}, {},
                                    /*isVarArg=*/false);

    // Insert or retrieve the main function
    auto mainFunc = getOrInsertFunction(module, "main", mainFnType);

    if (!mainFunc.empty()) {
      emitError(mainFunc.getLoc()) << "main function already exists";
      signalPassFailure();
      return;
    }

    // 3) Build the main function body
    buildMainFunctionBody(module, mainFunc, taskNames);

    LLVM_DEBUG(llvm::dbgs() << "[CreateMainForTasksPass] main created with "
                            << taskNames.size() << " tasks.\n");
  }
};

} // namespace mlir::intermittent