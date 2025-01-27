//===- ConvertIntermittentToLLVM.cpp - Convert Intermittent to LLVM
//-----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Intermittent/IntermittentPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_CONVERTINTERMITTENTTOLLVM
#include "Intermittent/IntermittentPasses.h.inc"

static LLVM::LLVMFuncOp getOrInsertFunction(StringRef funcName,
                                            FunctionType functionType,
                                            PatternRewriter &rewriter,
                                            ModuleOp module) {
  // Check if function already exists
  if (auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    return funcOp;
  }

  // If not, create it
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      module.getLoc(), funcName, functionType, LLVM::Linkage::External,
      /*dsoLocal=*/false);
  return llvmFuncOp;
}

// ------------------------------------------------------------------
// Helper: Insert coroutine intrinsics and memory allocation
// Returns (id, hdl) where:
//   id  : token from llvm.coro.id
//   hdl : pointer from llvm.coro.begin
// ------------------------------------------------------------------
static std::pair<Value, Value> createCoroutineSetup(Location loc,
                                                    ModuleOp module,
                                                    PatternRewriter &rewriter,
                                                    ValueRange extraArgs = {}) {
  auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto tokenTy = LLVM::LLVMTokenType::get(rewriter.getContext());
  auto i32Ty = IntegerType::get(rewriter.getContext(), 32);

  // CoroId: %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  auto coroIdFuncType = rewriter.getFunctionType(
      {i32Ty, llvmPtrTy, llvmPtrTy, llvmPtrTy}, tokenTy);
  auto coroIdFunc =
      getOrInsertFunction("llvm.coro.id", coroIdFuncType, rewriter, module);

  Value zeroConst = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(0));
  Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, llvmPtrTy);
  SmallVector<Value, 4> coroIdArgs = {zeroConst, nullPtr, nullPtr, nullPtr};
  Value id = rewriter
                 .create<LLVM::CallOp>(
                     loc, tokenTy, SymbolRefAttr::get(coroIdFunc), coroIdArgs)
                 .getResult();

  // CoroSize: %size = call i32 @llvm.coro.size.i32()
  auto coroSizeFuncType = rewriter.getFunctionType({}, i32Ty);
  auto coroSizeFunc = getOrInsertFunction("llvm.coro.size.i32",
                                          coroSizeFuncType, rewriter, module);
  Value size =
      rewriter
          .create<LLVM::CallOp>(loc, i32Ty, SymbolRefAttr::get(coroSizeFunc),
                                ValueRange{})
          .getResult();

  // malloc: %alloc = call ptr @malloc(i32 %size)
  auto mallocFuncType = rewriter.getFunctionType({i32Ty}, llvmPtrTy);
  auto mallocFunc =
      getOrInsertFunction("malloc", mallocFuncType, rewriter, module);
  Value alloc = rewriter
                    .create<LLVM::CallOp>(loc, llvmPtrTy,
                                          SymbolRefAttr::get(mallocFunc), size)
                    .getResult();

  // CoroBegin: %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  auto coroBeginFuncType =
      rewriter.getFunctionType({tokenTy, llvmPtrTy}, llvmPtrTy);
  auto coroBeginFunc = getOrInsertFunction("llvm.coro.begin", coroBeginFuncType,
                                           rewriter, module);
  Value hdl = rewriter
                  .create<LLVM::CallOp>(loc, llvmPtrTy,
                                        SymbolRefAttr::get(coroBeginFunc),
                                        ValueRange{id, alloc})
                  .getResult();

  return std::make_pair(id, hdl);
}

// ------------------------------------------------------------------
// Helper: Insert the body (or a placeholder) into the loop block
// (Replace or extend this function to clone/copy the ops of IdempotentTaskOp)
// ------------------------------------------------------------------
static void insertTaskBody(Location loc, IdempotentTaskOp op,
                           PatternRewriter &rewriter, Block *loopBlock,
                           ModuleOp module) {
  // Example: call getTaskIndex("task2") and store to a global
  // Here you could transform or clone the actual ops within op's body.

  // For demonstration, assume we skip the details and just insert a comment
  rewriter.setInsertionPointToStart(loopBlock);
  // ... your transformation/cloning code goes here ...
}

// ------------------------------------------------------------------
// Helper: Insert coro.suspend and branching logic
//    loop -> switch -> (loop or cleanup), then -> suspend
// Returns a pair of (suspendBlock, cleanupBlock) for clarity.
// ------------------------------------------------------------------
static std::pair<Block *, Block *>
insertCoroutineSuspend(Location loc, ModuleOp module, PatternRewriter &rewriter,
                       Value id, Value hdl, Block *currentBlock) {
  auto tokenTy = LLVM::LLVMTokenType::get(rewriter.getContext());
  auto i1Ty = rewriter.getI1Type();
  auto i8Ty = rewriter.getI8Type();

  rewriter.setInsertionPointToEnd(currentBlock);

  // %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  auto coroSuspendType = rewriter.getFunctionType({tokenTy, i1Ty}, i8Ty);
  auto coroSuspendFunc = getOrInsertFunction("llvm.coro.suspend",
                                             coroSuspendType, rewriter, module);

  Value noneToken = rewriter.create<LLVM::UndefOp>(loc, tokenTy);
  Value falseConst =
      rewriter.create<LLVM::ConstantOp>(loc, i1Ty, rewriter.getBoolAttr(false));
  Value suspendVal =
      rewriter
          .create<LLVM::CallOp>(loc, i8Ty, SymbolRefAttr::get(coroSuspendFunc),
                                ValueRange{noneToken, falseConst})
          .getResult();

  // Create the suspend and cleanup blocks
  auto *funcBody = currentBlock->getParent();
  auto *suspendBlock = rewriter.createBlock(funcBody);
  auto *cleanupBlock = rewriter.createBlock(funcBody);

  SmallVector<APInt, 2> caseValues;
  caseValues.push_back(APInt(32, 0));
  caseValues.push_back(APInt(32, 1));

  auto caseValuesAttr = DenseIntElementsAttr::get(
      VectorType::get(2, rewriter.getI32Type()), caseValues);

  rewriter.create<LLVM::SwitchOp>(
      loc, suspendVal, suspendBlock,
      /*defaultOperands=*/ValueRange{},
      /*caseValues=*/caseValuesAttr,
      /*caseDestinations=*/ArrayRef<Block *>{currentBlock, cleanupBlock},
      /*caseOperands=*/ArrayRef<ValueRange>{{}, {}});

  return std::make_pair(suspendBlock, cleanupBlock);
}

// ------------------------------------------------------------------
// Helper: Insert cleanup logic (coro.free, free) then branch to suspend
// ------------------------------------------------------------------
static void insertCoroutineCleanup(Location loc, ModuleOp module,
                                   PatternRewriter &rewriter,
                                   Block *cleanupBlock, Block *suspendBlock,
                                   Value id, Value hdl) {
  auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto tokenTy = LLVM::LLVMTokenType::get(rewriter.getContext());

  rewriter.setInsertionPointToStart(cleanupBlock);

  // %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  auto coroFreeFuncType =
      rewriter.getFunctionType({tokenTy, llvmPtrTy}, llvmPtrTy);
  auto coroFreeFunc =
      getOrInsertFunction("llvm.coro.free", coroFreeFuncType, rewriter, module);
  Value mem = rewriter
                  .create<LLVM::CallOp>(loc, llvmPtrTy,
                                        SymbolRefAttr::get(coroFreeFunc),
                                        ValueRange{id, hdl})
                  .getResult();

  // call void @free(ptr %mem)
  auto freeFuncType = rewriter.getFunctionType(
      {llvmPtrTy}, LLVM::LLVMVoidType::get(rewriter.getContext()));
  auto freeFunc = getOrInsertFunction("free", freeFuncType, rewriter, module);
  rewriter.create<LLVM::CallOp>(loc, TypeRange{}, SymbolRefAttr::get(freeFunc),
                                mem);

  // branch to suspend
  rewriter.create<LLVM::BrOp>(loc, ValueRange{}, suspendBlock);
}

// ------------------------------------------------------------------
// Helper: finalize the suspend block (llvm.coro.end) and return
// ------------------------------------------------------------------
static void insertCoroutineEnd(Location loc, ModuleOp module,
                               PatternRewriter &rewriter, Block *suspendBlock,
                               Value hdl, Value id) {
  rewriter.setInsertionPointToStart(suspendBlock);

  auto i1Ty = rewriter.getI1Type();
  auto tokenTy = LLVM::LLVMTokenType::get(rewriter.getContext());
  auto falseVal =
      rewriter.create<LLVM::ConstantOp>(loc, i1Ty, rewriter.getBoolAttr(false));
  auto noneToken = rewriter.create<LLVM::UndefOp>(loc, tokenTy);

  // %unused = call i1 @llvm.coro.end(ptr %hdl, i1 false, token none)
  auto coroEndFuncType =
      rewriter.getFunctionType({hdl.getType(), i1Ty, tokenTy}, i1Ty);
  auto coroEndFunc =
      getOrInsertFunction("llvm.coro.end", coroEndFuncType, rewriter, module);
  rewriter.create<LLVM::CallOp>(loc, i1Ty, SymbolRefAttr::get(coroEndFunc),
                                ValueRange{hdl, falseVal, noneToken});

  // ret ptr %hdl
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange{hdl});
}

// ------------------------------------------------------------------
// Pattern: Lower TaskOp to an LLVM Function using coroutines
// ------------------------------------------------------------------
struct IdempotentTaskOpLowering : public OpConversionPattern<IdempotentTaskOp> {
  using OpConversionPattern<IdempotentTaskOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IdempotentTaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // 1. Create LLVM function with ptr return type
    auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto funcType = rewriter.getFunctionType({}, llvmPtrTy);

    StringRef taskName = op.getSymName();
    auto newFuncOp = getOrInsertFunction(taskName, funcType, rewriter, module);

    // Optionally add 'presplitcoroutine' attribute
    // newFuncOp->setAttr("passthrough",
    // rewriter.getArrayAttr({rewriter.getStringAttr("presplitcoroutine")}));

    // 2. Create entry block
    auto &entryBlock = *newFuncOp.addEntryBlock(rewriter);
    rewriter.setInsertionPointToStart(&entryBlock);

    // 3. Insert coroutine setup: id, hdl
    auto [id, hdl] = createCoroutineSetup(loc, module, rewriter);

    // 4. Create loop block and branch there
    auto loopBlock = rewriter.createBlock(&newFuncOp.getBody());
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, loopBlock);

    // 5. Insert body in loop block
    insertTaskBody(loc, op, rewriter, loopBlock, module);

    // 6. Insert coro.suspend logic
    auto [suspendBlock, cleanupBlock] =
        insertCoroutineSuspend(loc, module, rewriter, id, hdl, loopBlock);

    // 7. Insert cleanup logic
    insertCoroutineCleanup(loc, module, rewriter, cleanupBlock, suspendBlock,
                           id, hdl);

    // 8. Insert final coro.end and return
    insertCoroutineEnd(loc, module, rewriter, suspendBlock, hdl, id);

    // Remove original op
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertIntermittentToLLVM
    : public impl::ConvertIntermittentToLLVMBase<ConvertIntermittentToLLVM> {
  using ConvertIntermittentToLLVMBase<
      ConvertIntermittentToLLVM>::ConvertIntermittentToLLVMBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto context = &getContext();

    // Create type converter and populate conversion target + patterns

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<IdempotentTaskOp>();

    RewritePatternSet patterns(context);
    patterns.add<IdempotentTaskOpLowering>(context);

    // Apply the conversion
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::intermittent