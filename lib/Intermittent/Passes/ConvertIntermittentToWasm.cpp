//===- ConvertIntermittentToWasm.cpp - Convert Intermittent to Wasm
//-----------------*- C++ -*-===//

#include "Intermittent/IntermittentPasses.h"
#include "Wasm/WasmOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_CONVERTINTERMITTENTTOWASM
#include "Intermittent/IntermittentPasses.h.inc"
namespace {

struct NonVolatileNewOpLowering : public OpConversionPattern<NonVolatileNewOp> {
  using OpConversionPattern<NonVolatileNewOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NonVolatileNewOp newOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<wasm::TempGlobalOp>(newOp, true,
                                                    adaptor.getInner());
    return success();
  }
};
struct NonVolatileLoadOpLowering
    : public OpConversionPattern<NonVolatileLoadOp> {
  using OpConversionPattern<NonVolatileLoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NonVolatileLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    auto elementType = loadOp.getVar().getType().getElementType();

    auto localOp = rewriter.create<wasm::TempLocalOp>(loc, elementType);

    // get the global variable and set it to the local variable
    // because we currently use local variables to pass information
    // across patterns
    rewriter.create<wasm::TempGlobalGetOp>(loc, adaptor.getVar());
    rewriter.create<wasm::TempLocalSetOp>(loc, localOp.getResult());

    rewriter.replaceOp(loadOp, localOp.getResult());

    return success();
  }
};

struct NonVolatileStoreOpLowering
    : public OpConversionPattern<NonVolatileStoreOp> {
  using OpConversionPattern<NonVolatileStoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NonVolatileStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    rewriter.create<wasm::TempLocalGetOp>(loc, adaptor.getValue());

    rewriter.replaceOpWithNewOp<wasm::TempGlobalSetOp>(storeOp,
                                                       adaptor.getVar());
    return success();
  }
};

wasm::TempGlobalOp findTempGlobalOpWithSymName(ModuleOp module, std::string key,
                                               std::string value) {
  // Traverse through each operation in the module
  for (Operation &op : module.getOps()) {
    // Check if the operation is of type TempGlobalOp
    if (auto tempGlobalOp = dyn_cast<wasm::TempGlobalOp>(&op)) {
      // Get the "global_name" attribute and check its value
      if (auto symNameAttr = tempGlobalOp->getAttrOfType<StringAttr>(key)) {
        if (symNameAttr.getValue() == value) {
          return tempGlobalOp;
        }
      }
    }
  }
  return nullptr; // Return null if no matching TempGlobalOp is found
}

int getTaskIndexBySymbolName(ModuleOp module, StringRef symName) {
  // Traverse each operation in the module
  for (Operation &op : module.getOps()) {
    // Check if the operation has a "sym_name" attribute
    if (auto symNameAttr = op.getAttrOfType<StringAttr>("sym_name")) {
      // Compare the "sym_name" attribute with symName
      if (symNameAttr.getValue() == symName) {
        // Retrieve the "index" attribute if it exists
        if (auto indexAttr = op.getAttrOfType<IntegerAttr>("task_index")) {
          return indexAttr.getValue().getSExtValue();
        } else {
          // TODO: Error handling
          llvm::errs() << "Operation with sym_name '" << symName
                       << "' found but lacks an 'index' attribute.\n";
          return -1;
        }
      }
    }
  }

  // If no operation with the matching symbol name is found
  // TODO: Error handling
  llvm::errs() << "No operation with sym_name '" << symName
               << "' found in the module.\n";
  return -1;
}

void convertTransitionToOp(MLIRContext *context, TransitionToOp transitionToOp,
                           wasm::LoopOpDeprecated loopOp, Value contLocal,
                           ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPoint(transitionToOp);
  Location loc = transitionToOp.getLoc();
  rewriter.create<wasm::CallOp>(loc, "begin_commit");
  for (auto var : transitionToOp.getVarsToStore()) {

    // var can be either of type GlobalType or a NonVolatileType
    Type innerType;
    if (auto globalType = dyn_cast<wasm::GlobalType>(var.getType())) {
      innerType = globalType.getInner();
    } else if (auto nonVolatileType =
                   dyn_cast<NonVolatileType>(var.getType())) {
      innerType = nonVolatileType.getElementType();
    }
    auto castedVar =
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, wasm::GlobalType::get(context, innerType), var)
            .getResult(0);
    rewriter.create<wasm::TempGlobalIndexOp>(loc, castedVar);
    rewriter.create<wasm::TempGlobalGetOp>(loc, castedVar);

    if (auto intType = dyn_cast<IntegerType>(innerType)) {
      if (intType.getWidth() == 32) {
        rewriter.create<wasm::CallOp>(loc, "set_i32");
      } else if (intType.getWidth() == 64) {
        rewriter.create<wasm::CallOp>(loc, "set_i64");
      }
    } else if (auto floatType = dyn_cast<FloatType>(innerType)) {
      if (floatType.getWidth() == 32) {
        rewriter.create<wasm::CallOp>(loc, "set_f32");
      } else if (floatType.getWidth() == 64) {
        rewriter.create<wasm::CallOp>(loc, "set_f64");
      }
    }
  }
  // store the next task index, if it exists
  auto nextTask = transitionToOp.getNextTask();
  auto moduleOp = transitionToOp->getParentOfType<ModuleOp>();
  Value currTaskGlobal =
      findTempGlobalOpWithSymName(moduleOp, "global_name", "curr_task")
          .getResult();
  rewriter.create<wasm::TempGlobalIndexOp>(loc, currTaskGlobal);
  // value
  int taskIndex = getTaskIndexBySymbolName(moduleOp, nextTask);
  rewriter.create<wasm::ConstantOp>(loc, rewriter.getI32IntegerAttr(taskIndex));
  rewriter.create<wasm::CallOp>(loc, "set_i32");
  rewriter.create<wasm::CallOp>(loc, "end_commit");

  // TODO: get the next task continuation from the table

  rewriter.replaceOpWithNewOp<wasm::SwitchOp>(transitionToOp, "ct", "yield");
  // when returning to this point, the previous task continuation is on the
  // stack. save it
  rewriter.create<wasm::TempLocalSetOp>(loc, contLocal);
  // jump to the beginning of the loop
  rewriter.create<wasm::BranchOpDeprecated>(loc, loopOp.getMainBlock());
}

struct IdempotentTaskOpLowering : public OpConversionPattern<IdempotentTaskOp> {
  using OpConversionPattern<IdempotentTaskOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IdempotentTaskOp taskOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto taskLoc = taskOp.getLoc();
    MLIRContext *context = taskOp.getContext();
    // Create the WasmFuncOp
    auto contType = wasm::ContinuationType::get(context, "ct", "ft");
    auto contLocalType = wasm::LocalType::get(context, contType);

    auto funcOp = rewriter.create<wasm::WasmFuncOp>(
        taskLoc, taskOp.getSymName(),
        rewriter.getFunctionType(
            /*inputs=*/{contLocalType}, /*results=*/{}));
    auto loc = funcOp.getLoc();

    // Insert a block in the function for its body
    auto *entryBlock = new Block();
    auto funcArg = entryBlock->addArgument(contLocalType, loc);
    funcOp.getBody().push_back(entryBlock);
    rewriter.setInsertionPointToStart(entryBlock);

    // declare a local variable to store the last task continuation
    auto contLocal = rewriter.create<wasm::TempLocalOp>(loc, contType);
    // save the function argument (initial task) in this local variable
    rewriter.create<wasm::TempLocalGetOp>(loc, funcArg);
    rewriter.create<wasm::TempLocalSetOp>(loc, contLocal);

    // Create an enclosing loop
    auto loopName = taskOp.getSymName().str() + "_loop";
    auto loopOp = rewriter.create<wasm::LoopOpDeprecated>(loc, loopName);
    loopOp.initialize(rewriter);

    // Inline the original task body into the loop's region
    if (failed(loopOp.inlineRegionToMainBlock(taskOp.getBody(), rewriter))) {
      return failure();
    }

    // Save the continuation of the previous task to the continuation table at
    // the beginning of the loop
    rewriter.setInsertionPointToStart(loopOp.getMainBlock());

    // retrieve the current task index
    auto moduleOp = taskOp->getParentOfType<ModuleOp>();
    Value currTaskGlobal =
        findTempGlobalOpWithSymName(moduleOp, "global_name", "curr_task")
            .getResult();
    rewriter.create<wasm::TempGlobalGetOp>(loc, currTaskGlobal);

    // retrieve the continuation
    rewriter.create<wasm::TempLocalGetOp>(loc, contLocal);

    // store the continuation to the current task index
    rewriter.create<wasm::TableSetOp>(loc, "task_table");

    // Place the wasm.return after the loop
    rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
    rewriter.create<wasm::WasmReturnOp>(loc);

    // Replace the original op with the new function
    rewriter.replaceOp(taskOp, funcOp);

    // Declare the newly created function
    rewriter.setInsertionPointAfter(funcOp);
    rewriter.create<wasm::ElemDeclareFuncOp>(loc, taskOp.getSymName());

    // handle transitionToOps
    funcOp.walk([&](TransitionToOp transitionToOp) {
      convertTransitionToOp(context, transitionToOp, loopOp, contLocal,
                            rewriter);
    });

    return success();
  }
};

class IntermittentToWasmTypeConverter : public TypeConverter {
public:
  IntermittentToWasmTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](NonVolatileType type) -> Type {
      auto elementType = type.getElementType();
      return wasm::GlobalType::get(ctx, elementType);
    });
    addConversion([ctx](IntegerType type) -> Type {
      return wasm::LocalType::get(ctx, type);
    });
    addConversion([ctx](FloatType type) -> Type {
      return wasm::LocalType::get(ctx, type);
    });
    addConversion([ctx](IndexType type) -> Type {
      return wasm::LocalType::get(ctx, IntegerType::get(ctx, 32));
    });
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      // if (inputs.size() != 1)
      //   return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });

    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      // if (inputs.size() != 1)
      //   return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });
  }
};
} // namespace

class ConvertIntermittentToWasm
    : public impl::ConvertIntermittentToWasmBase<ConvertIntermittentToWasm> {
public:
  using impl::ConvertIntermittentToWasmBase<
      ConvertIntermittentToWasm>::ConvertIntermittentToWasmBase;
  void runOnOperation() final {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    IntermittentToWasmTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<IntermittentDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    patterns.add<NonVolatileNewOpLowering, NonVolatileLoadOpLowering,
                 NonVolatileStoreOpLowering, IdempotentTaskOpLowering>(
        typeConverter, context);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::intermittent