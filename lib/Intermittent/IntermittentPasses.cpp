//===- IntermittentPasses.cpp - Wasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Intermittent/IntermittentPasses.h"
#include "Wasm/WasmOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_PREPAREFORINTERMITTENT
#define GEN_PASS_DEF_CONVERTINTERMITTENTTASKTOWASM
#include "Intermittent/IntermittentPasses.h.inc"

void importHostFunctions(OpBuilder &builder, Location loc) {
  auto simpleFuncType = builder.getFunctionType({}, {});
  builder.create<wasm::ImportFuncOp>(loc, "begin_commit", simpleFuncType);
  builder.create<wasm::ImportFuncOp>(loc, "end_commit", simpleFuncType);
  builder.create<wasm::ImportFuncOp>(
      loc, "set_i32",
      builder.getFunctionType({builder.getI32Type(), builder.getI32Type()},
                              {}));
  builder.create<wasm::ImportFuncOp>(
      loc, "set_i64",
      builder.getFunctionType({builder.getI32Type(), builder.getI64Type()},
                              {}));
  builder.create<wasm::ImportFuncOp>(
      loc, "set_f32",
      builder.getFunctionType({builder.getI32Type(), builder.getF32Type()},
                              {}));
  builder.create<wasm::ImportFuncOp>(
      loc, "set_f64",
      builder.getFunctionType({builder.getI32Type(), builder.getF64Type()},
                              {}));
  builder.create<wasm::ImportFuncOp>(
      loc, "get_i32",
      builder.getFunctionType({builder.getI32Type()}, {builder.getI32Type()}));
  builder.create<wasm::ImportFuncOp>(
      loc, "get_i64",
      builder.getFunctionType({builder.getI32Type()}, {builder.getI64Type()}));
  builder.create<wasm::ImportFuncOp>(
      loc, "get_f32",
      builder.getFunctionType({builder.getI32Type()}, {builder.getF32Type()}));
  builder.create<wasm::ImportFuncOp>(
      loc, "get_f64",
      builder.getFunctionType({builder.getI32Type()}, {builder.getF64Type()}));
}

void addTypes(OpBuilder &builder, Location loc) {
  builder.create<wasm::FuncTypeOp>(loc, "ft", builder.getFunctionType({}, {}));
  builder.create<wasm::ContinuationTypeOp>(loc, "ct", "ft");
}

void addTag(OpBuilder &builder, Location loc) {
  builder.create<wasm::TagOp>(loc, "yield");
}

void addTable(ModuleOp &moduleOp, MLIRContext *context, OpBuilder &builder,
              Location loc) {
  int numTasks = 0;
  llvm::SmallVector<mlir::Attribute, 4> taskNames;
  moduleOp.walk([&](IdempotentTaskOp taskOp) {
    numTasks++;
    taskNames.push_back(FlatSymbolRefAttr::get(context, taskOp.getSymName()));
  });
  builder.create<wasm::ContinuationTableOp>(loc, "task_table", numTasks, "ct");
}

void addMainFunction(ModuleOp &moduleOp, MLIRContext *context,
                     OpBuilder &builder, Location loc) {

  // add a global variable to store the current task
  auto currTaskGlobalOp =
      builder.create<wasm::TempGlobalOp>(loc, true, builder.getI32Type());
  // global_name is a temporary attribute used to reference this global variable
  currTaskGlobalOp->setAttr("global_name", builder.getStringAttr("curr_task"));

  auto mainFuncOp = builder.create<wasm::WasmFuncOp>(
      loc, "main", builder.getFunctionType({}, {}));
  auto &entryRegion = mainFuncOp.getBody();
  auto *entryBlock = builder.createBlock(&entryRegion);
  builder.setInsertionPointToEnd(entryBlock);

  // Set table elements (TODO: Ideally tables should be initialized through the
  // elem segment)

  size_t taskIndex = 0;
  for (auto taskOp : moduleOp.getOps<IdempotentTaskOp>()) {
    taskOp->setAttr("task_index", builder.getI32IntegerAttr(taskIndex));
    builder.create<wasm::ConstantOp>(loc, builder.getI32IntegerAttr(taskIndex));
    taskIndex++;
    builder.create<wasm::FuncRefOp>(loc, taskOp.getSymName());
    builder.create<wasm::ContNewOp>(loc, "ct");
    builder.create<wasm::TableSetOp>(loc, "task_table");
  }

  // Restore global variables Restore the current task
  for (auto tempGlobalOp : moduleOp.getOps<wasm::TempGlobalOp>()) {
    // The TempGlobalIndexOp will be replaced by wasm::ConstantOp
    // representing the index value of the global variable.
    builder.create<wasm::TempGlobalIndexOp>(loc, tempGlobalOp.getResult());
    builder.create<wasm::CallOp>(loc, "get_i32");
    builder.create<wasm::TempGlobalSetOp>(loc, tempGlobalOp.getResult());
  }
  // restore the current task
  builder.create<wasm::TempGlobalGetOp>(loc, currTaskGlobalOp.getResult());
  builder.create<wasm::TableGetOp>(loc, "task_table");

  // Run the current task
  builder.create<wasm::ResumeSwitchOp>(loc, "ct", "yield");

  builder.create<wasm::WasmReturnOp>(loc);
}

class PrepareForIntermittent
    : public impl::PrepareForIntermittentBase<PrepareForIntermittent> {
public:
  using impl::PrepareForIntermittentBase<
      PrepareForIntermittent>::PrepareForIntermittentBase;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    OpBuilder builder(context);
    builder.setInsertionPoint(moduleOp.getBody(), moduleOp.getBody()->begin());

    auto loc = moduleOp.getLoc();
    importHostFunctions(builder, loc);
    addTypes(builder, loc);
    addTag(builder, loc);
    addTable(moduleOp, context, builder, loc);
    addMainFunction(moduleOp, context, builder, loc);
  }
};

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
    MLIRContext *context = loadOp.getContext();
    auto elementType = loadOp.getVar().getType().getElementType();

    auto globalCastOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, wasm::GlobalType::get(context, elementType), adaptor.getVar());

    auto localOp = rewriter.create<wasm::TempLocalOp>(loc, elementType);

    // get the global variable and set it to the local variable
    // because we currently use local variables to pass information
    // across patterns
    rewriter.create<wasm::TempGlobalGetOp>(loc, globalCastOp.getResult(0));
    rewriter.create<wasm::TempLocalSetOp>(loc, localOp.getResult());

    auto localCastOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, elementType, localOp.getResult());

    rewriter.replaceOp(loadOp, localCastOp.getResult(0));

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
    MLIRContext *context = storeOp.getContext();

    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, wasm::LocalType::get(context, adaptor.getValue().getType()),
        adaptor.getValue());
    rewriter.create<wasm::TempLocalGetOp>(loc, castOp.getResult(0));

    rewriter.replaceOpWithNewOp<wasm::TempGlobalSetOp>(storeOp,
                                                       adaptor.getVar());
    return success();
  }
};

struct IdempotentTaskOpLowering : public OpConversionPattern<IdempotentTaskOp> {
  using OpConversionPattern<IdempotentTaskOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IdempotentTaskOp taskOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto funcOp = rewriter.create<wasm::WasmFuncOp>(
        taskOp.getLoc(), taskOp.getSymName(), rewriter.getFunctionType({}, {}));
    rewriter.inlineRegionBefore(taskOp.getBody(), funcOp.getBody(),
                                funcOp.end());

    rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
    rewriter.create<wasm::WasmReturnOp>(taskOp.getLoc());

    rewriter.replaceOp(taskOp, funcOp);
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

struct TransitionToOpLowering : public OpConversionPattern<TransitionToOp> {
  using OpConversionPattern<TransitionToOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TransitionToOp transitionToOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = transitionToOp.getLoc();
    rewriter.create<wasm::CallOp>(loc, "begin_commit");
    for (auto var : adaptor.getVarsToStore()) {
      rewriter.create<wasm::TempGlobalIndexOp>(loc, var);
      rewriter.create<wasm::TempGlobalGetOp>(loc, var);

      // var can be either of type GlobalType or a NonVolatileType
      Type innerType;
      if (auto globalType = dyn_cast<wasm::GlobalType>(var.getType())) {
        innerType = globalType.getInner();
      } else if (auto nonVolatileType =
                     dyn_cast<NonVolatileType>(var.getType())) {
        innerType = nonVolatileType.getElementType();
      }

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
    if (nextTask.has_value()) {
      // index
      auto moduleOp = transitionToOp->getParentOfType<ModuleOp>();
      Value currTaskGlobal =
          findTempGlobalOpWithSymName(moduleOp, "global_name", "curr_task")
              .getResult();
      rewriter.create<wasm::TempGlobalIndexOp>(loc, currTaskGlobal);
      // value
      int taskIndex = getTaskIndexBySymbolName(moduleOp, nextTask.value());
      rewriter.create<wasm::ConstantOp>(loc,
                                        rewriter.getI32IntegerAttr(taskIndex));
      rewriter.create<wasm::CallOp>(loc, "set_i32");
    }

    rewriter.create<wasm::CallOp>(loc, "end_commit");

    rewriter.replaceOpWithNewOp<wasm::SwitchOp>(transitionToOp, "ct", "yield");
    return success();
  }
};

class ConvertIntermittentTaskToWasm
    : public impl::ConvertIntermittentTaskToWasmBase<
          ConvertIntermittentTaskToWasm> {
public:
  using impl::ConvertIntermittentTaskToWasmBase<
      ConvertIntermittentTaskToWasm>::ConvertIntermittentTaskToWasmBase;
  void runOnOperation() final {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

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
                 NonVolatileStoreOpLowering, IdempotentTaskOpLowering,
                 TransitionToOpLowering>(context);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::intermittent
