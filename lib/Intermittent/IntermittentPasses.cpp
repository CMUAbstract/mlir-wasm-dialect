//===- IntermittentPasses.cpp - Wasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Intermittent/IntermittentPasses.h"
#include "Wasm/WasmOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <map>
#include <string>

namespace mlir::intermittent {
#define GEN_PASS_DEF_POLYGEISTTOINTERMITTENT
#define GEN_PASS_DEF_PREPAREFORINTERMITTENT
#define GEN_PASS_DEF_CONVERTINTERMITTENTTASKTOWASM
#include "Intermittent/IntermittentPasses.h.inc"

namespace {
struct FuncOpLowering : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Ensure the fuction has no arguments and returns an integer type
    if (!funcOp.getFunctionType().getInputs().empty()) {
      return rewriter.notifyMatchFailure(funcOp, "expected no arguments");
    }
    if (funcOp.getFunctionType().getResults().empty() ||
        !isa<IntegerType>(funcOp.getFunctionType().getResult(0))) {
      return rewriter.notifyMatchFailure(funcOp,
                                         "expected an integer return type");
    }

    // Create IdempotentTaskOp with the function's name as a SymbolNameAttr
    auto idempotentTaskOp =
        rewriter.create<IdempotentTaskOp>(funcOp.getLoc(), funcOp.getName());
    idempotentTaskOp->setAttr(
        "intermittent.task",
        funcOp->getAttrOfType<IntegerAttr>("intermittent.task"));

    // Inline the body of funcOp into the new IdempotentTaskOp body
    rewriter.inlineRegionBefore(funcOp.getBody(), idempotentTaskOp.getBody(),
                                idempotentTaskOp.getBody().end());

    rewriter.eraseOp(funcOp);

    return success();
  }
};

class TaskNameAnalysis {
public:
  TaskNameAnalysis(ModuleOp module) {
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp->hasAttr("intermittent.task")) {
        taskNames[funcOp->getAttrOfType<IntegerAttr>("intermittent.task")
                      .getInt()] = funcOp.getName().str();
      }
    }
  }
  std::string get(int taskIndex) { return taskNames[taskIndex]; }

private:
  std::map<int, std::string> taskNames;
};

template <typename SourceOp>
class OpConversionPatternWithAnalysis : public OpConversionPattern<SourceOp> {
public:
  OpConversionPatternWithAnalysis(MLIRContext *context,
                                  TaskNameAnalysis &analysis,
                                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit), analysis(analysis) {}

  TaskNameAnalysis &getAnalysis() const { return analysis; }

private:
  TaskNameAnalysis &analysis;
};

struct ReturnOpLowering
    : public OpConversionPatternWithAnalysis<func::ReturnOp> {
  using OpConversionPatternWithAnalysis<
      func::ReturnOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TaskNameAnalysis &taskNameMap = getAnalysis();
    // Ensure that the returnOp has only one operand
    assert(returnOp.getNumOperands() == 1 &&
           "Expected returnOp to have only one operand");

    // Retrieve the next task index
    auto nextTaskIndexOp =
        returnOp.getOperand(0).getDefiningOp<arith::ConstantOp>();
    if (!nextTaskIndexOp)
      return failure();

    int nextTaskIndex = cast<IntegerAttr>(nextTaskIndexOp.getValue()).getInt();

    // get next task name
    std::string nextTaskName = taskNameMap.get(nextTaskIndex);

    // Gather all accessed non-volatile variables in the function
    SmallVector<Value, 4> nonVolatileVars;
    DenseSet<Value> uniqueVars;

    // FIXME: We are assuming that FuncOp is lowered before lowering ReturnOp
    // We should not assume this
    IdempotentTaskOp currTask = returnOp->getParentOfType<IdempotentTaskOp>();
    currTask.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<NonVolatileLoadOp>(op)) {
        uniqueVars.insert(loadOp.getOperand());
      } else if (auto storeOp = dyn_cast<NonVolatileStoreOp>(op)) {
        uniqueVars.insert(storeOp.getOperand(0));
      }
    });
    nonVolatileVars.append(uniqueVars.begin(), uniqueVars.end());

    // Replace returnOp with TransitionToOp, passing the non-volatile vars
    rewriter.replaceOpWithNewOp<TransitionToOp>(
        returnOp, FlatSymbolRefAttr::get(rewriter.getContext(), nextTaskName),
        nonVolatileVars);
    return success();
  }
};

struct MemRefGlobalOpLowering : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern<memref::GlobalOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check if the memref global has the intermittent.nonvolatile attribute
    if (!globalOp->hasAttr("intermittent.nonvolatile"))
      return failure();

    // Replace the memref global with a NonVolatileNewOp
    Type elementType = globalOp.getType().getElementType();

    auto nonVolatileNewOp = rewriter.create<NonVolatileNewOp>(
        globalOp.getLoc(),
        NonVolatileType::get(rewriter.getContext(), elementType),
        TypeAttr::get(elementType));

    nonVolatileNewOp->setAttr("name",
                              rewriter.getStringAttr(globalOp.getName()));

    rewriter.eraseOp(globalOp);

    return success();
  }
};

struct MemRefGetGlobalOpLowering
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern<memref::GetGlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getGlobalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Erase the memref::GetGlobalOp if it's accessing a non-volatile global

    auto globalSymbol = getGlobalOp.getName(); // This gets the symbol name

    // Search for the corresponding memref::GlobalOp in the module
    auto moduleOp = getGlobalOp->getParentOfType<ModuleOp>();
    auto global = moduleOp.lookupSymbol<memref::GlobalOp>(globalSymbol);

    if (!global || !global->hasAttr("intermittent.nonvolatile"))
      return failure();

    rewriter.eraseOp(getGlobalOp);
    return success();
  }
};

struct AffineLoadOpLowering : public OpConversionPattern<affine::AffineLoadOp> {
  using OpConversionPattern<affine::AffineLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(affine::AffineLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Retrieve the memref operand
    Value memref = loadOp.getMemRef();
    ModuleOp moduleOp = loadOp->getParentOfType<ModuleOp>();

    // Check if the memref comes from a memref::GetGlobalOp
    if (auto getGlobalOp = memref.getDefiningOp<memref::GetGlobalOp>()) {
      auto globalOp =
          moduleOp.lookupSymbol<memref::GlobalOp>(getGlobalOp.getName());
      if (globalOp && globalOp->hasAttr("intermittent.nonvolatile")) {
        auto globalName = globalOp.getSymName();
        Value nonVolatile;
        // FIXME: We are assuming that GlobalOps/GetGlobalOps are lowered before
        // AffineLoadOps. We should not assume this.
        for (auto nonVolatileNewOp : moduleOp.getOps<NonVolatileNewOp>()) {
          if (nonVolatileNewOp->getAttrOfType<StringAttr>("name").getValue() ==
              globalName) {
            nonVolatile = nonVolatileNewOp.getResult();
            break;
          }
        }
        rewriter.replaceOpWithNewOp<NonVolatileLoadOp>(
            loadOp, loadOp.getResult().getType(), nonVolatile);
        return success();
      }
    }

    return failure();
  }
};

struct AffineStoreOpLowering
    : public OpConversionPattern<affine::AffineStoreOp> {
  using OpConversionPattern<affine::AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(affine::AffineStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Retrieve the memref operand
    Value memref = storeOp.getMemRef();
    ModuleOp moduleOp = storeOp->getParentOfType<ModuleOp>();

    // Check if the memref comes from a memref::GetGlobalOp
    if (auto getGlobalOp = memref.getDefiningOp<memref::GetGlobalOp>()) {
      auto globalOp =
          moduleOp.lookupSymbol<memref::GlobalOp>(getGlobalOp.getName());
      if (globalOp && globalOp->hasAttr("intermittent.nonvolatile")) {
        auto globalName = globalOp.getSymName();
        Value nonVolatile;
        // FIXME: We are assuming that GlobalOps/GetGlobalOps are lowered before
        // AffineStoreOps. We should not assume this.
        for (auto nonVolatileNewOp : moduleOp.getOps<NonVolatileNewOp>()) {
          if (nonVolatileNewOp->getAttrOfType<StringAttr>("name").getValue() ==
              globalName) {
            nonVolatile = nonVolatileNewOp.getResult();
            break;
          }
        }
        rewriter.replaceOpWithNewOp<NonVolatileStoreOp>(
            storeOp, nonVolatile, storeOp.getValueToStore());
        return success();
      }
    }

    return failure();
  }
};

class PolygeistToIntermittent
    : public impl::PolygeistToIntermittentBase<PolygeistToIntermittent> {
public:
  using impl::PolygeistToIntermittentBase<
      PolygeistToIntermittent>::PolygeistToIntermittentBase;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    ConversionTarget target(*context);
    TaskNameAnalysis analysis(moduleOp);

    // TODO: They should be illegal only if they have the
    // intermittent.nonvolatile
    // for some reason target.addDynamicallyLegalOp() doesn't work
    target.addIllegalOp<memref::GlobalOp>();
    target.addIllegalOp<memref::GetGlobalOp>();
    target.addIllegalOp<affine::AffineLoadOp>();
    target.addIllegalOp<affine::AffineStoreOp>();
    target.addLegalDialect<intermittent::IntermittentDialect>();

    RewritePatternSet memRefLoweringPatterns(context);
    memRefLoweringPatterns
        .add<MemRefGlobalOpLowering, MemRefGetGlobalOpLowering,
             AffineLoadOpLowering, AffineStoreOpLowering>(context);

    if (failed(applyPartialConversion(moduleOp, target,
                                      std::move(memRefLoweringPatterns)))) {
      signalPassFailure();
    }

    // TODO: They should be illegal only if they have the
    // intermittent.task attribute
    // for some reason target.addDynamicallyLegalOp() doesn't work
    target.addIllegalOp<func::FuncOp>();
    target.addIllegalOp<func::ReturnOp>();

    RewritePatternSet funcLoweringPatterns(context);
    funcLoweringPatterns.add<FuncOpLowering>(context);
    funcLoweringPatterns.add<ReturnOpLowering>(context, analysis);

    if (failed(applyPartialConversion(moduleOp, target,
                                      std::move(funcLoweringPatterns)))) {
      signalPassFailure();
    }
  }
};

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
  auto contType = wasm::ContinuationType::get(builder.getContext(), "ct", "ft");
  builder.create<wasm::FuncTypeDeclOp>(loc, "ft",
                                       builder.getFunctionType({contType}, {}));
  builder.create<wasm::ContinuationTypeDeclOp>(loc, contType);
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

    // after the end of the function, add a declaration for the function
    rewriter.setInsertionPointAfter(taskOp);
    rewriter.create<wasm::ElemDeclareFuncOp>(taskOp.getLoc(),
                                             taskOp.getSymName());
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
} // namespace

} // namespace mlir::intermittent
