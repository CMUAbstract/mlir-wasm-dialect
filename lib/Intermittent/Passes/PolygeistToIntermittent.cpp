//===- PolygeistToIntermittent.cpp - Convert Polygeist to Intermittent
//-----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Intermittent/IntermittentPasses.h"
#include "Wasm/WasmOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_POLYGEISTTOINTERMITTENT
#include "Intermittent/IntermittentPasses.h.inc"

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
  TaskNameAnalysis(ModuleOp moduleOp) {
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
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

} // namespace mlir::intermittent
