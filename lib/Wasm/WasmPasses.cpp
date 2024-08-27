//===- WasmPasses.cpp - Wasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Wasm/WasmPasses.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace mlir::wasm {
#define GEN_PASS_DEF_VARIABLEANALYSIS
#define GEN_PASS_DEF_CONVERTTOWASM
#include "Wasm/WasmPasses.h.inc"

using func::FuncOp;
using mlir::Value;
class VariableAnalysis {
public:
  VariableAnalysis(Operation *op) {
    numArguments = 0;
    numVariables = 0;
    if (auto func = dyn_cast<FuncOp>(op)) {
      numArguments = func.getNumArguments();
      // TODO: initialize reg2Loc with arguments

      func.walk([&](Operation *op) {
        if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
          mlir::Value result = constantOp.getResult();
          reg2Loc.push_back(result);
          numVariables++;
        }
        if (auto addOp = dyn_cast<mlir::arith::AddIOp>(op)) {
          mlir::Value result = addOp.getResult();
          reg2Loc.push_back(result);
          numVariables++;
        }
        // TODO: handle other operations that define new variables
      });
    }
  }
  int getNumVariables() { return numVariables; }
  int getLocal(const mlir::Value &reg) {
    auto result = std::find(reg2Loc.begin(), reg2Loc.end(), reg);
    if (result != reg2Loc.end()) {
      return result - reg2Loc.begin() + numArguments;
    }
    return -1;
  }
  // NOTE: This function should be called before erasing operations
  std::vector<mlir::Attribute> getTypeAttrs() {
    std::vector<mlir::Attribute> types;
    types.reserve(reg2Loc.size());
    std::transform(
        reg2Loc.begin(), reg2Loc.end(), std::back_inserter(types),
        [](const auto &reg) { return mlir::TypeAttr::get(reg.getType()); });
    return types;
  }

private:
  int numArguments;
  int numVariables;
  std::vector<mlir::Value> reg2Loc;
};

template <typename SourceOp>
class OpConversionPatternWithAnalysis : public OpConversionPattern<SourceOp> {
public:
  OpConversionPatternWithAnalysis(MLIRContext *context,
                                  VariableAnalysis &analysis,
                                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit), analysis(analysis) {}

  VariableAnalysis &getAnalysis() const { return analysis; }

private:
  VariableAnalysis &analysis;
};

struct ConvertAdd : public OpConversionPatternWithAnalysis<arith::AddIOp> {
  using OpConversionPatternWithAnalysis<
      arith::AddIOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::Value result = op.getResult();

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    rewriter.setInsertionPoint(op);
    auto tempLocalOp =
        rewriter.create<wasm::TempLocalOp>(op->getLoc(), result.getType());

    auto localType =
        mlir::wasm::LocalType::get(op->getContext(), result.getType());
    auto lhsCastOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), localType, lhs);
    rewriter.create<wasm::TempLocalGetOp>(op->getLoc(), lhsCastOp.getResult(0));
    auto rhsCastOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), localType, rhs);
    rewriter.create<wasm::TempLocalGetOp>(op->getLoc(), rhsCastOp.getResult(0));
    // TODO: Verify somewhere that two locals are of same type
    rewriter.create<wasm::AddOp>(op->getLoc(), lhs.getType());
    rewriter.create<wasm::TempLocalSetOp>(op->getLoc(),
                                          tempLocalOp.getResult());

    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), result.getType(), tempLocalOp.getResult());
    rewriter.clearInsertionPoint();

    rewriter.replaceOp(op, castOp);

    return success();
  }
};

struct ConvertConstant
    : public OpConversionPatternWithAnalysis<arith::ConstantOp> {
  using OpConversionPatternWithAnalysis<
      arith::ConstantOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::Value result = op.getResult();

    mlir::Attribute attr = op->getAttr("value");

    rewriter.setInsertionPoint(op);
    auto tempLocalOp =
        rewriter.create<wasm::TempLocalOp>(op->getLoc(), result.getType());
    rewriter.create<wasm::ConstantOp>(op->getLoc(), attr);
    rewriter.create<wasm::TempLocalSetOp>(op->getLoc(),
                                          tempLocalOp.getResult());
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), result.getType(), tempLocalOp.getResult());
    rewriter.clearInsertionPoint();

    rewriter.replaceOp(op, castOp);

    return success();
  }
};

class ConvertToWasm : public impl::ConvertToWasmBase<ConvertToWasm> {
public:
  using impl::ConvertToWasmBase<ConvertToWasm>::ConvertToWasmBase;

  void runOnOperation() final {

    VariableAnalysis &analysis = getAnalysis<VariableAnalysis>();

    FuncOp func = getOperation();
    MLIRContext *context = func.getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertAdd, ConvertConstant>(context, analysis);

    PatternRewriter rewriter(context);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::wasm
