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

namespace mlir::wasm {
#define GEN_PASS_DEF_CONVERTTOWASM
#include "Wasm/WasmPasses.h.inc"

struct ConvertAdd : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

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

struct ConvertConstant : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

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
    func::FuncOp func = getOperation();
    MLIRContext *context = func.getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertAdd, ConvertConstant>(context);

    PatternRewriter rewriter(context);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::wasm
