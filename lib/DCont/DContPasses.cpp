
//===- DContPasses.cpp - DCont passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "DCont/DContPasses.h"
#include "Intermittent/IntermittentOps.h"

using namespace std;

namespace mlir::dcont {
#define GEN_PASS_DEF_CONVERTINTERMITTENTTODCONT
#include "DCont/DContPasses.h.inc"

namespace {
struct IdempotentTaskOpLowering
    : public OpConversionPattern<intermittent::IdempotentTaskOp> {
  using OpConversionPattern<
      intermittent::IdempotentTaskOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(intermittent::IdempotentTaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }
};

struct TransitionToOpLowering
    : public OpConversionPattern<intermittent::TransitionToOp> {
  using OpConversionPattern<intermittent::TransitionToOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(intermittent::TransitionToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }
};
} // namespace
class ConvertIntermittentToDCont
    : public impl::ConvertIntermittentToDContBase<ConvertIntermittentToDCont> {
  using impl::ConvertIntermittentToDContBase<
      ConvertIntermittentToDCont>::ConvertIntermittentToDContBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<IdempotentTaskOpLowering, TransitionToOpLowering>(context);

    ConversionTarget target(getContext());
    target.addLegalDialect<dcont::DContDialect>();
    target.addIllegalDialect<intermittent::IntermittentDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::dcont