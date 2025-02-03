//===- SsaWasmPasses.cpp - SsaWasm passes -----------------*- C++ -*-===//
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "SsaWasm/ConversionPatterns/ArithToSsaWasm.h"
#include "SsaWasm/ConversionPatterns/FuncToSsaWasm.h"
#include "SsaWasm/SsaWasmPasses.h"
#include "SsaWasm/SsaWasmTypeConverter.h"
#include <vector>

using namespace std;

namespace mlir::ssawasm {
#define GEN_PASS_DEF_CONVERTTOSSAWASM
#define GEN_PASS_DEF_REPLACEMULTIUSEOPSWITHLOCALS
#define GEN_PASS_DEF_CONVERTSSAWASMTOWASM
#include "SsaWasm/SsaWasmPasses.h.inc"

class ConvertToSsaWasm : public impl::ConvertToSsaWasmBase<ConvertToSsaWasm> {
public:
  using impl::ConvertToSsaWasmBase<ConvertToSsaWasm>::ConvertToSsaWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<func::FuncDialect>();

    RewritePatternSet patterns(context);
    SsaWasmTypeConverter typeConverter(context);
    populateArithToSsaWasmPatterns(typeConverter, patterns);
    populateFuncToSsaWasmPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

namespace {
struct MultiUseOpLowering : public RewritePattern {
  MultiUseOpLowering(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult match(Operation *op) const override {
    assert(op->getDialect() ==
           op->getContext()->getLoadedDialect<ssawasm::SsaWasmDialect>());
    if (isa<LocalOp>(op)) {
      return failure();
    }
    if (std::distance(op->getUsers().begin(), op->getUsers().end()) > 1) {
      return success();
    }
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    auto localOp = rewriter.create<LocalOp>(op->getLoc(), op->getResult(0));
    // we assume that the op has one operand
    rewriter.replaceAllUsesExcept(op->getResult(0), localOp.getResult(),
                                  localOp);
  }
};
} // namespace

class ReplaceMultiUseOpsWithLocals
    : public impl::ReplaceMultiUseOpsWithLocalsBase<
          ReplaceMultiUseOpsWithLocals> {
public:
  using impl::ReplaceMultiUseOpsWithLocalsBase<
      ReplaceMultiUseOpsWithLocals>::ReplaceMultiUseOpsWithLocalsBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder builder(context);

    RewritePatternSet patterns(context);
    patterns.add<MultiUseOpLowering>(context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class Stackify : public impl::StackifyBase<Stackify> {
public:
  using impl::StackifyBase<Stackify>::StackifyBase;

  void runOnOperation() final {
    auto module = getOperation();
    // TODO
  }
};
} // namespace mlir::ssawasm
