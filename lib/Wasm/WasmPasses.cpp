//===- WasmPasses.cpp - Wasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Wasm/WasmPasses.h"

namespace mlir::wasm {
#define GEN_PASS_DEF_ARITHTOWASMPASS
#include "Wasm/WasmPasses.h.inc"

namespace {

class ConstantOpLowering : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    // rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.value());
    return success();
  }
};

void populateArithToWasmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConstantOpLowering>(patterns.getContext());
}

class ArithToWasmPass : public impl::ArithToWasmPassBase<ArithToWasmPass> {
public:
  using impl::ArithToWasmPassBase<ArithToWasmPass>::ArithToWasmPassBase;
  void runOnOperation() final {
    auto module = getOperation();
    RewritePatternSet patterns(&getContext());
    populateArithToWasmPatterns(patterns);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(module, patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::wasm
