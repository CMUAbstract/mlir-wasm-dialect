//===- RemainderToAnd.cpp - Remainder to AND pass -----------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion of remainder by power-of-2 to bitwise AND.
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_REMAINDERTOAND
#include "Transforms/TransformsPasses.h.inc"

class RemainderToAnd : public impl::RemainderToAndBase<RemainderToAnd> {
public:
  using impl::RemainderToAndBase<RemainderToAnd>::RemainderToAndBase;

  void runOnOperation() final {
    auto module = getOperation();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();

    IRRewriter rewriter(module.getContext());

    // Collect candidates first to avoid invalidating the walk.
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (isa<arith::RemUIOp, arith::RemSIOp>(op))
        candidates.push_back(op);
    });

    for (auto *op : candidates) {
      Value rhs = op->getOperand(1);
      auto constOp = rhs.getDefiningOp<arith::ConstantOp>();
      if (!constOp)
        continue;
      auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (!intAttr)
        continue;

      auto divisor = intAttr.getValue();
      // Must be power of 2 and > 1 (leave rem x, 1 for existing folds).
      if (!divisor.isPowerOf2() || divisor.isOne())
        continue;

      Value lhs = op->getOperand(0);

      // For remsi, LHS must be provably non-negative.
      if (isa<arith::RemSIOp>(op)) {
        if (failed(dataflow::staticallyNonNegative(solver, lhs)))
          continue;
      }

      // Replace: rem(lhs, 2^k) -> and(lhs, 2^k - 1)
      rewriter.setInsertionPoint(op);
      Location loc = op->getLoc();
      Type type = op->getResult(0).getType();

      auto mask = divisor - 1;
      Value maskConst = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(type, mask));
      Value andResult = arith::AndIOp::create(rewriter, loc, lhs, maskConst);

      rewriter.replaceAllUsesWith(op->getResult(0), andResult);
      rewriter.eraseOp(op);
    }
  }
};

} // namespace mlir::transforms
