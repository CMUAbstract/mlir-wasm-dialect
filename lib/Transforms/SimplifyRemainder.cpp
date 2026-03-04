//===- SimplifyRemainder.cpp - Simplify remainder ops ------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file simplifies remainder operations using integer range analysis.
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_SIMPLIFYREMAINDER
#include "Transforms/TransformsPasses.h.inc"

class SimplifyRemainder
    : public impl::SimplifyRemainderBase<SimplifyRemainder> {
public:
  using impl::SimplifyRemainderBase<SimplifyRemainder>::SimplifyRemainderBase;

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
      // Leave rem x, 1 for existing constant folds.
      if (divisor.isOne())
        continue;

      Value lhs = op->getOperand(0);
      rewriter.setInsertionPoint(op);
      Location loc = op->getLoc();
      Type type = op->getResult(0).getType();

      // Case 1: Power-of-2 AND (remui always, remsi when non-negative).
      if (divisor.isPowerOf2()) {
        if (isa<arith::RemSIOp>(op)) {
          if (failed(dataflow::staticallyNonNegative(solver, lhs)))
            continue;
        }
        auto mask = divisor - 1;
        Value maskConst = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(type, mask));
        Value andResult = arith::AndIOp::create(rewriter, loc, lhs, maskConst);
        rewriter.replaceAllUsesWith(op->getResult(0), andResult);
        rewriter.eraseOp(op);
        continue;
      }

      // Remaining cases only apply to remsi with non-power-of-2 divisors.
      if (!isa<arith::RemSIOp>(op))
        continue;

      // Query the full signed range of LHS.
      auto *rangeState =
          solver.lookupState<dataflow::IntegerValueRangeLattice>(lhs);
      if (!rangeState || rangeState->getValue().isUninitialized())
        continue;
      const auto &range = rangeState->getValue().getValue();
      int64_t smin = range.smin().getSExtValue();
      int64_t smax = range.smax().getSExtValue();
      int64_t N = intAttr.getValue().getSExtValue();

      if (N <= 0)
        continue;

      if (smin >= 0 && smax < N) {
        // Case 2: Eliminate — rem is a no-op.
        rewriter.replaceAllUsesWith(op->getResult(0), lhs);
        rewriter.eraseOp(op);
      } else if (smin >= 0 && smax < 2 * N) {
        // Case 3: Select — conditional subtraction.
        auto cmp = arith::CmpIOp::create(rewriter, loc,
                                         arith::CmpIPredicate::slt, lhs, rhs);
        auto sub = arith::SubIOp::create(rewriter, loc, lhs, rhs);
        auto sel = arith::SelectOp::create(rewriter, loc, cmp, lhs, sub);
        rewriter.replaceAllUsesWith(op->getResult(0), sel.getResult());
        rewriter.eraseOp(op);
      } else if (smin >= 0) {
        // Case 4: Signed → unsigned.
        auto remui = arith::RemUIOp::create(rewriter, loc, lhs, rhs);
        rewriter.replaceAllUsesWith(op->getResult(0), remui.getResult());
        rewriter.eraseOp(op);
      }
    }
  }
};

} // namespace mlir::transforms
