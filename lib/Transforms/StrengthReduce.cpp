//===- StrengthReduce.cpp - Strength reduction pass --------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements strength reduction for induction variable multiplies
// in SCF loops.
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_STRENGTHREDUCE
#include "Transforms/TransformsPasses.h.inc"

class StrengthReduce : public impl::StrengthReduceBase<StrengthReduce> {
public:
  using impl::StrengthReduceBase<StrengthReduce>::StrengthReduceBase;

  void runOnOperation() final {
    auto module = getOperation();
    IRRewriter rewriter(module.getContext());

    // Collect all ForOps, then process inner-first (reverse of pre-order).
    SmallVector<scf::ForOp> forOps;
    module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (auto forOp : llvm::reverse(forOps))
      strengthReduceLoop(rewriter, forOp);
  }

private:
  /// Check if v is a cast of forOp's induction variable.
  static bool isIVCast(Value v, scf::ForOp forOp) {
    Value iv = forOp.getInductionVar();
    if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
      return castOp.getInputs().size() == 1 && castOp.getInputs()[0] == iv;
    if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
      return castOp.getIn() == iv;
    return false;
  }

  /// Create a cast of an index value, matching the kind of exampleCast.
  static Value createIndexCast(OpBuilder &builder, Location loc, Value indexVal,
                               Type targetType, Value exampleCast) {
    if (exampleCast.getDefiningOp<UnrealizedConversionCastOp>())
      return UnrealizedConversionCastOp::create(builder, loc, targetType,
                                                indexVal)
          .getResult(0);
    return arith::IndexCastOp::create(builder, loc, targetType, indexVal);
  }

  static bool isConstantZero(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return intAttr.getValue().isZero();
    return false;
  }

  static bool isConstantOne(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return intAttr.getValue().isOne();
    return false;
  }

  void strengthReduceLoop(IRRewriter &rewriter, scf::ForOp forOp) {
    struct Candidate {
      arith::MulIOp mulOp;
      Value ivCast;
      Value factor;
    };
    SmallVector<Candidate> candidates;

    for (auto &op : forOp.getBody()->getOperations()) {
      auto mulOp = dyn_cast<arith::MulIOp>(&op);
      if (!mulOp)
        continue;
      Value lhs = mulOp.getLhs();
      Value rhs = mulOp.getRhs();
      if (isIVCast(lhs, forOp) && forOp.isDefinedOutsideOfLoop(rhs))
        candidates.push_back({mulOp, lhs, rhs});
      else if (isIVCast(rhs, forOp) && forOp.isDefinedOutsideOfLoop(lhs))
        candidates.push_back({mulOp, rhs, lhs});
    }

    if (candidates.empty())
      return;

    // Compute init values and increments before the loop.
    rewriter.setInsertionPoint(forOp);
    Location loc = forOp.getLoc();
    Value lb = forOp.getLowerBound();
    Value step = forOp.getStep();

    SmallVector<Value> initValues;
    SmallVector<Value> increments;

    for (auto &c : candidates) {
      Type resultType = c.mulOp.getResult().getType();

      // init = cast(lb) * factor  (if lb==0 → just 0)
      Value init;
      if (isConstantZero(lb)) {
        init = arith::ConstantOp::create(rewriter, loc, resultType,
                                         rewriter.getZeroAttr(resultType));
      } else {
        Value lbCast = createIndexCast(rewriter, loc, lb, resultType, c.ivCast);
        init = arith::MulIOp::create(rewriter, loc, lbCast, c.factor);
      }

      // increment = cast(step) * factor  (if step==1 → just factor)
      Value increment;
      if (isConstantOne(step)) {
        increment = c.factor;
      } else {
        Value stepCast =
            createIndexCast(rewriter, loc, step, resultType, c.ivCast);
        increment = arith::MulIOp::create(rewriter, loc, stepCast, c.factor);
      }

      initValues.push_back(init);
      increments.push_back(increment);
    }

    // Add accumulator iter_args.
    unsigned numOrigIterArgs = forOp.getNumRegionIterArgs();
    auto newYieldFn =
        [&](OpBuilder &b, Location yieldLoc,
            ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
      SmallVector<Value> yieldValues;
      for (auto [bbArg, inc] : llvm::zip(newBbArgs, increments)) {
        Value next = arith::AddIOp::create(b, yieldLoc, bbArg, inc);
        yieldValues.push_back(next);
      }
      return yieldValues;
    };

    auto result = forOp.replaceWithAdditionalYields(
        rewriter, initValues, /*replaceInitOperandUsesInLoop=*/false,
        newYieldFn);
    if (failed(result))
      return;

    auto newForOp = cast<scf::ForOp>(*result);

    // Replace muli uses with accumulator block args and erase dead mulis.
    for (auto [i, c] : llvm::enumerate(candidates)) {
      BlockArgument acc = newForOp.getRegionIterArgs()[numOrigIterArgs + i];
      rewriter.replaceAllUsesWith(c.mulOp.getResult(), acc);
      rewriter.eraseOp(c.mulOp);
    }
  }
};

} // namespace mlir::transforms
