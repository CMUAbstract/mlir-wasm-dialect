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
  /// Check if v is the induction variable or a (chain of) cast(s) of it.
  static bool isIVOrCast(Value v, scf::ForOp forOp) {
    Value iv = forOp.getInductionVar();
    if (v == iv)
      return true;
    if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
      return castOp.getInputs().size() == 1 &&
             isIVOrCast(castOp.getInputs()[0], forOp);
    if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
      return isIVOrCast(castOp.getIn(), forOp);
    if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
      return isIVOrCast(extOp.getIn(), forOp);
    if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
      return isIVOrCast(extOp.getIn(), forOp);
    return false;
  }

  /// Recreate the cast chain from exampleCast, applied to indexVal.
  /// Returns indexVal unchanged if no cast is needed (types already match).
  static Value createIndexCast(OpBuilder &builder, Location loc, Value indexVal,
                               Type targetType, Value exampleCast) {
    if (indexVal.getType() == targetType)
      return indexVal;
    // Handle extension ops by recursively creating the inner cast first.
    if (auto extOp = exampleCast.getDefiningOp<arith::ExtSIOp>()) {
      Value inner = createIndexCast(builder, loc, indexVal,
                                    extOp.getIn().getType(), extOp.getIn());
      return arith::ExtSIOp::create(builder, loc, targetType, inner);
    }
    if (auto extOp = exampleCast.getDefiningOp<arith::ExtUIOp>()) {
      Value inner = createIndexCast(builder, loc, indexVal,
                                    extOp.getIn().getType(), extOp.getIn());
      return arith::ExtUIOp::create(builder, loc, targetType, inner);
    }
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

  /// Unwrap cast chains (index_cast, extsi, extui, unrealized_conversion_cast)
  /// to find the innermost non-cast value.
  static Value unwrapCasts(Value v) {
    while (true) {
      if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() == 1) {
          v = castOp.getInputs()[0];
          continue;
        }
      }
      if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
        v = castOp.getIn();
        continue;
      }
      if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
        v = extOp.getIn();
        continue;
      }
      if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
        v = extOp.getIn();
        continue;
      }
      return v;
    }
  }

  /// Phase 0: Distribute multiply/shift over IV addition/subtraction.
  ///
  /// Rewrites:
  ///   muli(castChain(addi(iv, k)), factor)
  ///     → addi(muli(castChain(iv), factor), muli(castChain(k), factor))
  ///
  /// After this, the inner muli becomes a Phase 1 candidate and the
  /// wrapping addi becomes a Phase 2 candidate.
  void distributeMultiplyOverIVAdd(IRRewriter &rewriter, scf::ForOp forOp) {
    SmallVector<Operation *> mulOps;
    for (auto &op : forOp.getBody()->getOperations())
      if (isa<arith::MulIOp, arith::ShLIOp>(&op))
        mulOps.push_back(&op);

    for (auto *op : mulOps) {
      Value ivSide, factorSide;
      bool isShift = false;

      if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        Value lhs = mulOp.getLhs(), rhs = mulOp.getRhs();
        if (forOp.isDefinedOutsideOfLoop(rhs)) {
          ivSide = lhs;
          factorSide = rhs;
        } else if (forOp.isDefinedOutsideOfLoop(lhs)) {
          ivSide = rhs;
          factorSide = lhs;
        } else {
          continue;
        }
      } else if (auto shlOp = dyn_cast<arith::ShLIOp>(op)) {
        Value rhs = shlOp.getRhs();
        if (!rhs.getDefiningOp<arith::ConstantOp>() ||
            !forOp.isDefinedOutsideOfLoop(rhs))
          continue;
        ivSide = shlOp.getLhs();
        factorSide = rhs;
        isShift = true;
      } else {
        continue;
      }

      // Skip if Phase 1 already handles this (direct iv * factor).
      if (isIVOrCast(ivSide, forOp))
        continue;

      // Unwrap cast chain to find inner addi/subi.
      Value inner = unwrapCasts(ivSide);
      Value ivPart, kPart;
      bool isSub = false;

      if (auto addOp = inner.getDefiningOp<arith::AddIOp>()) {
        Value lhs = addOp.getLhs(), rhs = addOp.getRhs();
        if (isIVOrCast(lhs, forOp) && forOp.isDefinedOutsideOfLoop(rhs)) {
          ivPart = lhs;
          kPart = rhs;
        } else if (isIVOrCast(rhs, forOp) &&
                   forOp.isDefinedOutsideOfLoop(lhs)) {
          ivPart = rhs;
          kPart = lhs;
        } else {
          continue;
        }
      } else if (auto subOp = inner.getDefiningOp<arith::SubIOp>()) {
        Value lhs = subOp.getLhs(), rhs = subOp.getRhs();
        if (isIVOrCast(lhs, forOp) && forOp.isDefinedOutsideOfLoop(rhs)) {
          ivPart = lhs;
          kPart = rhs;
          isSub = true;
        } else {
          continue;
        }
      } else {
        continue;
      }

      Location loc = op->getLoc();
      Type mulType = ivSide.getType();

      // Compute k * factor before the loop (loop-invariant).
      rewriter.setInsertionPoint(forOp);
      Value kCasted = createIndexCast(rewriter, loc, kPart, mulType, ivSide);
      Value kProduct;
      if (isShift)
        kProduct = arith::ShLIOp::create(rewriter, loc, kCasted, factorSide);
      else
        kProduct = arith::MulIOp::create(rewriter, loc, kCasted, factorSide);

      if (isSub) {
        Value zero =
            arith::ConstantOp::create(rewriter, loc, kProduct.getType(),
                                      rewriter.getZeroAttr(kProduct.getType()));
        kProduct = arith::SubIOp::create(rewriter, loc, zero, kProduct);
      }

      // Create iv * factor + k_product in the loop body.
      rewriter.setInsertionPoint(op);
      Value ivCasted = createIndexCast(rewriter, loc, ivPart, mulType, ivSide);
      Value ivProduct;
      if (isShift)
        ivProduct = arith::ShLIOp::create(rewriter, loc, ivCasted, factorSide);
      else
        ivProduct = arith::MulIOp::create(rewriter, loc, ivCasted, factorSide);

      Value result = arith::AddIOp::create(rewriter, loc, ivProduct, kProduct);

      rewriter.replaceAllUsesWith(op->getResult(0), result);
      rewriter.eraseOp(op);
    }
  }

  /// Fold chains of addi with loop-invariant operands.
  ///
  /// Rewrites:
  ///   addi(addi(variant, inv1), inv2) → addi(variant, addi(inv1, inv2))
  ///
  /// The combined invariant is hoisted before the loop, so Phase 2 sees
  /// a single addi(muli_result, combined_offset) and absorbs it into
  /// the accumulator init.
  void foldInvariantAddiChains(IRRewriter &rewriter, scf::ForOp forOp) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &op : forOp.getBody()->getOperations()) {
        auto addOp = dyn_cast<arith::AddIOp>(&op);
        if (!addOp)
          continue;

        Value lhs = addOp.getLhs(), rhs = addOp.getRhs();

        // One side must be loop-invariant.
        Value outerInv, addiResult;
        if (forOp.isDefinedOutsideOfLoop(rhs)) {
          outerInv = rhs;
          addiResult = lhs;
        } else if (forOp.isDefinedOutsideOfLoop(lhs)) {
          outerInv = lhs;
          addiResult = rhs;
        } else {
          continue;
        }

        // The other side must be an addi with one loop-invariant operand.
        auto innerAdd = addiResult.getDefiningOp<arith::AddIOp>();
        if (!innerAdd)
          continue;

        Value iLhs = innerAdd.getLhs(), iRhs = innerAdd.getRhs();
        Value innerInv, variant;
        if (forOp.isDefinedOutsideOfLoop(iRhs) &&
            !forOp.isDefinedOutsideOfLoop(iLhs)) {
          innerInv = iRhs;
          variant = iLhs;
        } else if (forOp.isDefinedOutsideOfLoop(iLhs) &&
                   !forOp.isDefinedOutsideOfLoop(iRhs)) {
          innerInv = iLhs;
          variant = iRhs;
        } else {
          continue;
        }

        // Hoist: combined = addi(innerInv, outerInv)
        Location loc = addOp.getLoc();
        rewriter.setInsertionPoint(forOp);
        Value combined =
            arith::AddIOp::create(rewriter, loc, innerInv, outerInv);

        // Replace: addi(addi(variant, inv1), inv2) → addi(variant, combined)
        rewriter.setInsertionPoint(addOp);
        Value newAdd = arith::AddIOp::create(rewriter, loc, variant, combined);
        rewriter.replaceAllUsesWith(addOp.getResult(), newAdd);
        rewriter.eraseOp(addOp);

        if (innerAdd->use_empty())
          rewriter.eraseOp(innerAdd);

        changed = true;
        break;
      }
    }
  }

  void strengthReduceLoop(IRRewriter &rewriter, scf::ForOp forOp) {
    distributeMultiplyOverIVAdd(rewriter, forOp);
    foldInvariantAddiChains(rewriter, forOp);

    struct Candidate {
      Operation *op;            // MulIOp, ShLIOp, or AddIOp to replace
      Value ivOrCast;           // IV or cast of IV
      Value factor;             // loop-invariant factor (null for shli)
      int64_t shiftAmount = -1; // for shli: bit count
      Value offset;             // loop-invariant addend (null if no addi)
    };

    // Phase 1: Detect muli/shli candidates.
    SmallVector<Candidate> mulCandidates;
    DenseMap<Operation *, unsigned> mulCandidateMap;

    for (auto &op : forOp.getBody()->getOperations()) {
      if (auto mulOp = dyn_cast<arith::MulIOp>(&op)) {
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        if (isIVOrCast(lhs, forOp) && forOp.isDefinedOutsideOfLoop(rhs)) {
          mulCandidateMap[mulOp] = mulCandidates.size();
          mulCandidates.push_back({mulOp, lhs, rhs, -1, Value()});
        } else if (isIVOrCast(rhs, forOp) &&
                   forOp.isDefinedOutsideOfLoop(lhs)) {
          mulCandidateMap[mulOp] = mulCandidates.size();
          mulCandidates.push_back({mulOp, rhs, lhs, -1, Value()});
        }
        continue;
      }
      if (auto shlOp = dyn_cast<arith::ShLIOp>(&op)) {
        Value lhs = shlOp.getLhs();
        Value rhs = shlOp.getRhs();
        if (isIVOrCast(lhs, forOp)) {
          if (auto constOp = rhs.getDefiningOp<arith::ConstantOp>())
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
              mulCandidateMap[shlOp] = mulCandidates.size();
              mulCandidates.push_back({shlOp, lhs, Value(),
                                       intAttr.getValue().getSExtValue(),
                                       Value()});
            }
        }
        continue;
      }
    }

    // Phase 2: Detect addi candidates wrapping muli/shli results.
    SmallVector<Candidate> addiCandidates;
    DenseMap<Operation *, unsigned> addiUseCount;
    for (auto &mc : mulCandidates)
      addiUseCount[mc.op] = 0;

    for (auto &op : forOp.getBody()->getOperations()) {
      auto addOp = dyn_cast<arith::AddIOp>(&op);
      if (!addOp)
        continue;

      Value lhs = addOp.getLhs();
      Value rhs = addOp.getRhs();
      Operation *mulDef = nullptr;
      Value offset;

      Operation *lhsDef = lhs.getDefiningOp();
      Operation *rhsDef = rhs.getDefiningOp();

      if (lhsDef && mulCandidateMap.count(lhsDef) &&
          forOp.isDefinedOutsideOfLoop(rhs)) {
        mulDef = lhsDef;
        offset = rhs;
      } else if (rhsDef && mulCandidateMap.count(rhsDef) &&
                 forOp.isDefinedOutsideOfLoop(lhs)) {
        mulDef = rhsDef;
        offset = lhs;
      }

      if (mulDef) {
        unsigned idx = mulCandidateMap[mulDef];
        auto &mc = mulCandidates[idx];
        addiCandidates.push_back(
            {addOp, mc.ivOrCast, mc.factor, mc.shiftAmount, offset});
        addiUseCount[mulDef]++;
      }
    }

    // Phase 3: Determine which mulCandidates are fully absorbed by addis.
    SmallVector<Operation *> absorbedOps;
    SmallVector<Candidate> candidates;

    for (auto &mc : mulCandidates) {
      unsigned totalUses = 0;
      for (auto &use : mc.op->getResult(0).getUses())
        (void)use, totalUses++;

      if (addiUseCount[mc.op] > 0 && addiUseCount[mc.op] == totalUses)
        absorbedOps.push_back(mc.op);
      else
        candidates.push_back(mc);
    }

    candidates.append(addiCandidates.begin(), addiCandidates.end());

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
      Type resultType = c.op->getResult(0).getType();

      // Get or materialize the loop-invariant factor.
      Value factor;
      if (c.factor) {
        factor = c.factor;
      } else {
        // shli: factor = 1 << shiftAmount
        int64_t factorVal = 1LL << c.shiftAmount;
        factor = arith::ConstantOp::create(
            rewriter, loc, resultType,
            rewriter.getIntegerAttr(resultType, factorVal));
      }

      // init = cast(lb) * factor  (if lb==0 → just 0)
      Value init;
      if (isConstantZero(lb)) {
        init = arith::ConstantOp::create(rewriter, loc, resultType,
                                         rewriter.getZeroAttr(resultType));
      } else {
        Value lbCast =
            createIndexCast(rewriter, loc, lb, resultType, c.ivOrCast);
        init = arith::MulIOp::create(rewriter, loc, lbCast, factor);
      }

      // Add offset to init if present.
      if (c.offset) {
        if (isConstantZero(lb))
          init = c.offset; // 0 * factor + offset = offset
        else
          init = arith::AddIOp::create(rewriter, loc, init, c.offset);
      }

      // increment = cast(step) * factor  (if step==1 → just factor)
      Value increment;
      if (isConstantOne(step)) {
        increment = factor;
      } else {
        Value stepCast =
            createIndexCast(rewriter, loc, step, resultType, c.ivOrCast);
        increment = arith::MulIOp::create(rewriter, loc, stepCast, factor);
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

    // Replace candidate uses with accumulator block args and erase dead ops.
    for (auto [i, c] : llvm::enumerate(candidates)) {
      BlockArgument acc = newForOp.getRegionIterArgs()[numOrigIterArgs + i];
      rewriter.replaceAllUsesWith(c.op->getResult(0), acc);
      rewriter.eraseOp(c.op);
    }

    // Erase absorbed muli/shli ops (all their uses were removed above).
    for (auto *op : absorbedOps)
      rewriter.eraseOp(op);
  }
};

} // namespace mlir::transforms
