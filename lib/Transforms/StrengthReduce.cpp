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

#include "llvm/ADT/DenseSet.h"
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
  /// A muli or shli of the IV with a loop-invariant factor.
  struct Candidate {
    Operation *op;            // MulIOp or ShLIOp
    Value ivOrCast;           // IV or cast chain of IV
    Value factor;             // loop-invariant factor (null for shli)
    int64_t shiftAmount = -1; // for shli: bit count
  };

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

  /// Check if all uses of the IV lead only to ops in erasedOps
  /// (transitively through any intermediate ops).  If so, the IV becomes
  /// dead after those ops are erased — no register pressure increase.
  static bool
  isIVDeadAfterErasure(Value iv, const llvm::DenseSet<Operation *> &erasedOps) {
    llvm::DenseSet<Value> visited;
    std::function<bool(Value)> allUsesCovered = [&](Value v) -> bool {
      if (!visited.insert(v).second)
        return true; // already verified or in progress — treat as covered
      for (auto &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (erasedOps.contains(user))
          continue;
        // If all of this user's results are transitively covered, the
        // user is dead once erasedOps are removed.  Guard on
        // getNumResults() > 0 so terminators and side-effecting void
        // ops (e.g. func.call) aren't skipped.
        if (user->getNumResults() > 0 &&
            llvm::all_of(user->getResults(),
                         [&](Value r) { return allUsesCovered(r); }))
          continue;
        return false; // uncovered use
      }
      return true;
    };
    return allUsesCovered(iv);
  }

  /// Recursively decompose `val` as `IV + offset` where offset is
  /// loop-invariant.  Returns the loop-invariant offset, or a null Value
  /// if `val` cannot be expressed as IV + (loop-invariant).
  /// Materializes offset arithmetic before `forOp` — all created values
  /// are loop-invariant.
  Value decomposeAsIVPlusOffset(Value val, scf::ForOp forOp,
                                IRRewriter &rewriter) {
    Value iv = forOp.getInductionVar();

    // Base case: val IS the induction variable → offset = 0.
    if (val == iv) {
      rewriter.setInsertionPoint(forOp);
      return arith::ConstantOp::create(rewriter, forOp.getLoc(),
                                       rewriter.getIndexType(),
                                       rewriter.getIndexAttr(0));
    }

    // If val is entirely loop-invariant, it has no IV component.
    if (forOp.isDefinedOutsideOfLoop(val))
      return Value();

    // addi(A, B): one side must decompose, the other must be invariant.
    if (auto addOp = val.getDefiningOp<arith::AddIOp>()) {
      Value lhs = addOp.getLhs(), rhs = addOp.getRhs();
      if (forOp.isDefinedOutsideOfLoop(rhs)) {
        if (Value lhsOff = decomposeAsIVPlusOffset(lhs, forOp, rewriter)) {
          rewriter.setInsertionPoint(forOp);
          return arith::AddIOp::create(rewriter, addOp.getLoc(), lhsOff, rhs);
        }
      }
      if (forOp.isDefinedOutsideOfLoop(lhs)) {
        if (Value rhsOff = decomposeAsIVPlusOffset(rhs, forOp, rewriter)) {
          rewriter.setInsertionPoint(forOp);
          return arith::AddIOp::create(rewriter, addOp.getLoc(), lhs, rhsOff);
        }
      }
    }

    // subi(A, B): only A can decompose (B must be invariant).
    // subi(inv, IV+off) would give a negative IV coefficient — skip.
    if (auto subOp = val.getDefiningOp<arith::SubIOp>()) {
      Value lhs = subOp.getLhs(), rhs = subOp.getRhs();
      if (forOp.isDefinedOutsideOfLoop(rhs)) {
        if (Value lhsOff = decomposeAsIVPlusOffset(lhs, forOp, rewriter)) {
          rewriter.setInsertionPoint(forOp);
          return arith::SubIOp::create(rewriter, subOp.getLoc(), lhsOff, rhs);
        }
      }
    }

    return Value();
  }

  /// Phase 0: Distribute multiply/shift over IV expressions.
  ///
  /// Uses recursive decomposition to handle arbitrary nesting of
  /// addi/subi around the induction variable.  Rewrites:
  ///   muli(castChain(IV_expr), factor)
  ///     → addi(muli(castChain(iv), factor), muli(castChain(offset), factor))
  /// where IV_expr = iv + offset (with offset loop-invariant).
  ///
  /// After this, the inner muli becomes a Phase 1 candidate and the
  /// wrapping addi becomes a Phase 2 candidate.
  void distributeMultiplyOverIVExpr(IRRewriter &rewriter, scf::ForOp forOp) {
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

      // Unwrap cast chain to find inner expression.
      Value inner = unwrapCasts(ivSide);

      // Recursively decompose as IV + offset.
      Value offset = decomposeAsIVPlusOffset(inner, forOp, rewriter);
      if (!offset)
        continue;

      Location loc = op->getLoc();
      Type mulType = ivSide.getType();
      Value iv = forOp.getInductionVar();

      // Compute offset * factor before the loop (loop-invariant).
      rewriter.setInsertionPoint(forOp);
      Value offsetCasted =
          createIndexCast(rewriter, loc, offset, mulType, ivSide);
      Value offsetProduct;
      if (isShift)
        offsetProduct =
            arith::ShLIOp::create(rewriter, loc, offsetCasted, factorSide);
      else
        offsetProduct =
            arith::MulIOp::create(rewriter, loc, offsetCasted, factorSide);

      // Create iv * factor + offset_product in the loop body.
      rewriter.setInsertionPoint(op);
      Value ivCasted = createIndexCast(rewriter, loc, iv, mulType, ivSide);
      Value ivProduct;
      if (isShift)
        ivProduct = arith::ShLIOp::create(rewriter, loc, ivCasted, factorSide);
      else
        ivProduct = arith::MulIOp::create(rewriter, loc, ivCasted, factorSide);

      Value result =
          arith::AddIOp::create(rewriter, loc, ivProduct, offsetProduct);

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

  /// Try bounds-based reduction: fold the multiply factor into the loop
  /// bounds, eliminating per-iteration multiplies without new iter_args.
  /// Requires all candidates to share the same factor and the IV to be dead.
  bool tryBoundsBasedReduction(IRRewriter &rewriter, scf::ForOp forOp,
                               SmallVector<Candidate> &candidates) {
    // Same-factor check — all candidates must share the same factor.
    bool allMuli =
        llvm::all_of(candidates, [](const Candidate &c) { return !!c.factor; });
    bool allShli = llvm::all_of(
        candidates, [](const Candidate &c) { return c.shiftAmount >= 0; });

    if (!allMuli && !allShli)
      return false;

    if (allMuli) {
      Value firstFactor = candidates[0].factor;
      if (!llvm::all_of(candidates, [&](const Candidate &c) {
            return c.factor == firstFactor;
          }))
        return false;
    } else {
      int64_t firstShift = candidates[0].shiftAmount;
      if (!llvm::all_of(candidates, [&](const Candidate &c) {
            return c.shiftAmount == firstShift;
          }))
        return false;
    }

    // IV-dead check (correctness requirement for bounds transformation).
    llvm::DenseSet<Operation *> erasedOps;
    for (auto &c : candidates)
      erasedOps.insert(c.op);
    if (!isIVDeadAfterErasure(forOp.getInductionVar(), erasedOps))
      return false;

    // Bounds transformation: fold the factor into lb/ub/step.
    rewriter.setInsertionPoint(forOp);
    Location loc = forOp.getLoc();
    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();

    // Materialize factor as index type.
    Value factorIndex;
    if (allMuli) {
      Value factor = candidates[0].factor;
      if (factor.getType().isIndex())
        factorIndex = factor;
      else
        factorIndex = arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), factor);
    } else {
      int64_t factorVal = 1LL << candidates[0].shiftAmount;
      factorIndex =
          arith::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                    rewriter.getIndexAttr(factorVal));
    }

    // Compute new bounds: lb*K, ub*K, step*K.
    Value newLb = lb;
    if (!isConstantZero(lb))
      newLb = arith::MulIOp::create(rewriter, loc, lb, factorIndex);

    Value newUb = arith::MulIOp::create(rewriter, loc, ub, factorIndex);

    Value newStep;
    if (isConstantOne(step))
      newStep = factorIndex;
    else
      newStep = arith::MulIOp::create(rewriter, loc, step, factorIndex);

    // Modify loop bounds in-place.
    forOp->setOperand(0, newLb);
    forOp->setOperand(1, newUb);
    forOp->setOperand(2, newStep);

    // Replace each muli/shli result with its ivOrCast operand.
    for (auto &c : candidates) {
      rewriter.replaceAllUsesWith(c.op->getResult(0), c.ivOrCast);
      rewriter.eraseOp(c.op);
    }

    return true;
  }

  /// Try to absorb a common additive offset into the loop bounds.
  /// After bounds-based reduction folds muli into bounds, patterns like
  ///   addi(cast(iv), base)
  /// remain in the loop body.  This eliminates the per-iteration addi by
  /// shifting lb and ub by the offset.
  bool tryAbsorbCommonOffset(IRRewriter &rewriter, scf::ForOp forOp) {
    // Clean up dead ops first (e.g., stale addi/cast chains left behind
    // by the distribution phase).  Reverse iteration handles chains:
    // erasing a dead user makes its producer dead in the same pass.
    for (Operation &op : llvm::make_early_inc_range(
             llvm::reverse(forOp.getBody()->getOperations()))) {
      if (!op.use_empty())
        continue;
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
              arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
              arith::ConstantOp, UnrealizedConversionCastOp>(&op))
        rewriter.eraseOp(&op);
    }

    // Collect addi candidates: addi(ivOrCast, loop_invariant).
    // Only accept IV, index_cast, or unrealized_conversion_cast chains —
    // NOT extsi/extui, because addition does not distribute over
    // sign/zero extension (e.g., extsi(trunc(iv+off)) != extsi(trunc(iv))+off).
    auto isSimpleIVOrCast = [&](Value v) -> bool {
      Value iv = forOp.getInductionVar();
      while (true) {
        if (v == iv)
          return true;
        if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
          v = castOp.getIn();
          continue;
        }
        if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp.getInputs().size() == 1) {
            v = castOp.getInputs()[0];
            continue;
          }
        }
        return false;
      }
    };

    struct OffsetCandidate {
      arith::AddIOp addOp;
      Value ivOrCast;
      Value offset;
    };
    SmallVector<OffsetCandidate> candidates;

    for (auto &op : forOp.getBody()->getOperations()) {
      auto addOp = dyn_cast<arith::AddIOp>(&op);
      if (!addOp)
        continue;
      // Skip dead addis (e.g., leftovers from earlier distribution).
      if (addOp.getResult().use_empty())
        continue;
      Value lhs = addOp.getLhs(), rhs = addOp.getRhs();
      if (isSimpleIVOrCast(lhs) && forOp.isDefinedOutsideOfLoop(rhs))
        candidates.push_back({addOp, lhs, rhs});
      else if (isSimpleIVOrCast(rhs) && forOp.isDefinedOutsideOfLoop(lhs))
        candidates.push_back({addOp, rhs, lhs});
    }

    if (candidates.empty())
      return false;

    // All addis must share the same loop-invariant offset.
    Value commonOffset = candidates[0].offset;
    if (!llvm::all_of(candidates, [&](const OffsetCandidate &c) {
          return c.offset == commonOffset;
        }))
      return false;

    // IV must become dead after erasing all addis.
    llvm::DenseSet<Operation *> erasedOps;
    for (auto &c : candidates)
      erasedOps.insert(c.addOp.getOperation());
    if (!isIVDeadAfterErasure(forOp.getInductionVar(), erasedOps))
      return false;

    // Absorb offset into bounds.
    rewriter.setInsertionPoint(forOp);
    Location loc = forOp.getLoc();
    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();

    Value offsetIndex;
    if (commonOffset.getType().isIndex())
      offsetIndex = commonOffset;
    else
      offsetIndex = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), commonOffset);

    Value newLb;
    if (isConstantZero(lb))
      newLb = offsetIndex;
    else
      newLb = arith::AddIOp::create(rewriter, loc, lb, offsetIndex);

    Value newUb = arith::AddIOp::create(rewriter, loc, ub, offsetIndex);

    forOp->setOperand(0, newLb);
    forOp->setOperand(1, newUb);

    // Replace each addi with its ivOrCast operand.
    for (auto &c : candidates) {
      rewriter.replaceAllUsesWith(c.addOp.getResult(), c.ivOrCast);
      rewriter.eraseOp(c.addOp);
    }

    return true;
  }

  /// Accumulator-based reduction: convert each muli(iv, K) into a
  /// loop-carried accumulator that increments by K*step each iteration.
  /// Fires when bounds-based reduction cannot apply (different factors
  /// or IV still alive).
  bool accumulatorReduceLoop(IRRewriter &rewriter, scf::ForOp forOp,
                             SmallVector<Candidate> &candidates) {
    // Filter out candidates with no uses — adding an accumulator for a
    // dead multiply would be wasteful.
    llvm::erase_if(candidates, [](const Candidate &c) {
      return c.op->getResult(0).use_empty();
    });
    if (candidates.empty())
      return false;

    // Group candidates by factor.
    struct FactorGroup {
      SmallVector<size_t> candidateIndices;
      Value factor;
      int64_t shiftAmount;
      Value init;
      Value increment;
    };

    SmallVector<FactorGroup> groups;
    DenseMap<Value, size_t> muliFactorToGroup;
    DenseMap<int64_t, size_t> shliAmountToGroup;

    for (size_t i = 0; i < candidates.size(); ++i) {
      auto &c = candidates[i];
      if (c.factor) {
        auto it = muliFactorToGroup.find(c.factor);
        if (it != muliFactorToGroup.end()) {
          groups[it->second].candidateIndices.push_back(i);
        } else {
          muliFactorToGroup[c.factor] = groups.size();
          groups.push_back({{i}, c.factor, -1, Value(), Value()});
        }
      } else {
        auto it = shliAmountToGroup.find(c.shiftAmount);
        if (it != shliAmountToGroup.end()) {
          groups[it->second].candidateIndices.push_back(i);
        } else {
          shliAmountToGroup[c.shiftAmount] = groups.size();
          groups.push_back({{i}, Value(), c.shiftAmount, Value(), Value()});
        }
      }
    }

    // Verify all candidates in each group share the same result type.
    for (auto &group : groups) {
      Type expected =
          candidates[group.candidateIndices[0]].op->getResult(0).getType();
      for (size_t idx : group.candidateIndices)
        assert(candidates[idx].op->getResult(0).getType() == expected &&
               "all candidates in a factor group must share the same type");
    }

    // Register pressure check: only apply if root count does not increase.
    // delta = (roots added) - (roots removed).
    int delta = 0;
    for (auto &group : groups) {
      delta -= static_cast<int>(group.candidateIndices.size());
      delta += 1; // one accumulator per group
    }
    llvm::DenseSet<Operation *> erasedOps;
    for (auto &c : candidates)
      erasedOps.insert(c.op);
    if (isIVDeadAfterErasure(forOp.getInductionVar(), erasedOps))
      delta -= 1;
    if (delta > 0)
      return false;

    // Compute init/increment for each group.
    Location loc = forOp.getLoc();
    Value lb = forOp.getLowerBound();
    Value step = forOp.getStep();

    for (auto &group : groups) {
      auto &exampleCandidate = candidates[group.candidateIndices[0]];
      Type resultType = exampleCandidate.op->getResult(0).getType();
      Value exampleCast = exampleCandidate.ivOrCast;

      rewriter.setInsertionPoint(forOp);

      if (group.factor) {
        // muli group: init = cast(lb) * factor, incr = cast(step) * factor
        if (isConstantZero(lb)) {
          group.init = arith::ConstantOp::create(
              rewriter, loc, rewriter.getIntegerAttr(resultType, 0));
        } else {
          Value lbCasted =
              createIndexCast(rewriter, loc, lb, resultType, exampleCast);
          group.init =
              arith::MulIOp::create(rewriter, loc, lbCasted, group.factor);
        }

        if (isConstantOne(step)) {
          group.increment = group.factor;
        } else {
          Value stepCasted =
              createIndexCast(rewriter, loc, step, resultType, exampleCast);
          group.increment =
              arith::MulIOp::create(rewriter, loc, stepCasted, group.factor);
        }
      } else {
        // shli group: init = cast(lb) << N, incr = cast(step) << N
        if (isConstantZero(lb)) {
          group.init = arith::ConstantOp::create(
              rewriter, loc, rewriter.getIntegerAttr(resultType, 0));
        } else {
          Value lbCasted =
              createIndexCast(rewriter, loc, lb, resultType, exampleCast);
          Value shiftConst = arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getIntegerAttr(resultType, group.shiftAmount));
          group.init =
              arith::ShLIOp::create(rewriter, loc, lbCasted, shiftConst);
        }

        if (isConstantOne(step)) {
          group.increment = arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getIntegerAttr(resultType, 1LL << group.shiftAmount));
        } else {
          Value stepCasted =
              createIndexCast(rewriter, loc, step, resultType, exampleCast);
          Value shiftConst = arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getIntegerAttr(resultType, group.shiftAmount));
          group.increment =
              arith::ShLIOp::create(rewriter, loc, stepCasted, shiftConst);
        }
      }
    }

    // Collect init values.
    SmallVector<Value> initValues;
    for (auto &group : groups)
      initValues.push_back(group.init);

    // Add accumulators via replaceWithAdditionalYields.
    auto newYieldFn =
        [&](OpBuilder &b, Location yieldLoc,
            ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
      SmallVector<Value> yieldValues;
      for (auto [i, bbArg] : llvm::enumerate(newBbArgs)) {
        // Replace all multiply results in this group with the accumulator.
        for (size_t idx : groups[i].candidateIndices)
          rewriter.replaceAllUsesWith(candidates[idx].op->getResult(0), bbArg);
        // Yield: acc + increment.
        Value next =
            arith::AddIOp::create(b, yieldLoc, bbArg, groups[i].increment);
        yieldValues.push_back(next);
      }
      return yieldValues;
    };

    auto result = forOp.replaceWithAdditionalYields(
        rewriter, initValues, /*replaceInitOperandUsesInLoop=*/false,
        newYieldFn);
    if (failed(result))
      return false;

    // Erase dead multiply ops.
    for (auto &c : candidates) {
      if (c.op->use_empty())
        rewriter.eraseOp(c.op);
    }

    return true;
  }

  void strengthReduceLoop(IRRewriter &rewriter, scf::ForOp forOp) {
    distributeMultiplyOverIVExpr(rewriter, forOp);
    foldInvariantAddiChains(rewriter, forOp);

    // Phase 1: Detect muli/shli candidates.
    SmallVector<Candidate> candidates;

    for (auto &op : forOp.getBody()->getOperations()) {
      if (auto mulOp = dyn_cast<arith::MulIOp>(&op)) {
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        if (isIVOrCast(lhs, forOp) && forOp.isDefinedOutsideOfLoop(rhs))
          candidates.push_back({mulOp, lhs, rhs, -1});
        else if (isIVOrCast(rhs, forOp) && forOp.isDefinedOutsideOfLoop(lhs))
          candidates.push_back({mulOp, rhs, lhs, -1});
        continue;
      }
      if (auto shlOp = dyn_cast<arith::ShLIOp>(&op)) {
        Value lhs = shlOp.getLhs();
        Value rhs = shlOp.getRhs();
        if (isIVOrCast(lhs, forOp)) {
          if (auto constOp = rhs.getDefiningOp<arith::ConstantOp>())
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
              candidates.push_back(
                  {shlOp, lhs, Value(), intAttr.getValue().getSExtValue()});
        }
        continue;
      }
    }

    if (candidates.empty()) {
      tryAbsorbCommonOffset(rewriter, forOp);
      return;
    }

    // Try bounds-based reduction first (preferred: zero new iter_args).
    if (tryBoundsBasedReduction(rewriter, forOp, candidates)) {
      tryAbsorbCommonOffset(rewriter, forOp);
      return;
    }

    // Fallback: accumulator-based reduction.
    accumulatorReduceLoop(rewriter, forOp, candidates);
  }
};

} // namespace mlir::transforms
