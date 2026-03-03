//===- ReassociateAdditions.cpp - Reassociate addi trees ----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements reassociation of arith.addi trees to expose common
// sub-expressions for CSE.
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_REASSOCIATEADDITIONS
#include "Transforms/TransformsPasses.h.inc"

class ReassociateAdditions
    : public impl::ReassociateAdditionsBase<ReassociateAdditions> {
public:
  using impl::ReassociateAdditionsBase<
      ReassociateAdditions>::ReassociateAdditionsBase;

  void runOnOperation() final {
    auto module = getOperation();
    IRRewriter rewriter(module.getContext());

    // Collect all ForOps, then process inner-first (reverse of pre-order).
    SmallVector<scf::ForOp> forOps;
    module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (auto forOp : llvm::reverse(forOps))
      reassociateLoop(rewriter, forOp);
  }

private:
  /// A candidate arith.addi in the loop body with its flattened terms.
  struct Candidate {
    arith::AddIOp op;
    SmallVector<Value> nonConstTerms; // sorted canonically
    int64_t constSum;
  };

  /// Try to extract a compile-time constant integer from a value.
  /// Recognizes arith.constant directly and also
  /// unrealized_conversion_cast(arith.constant) which appears after
  /// applyPartialConversion for index-to-i32 conversions.
  static std::optional<int64_t> getConstantInt(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return intAttr.getValue().getSExtValue();
    if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
      if (castOp.getNumOperands() == 1)
        if (auto constOp =
                castOp.getOperand(0).getDefiningOp<arith::ConstantOp>())
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
            return intAttr.getValue().getSExtValue();
    return std::nullopt;
  }

  /// Recursively flatten an arith.addi tree into leaf terms.
  /// Constants are summed; non-constants are collected.
  /// Recurses through arith.addi regardless of where the op is defined.
  /// Also recognizes arith.muli of two effective constants (possibly
  /// through unrealized_conversion_cast) and folds their product into
  /// constSum, so that reassociation can group addresses that differ
  /// only by a constant offset.
  static void flattenAddiTree(Value v, SmallVectorImpl<Value> &nonConstLeaves,
                              int64_t &constSum) {
    if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
      flattenAddiTree(addOp.getLhs(), nonConstLeaves, constSum);
      flattenAddiTree(addOp.getRhs(), nonConstLeaves, constSum);
      return;
    }
    if (auto cval = getConstantInt(v)) {
      constSum += *cval;
      return;
    }
    // Fold muli where both operands are effective constants.
    if (auto mulOp = v.getDefiningOp<arith::MulIOp>()) {
      auto lhsConst = getConstantInt(mulOp.getLhs());
      auto rhsConst = getConstantInt(mulOp.getRhs());
      if (lhsConst && rhsConst) {
        constSum += *lhsConst * *rhsConst;
        return;
      }
    }
    nonConstLeaves.push_back(v);
  }

  /// Sort terms canonically for deterministic grouping.
  /// Block arguments first (by owner block, then arg number),
  /// then op results (by isBeforeInBlock order).
  static void sortTermsCanonically(SmallVectorImpl<Value> &terms) {
    llvm::sort(terms, [](Value a, Value b) {
      bool aIsArg = isa<BlockArgument>(a);
      bool bIsArg = isa<BlockArgument>(b);
      if (aIsArg != bIsArg)
        return aIsArg; // block args first
      if (aIsArg) {
        auto argA = cast<BlockArgument>(a);
        auto argB = cast<BlockArgument>(b);
        if (argA.getOwner() != argB.getOwner())
          return argA.getOwner() < argB.getOwner();
        return argA.getArgNumber() < argB.getArgNumber();
      }
      // Both are op results — sort by position.
      auto *opA = a.getDefiningOp();
      auto *opB = b.getDefiningOp();
      if (opA->getBlock() != opB->getBlock())
        return opA->getBlock() < opB->getBlock();
      return opA->isBeforeInBlock(opB);
    });
  }

  /// Build the anchor value by left-folding non-constant terms.
  /// Placed after the last-defined term inside the loop body,
  /// or at the start of the loop body if all terms are loop-invariant.
  static Value buildAnchor(IRRewriter &rewriter, scf::ForOp forOp,
                           ArrayRef<Value> terms, Type type) {
    if (terms.size() == 1)
      return terms[0];

    // Find insertion point: after the last-defined term inside the loop.
    Operation *lastOp = nullptr;
    Block *loopBody = forOp.getBody();
    for (Value term : terms) {
      if (auto *defOp = term.getDefiningOp()) {
        if (auto *ancestor = loopBody->findAncestorOpInBlock(*defOp)) {
          if (!lastOp || lastOp->isBeforeInBlock(ancestor))
            lastOp = ancestor;
        }
      }
    }

    if (lastOp)
      rewriter.setInsertionPointAfter(lastOp);
    else
      rewriter.setInsertionPointToStart(loopBody);

    Location loc = forOp.getLoc();
    Value result = terms[0];
    for (size_t i = 1; i < terms.size(); ++i)
      result = arith::AddIOp::create(rewriter, loc, result, terms[i]);
    return result;
  }

  void reassociateLoop(IRRewriter &rewriter, scf::ForOp forOp) {
    // COLLECT: For each addi in the loop body, flatten and partition.
    SmallVector<Candidate> candidates;
    for (auto &op : forOp.getBody()->getOperations()) {
      auto addOp = dyn_cast<arith::AddIOp>(&op);
      if (!addOp)
        continue;

      Candidate c;
      c.op = addOp;
      c.constSum = 0;
      flattenAddiTree(addOp.getResult(), c.nonConstTerms, c.constSum);

      // Skip pure-constant expressions (canonicalize handles these).
      if (c.nonConstTerms.empty())
        continue;

      sortTermsCanonically(c.nonConstTerms);
      candidates.push_back(std::move(c));
    }

    if (candidates.empty())
      return;

    // GROUP: Group candidates with identical sorted non-constant term lists.
    SmallVector<SmallVector<size_t>> groups;
    SmallVector<bool> assigned(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
      if (assigned[i])
        continue;
      SmallVector<size_t> group = {i};
      assigned[i] = true;
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        if (assigned[j])
          continue;
        if (candidates[i].nonConstTerms == candidates[j].nonConstTerms) {
          group.push_back(j);
          assigned[j] = true;
        }
      }
      groups.push_back(std::move(group));
    }

    // REWRITE: For groups with size >= 2, build shared anchor and replace.
    for (auto &group : groups) {
      if (group.size() < 2)
        continue;

      auto &exemplar = candidates[group[0]];
      Type type = exemplar.op.getType();

      // Build the shared anchor once for the group.
      Value anchor = buildAnchor(rewriter, forOp, exemplar.nonConstTerms, type);

      // Replace each candidate with addi(anchor, combinedConst).
      for (size_t idx : group) {
        auto &c = candidates[idx];
        Value replacement;
        if (c.constSum == 0) {
          replacement = anchor;
        } else {
          rewriter.setInsertionPoint(c.op);
          Value constVal;
          if (type.isIndex())
            constVal = arith::ConstantOp::create(
                rewriter, c.op.getLoc(), rewriter.getIndexType(),
                rewriter.getIndexAttr(c.constSum));
          else
            constVal = arith::ConstantOp::create(
                rewriter, c.op.getLoc(),
                rewriter.getIntegerAttr(type, c.constSum));
          replacement =
              arith::AddIOp::create(rewriter, c.op.getLoc(), anchor, constVal);
        }
        rewriter.replaceAllUsesWith(c.op.getResult(), replacement);
      }
    }

    // Erase dead original candidate ops.
    for (auto &group : groups) {
      if (group.size() < 2)
        continue;
      for (size_t idx : group) {
        if (candidates[idx].op->use_empty())
          rewriter.eraseOp(candidates[idx].op);
      }
    }

    // Clean up dead intermediate addi ops in the loop body.
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &op :
           llvm::make_early_inc_range(forOp.getBody()->getOperations())) {
        if (isa<arith::AddIOp>(&op) && op.use_empty() &&
            !op.hasTrait<OpTrait::IsTerminator>()) {
          rewriter.eraseOp(&op);
          changed = true;
        }
      }
    }
  }
};

} // namespace mlir::transforms
