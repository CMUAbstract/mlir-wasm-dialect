//===- AffineSimplifyIf.cpp - Eliminate provable affine.if ops ---*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that eliminates affine.if operations whose
// conditions are provably always true or always false given enclosing
// affine.for loop bounds. Uses MLIR's Presburger arithmetic (via
// FlatAffineValueConstraints) to check emptiness of the domain intersected
// with the negation of each constraint in the IntegerSet.
//
// Example (nussinov):
//
//   affine.for %arg0 = 0 to 500 {
//     affine.for %arg1 = affine_map<(d0) -> (-d0 + 500)>(%arg0) to 500 {
//       affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%arg1) {
//         ...  // always true since arg1 >= 500 - arg0 >= 1 (when arg0 < 500)
//       }
//     }
//   }
//
// After this pass, the affine.if is removed and the then-block is inlined.
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-simplify-if"

namespace mlir::transforms {

#define GEN_PASS_DEF_AFFINESIMPLIFYIF
#include "Transforms/TransformsPasses.h.inc"

class AffineSimplifyIf : public impl::AffineSimplifyIfBase<AffineSimplifyIf> {
public:
  using impl::AffineSimplifyIfBase<AffineSimplifyIf>::AffineSimplifyIfBase;

  void runOnOperation() final {
    // Collect all AffineIfOps. We process them bottom-up (innermost first)
    // so that after inlining an inner if, the outer if can be re-evaluated.
    SmallVector<affine::AffineIfOp> ifOps;
    getOperation()->walk([&](affine::AffineIfOp op) { ifOps.push_back(op); });

    for (auto ifOp : ifOps) {
      // Skip if ops with results — would require merging yields.
      if (ifOp->getNumResults() > 0)
        continue;
      // The op may have been erased by a prior iteration (nested case).
      if (!ifOp->getParentOp())
        continue;
      trySimplify(ifOp);
    }
  }

private:
  /// Build the iteration domain from enclosing affine.for ops (not including
  /// the affine.if itself), then check if the affine.if condition is always
  /// true or always false within that domain.
  void trySimplify(affine::AffineIfOp ifOp) {
    // Collect enclosing affine ops (for/if/parallel), outermost first.
    SmallVector<Operation *> enclosingOps;
    affine::getEnclosingAffineOps(*ifOp, &enclosingOps);

    // We only add affine.for ops to the domain (not enclosing affine.if ops,
    // since those would add the condition we're trying to prove redundant).
    // Filter to just affine.for ops.
    SmallVector<Operation *> forOps;
    for (Operation *op : enclosingOps) {
      if (isa<affine::AffineForOp>(op))
        forOps.push_back(op);
      else if (isa<affine::AffineIfOp>(op))
        forOps.push_back(op); // include enclosing if conditions too
    }

    if (forOps.empty())
      return;

    // Build the domain from enclosing ops using getIndexSet.
    affine::FlatAffineValueConstraints domain;
    if (failed(affine::getIndexSet(forOps, &domain)))
      return;

    // Get the IntegerSet and operands from the affine.if.
    IntegerSet iSet = ifOp.getIntegerSet();
    SmallVector<Value> operands(ifOp.getOperands());
    affine::canonicalizeSetAndOperands(&iSet, &operands);

    // Build a constraint system from the IntegerSet to get it in flattened
    // form.
    affine::FlatAffineValueConstraints condCst(iSet, operands);

    // Check always-true: for each constraint in the IntegerSet, check if
    // domain ∧ ¬constraint is empty.
    if (isAlwaysTrue(domain, condCst, iSet)) {
      LLVM_DEBUG(llvm::dbgs() << "affine-simplify-if: eliminating always-true "
                              << ifOp << "\n");
      inlineThenBlock(ifOp);
      return;
    }

    // Check always-false: domain ∧ all_conditions is empty.
    if (isAlwaysFalse(domain, condCst)) {
      LLVM_DEBUG(llvm::dbgs() << "affine-simplify-if: eliminating always-false "
                              << ifOp << "\n");
      inlineElseBlock(ifOp);
      return;
    }
  }

  /// Check if every constraint in the condition's IntegerSet is implied by
  /// the domain. For each constraint c_i, we check if domain ∧ ¬c_i is empty.
  bool isAlwaysTrue(affine::FlatAffineValueConstraints &domain,
                    affine::FlatAffineValueConstraints &condCst,
                    IntegerSet iSet) {
    // The condCst has the constraints already flattened. We need to align
    // its variables with the domain, then negate each constraint.
    //
    // Strategy: merge the domain and condCst variable spaces, then for each
    // constraint in condCst, add its negation to a copy of the domain and
    // check emptiness.

    // Merge variable spaces: align domain's variables with condCst's.
    affine::FlatAffineValueConstraints mergedDomain(domain);
    mergedDomain.mergeAndAlignVarsWithOther(0, &condCst);

    // Now condCst and mergedDomain share the same variable layout.
    // condCst contains the constraints from the IntegerSet. We iterate over
    // them and negate each one.
    unsigned numIneqs = condCst.getNumInequalities();
    unsigned numEqs = condCst.getNumEqualities();

    for (unsigned i = 0; i < numIneqs; ++i) {
      // Inequality: f(x) >= 0. Negation: -f(x) - 1 >= 0 (i.e. f(x) <= -1).
      affine::FlatAffineValueConstraints test(mergedDomain);
      // Align test with condCst (they should already share layout after merge).
      test.mergeAndAlignVarsWithOther(0, &condCst);
      SmallVector<int64_t> negated;
      for (unsigned j = 0; j < condCst.getNumCols(); ++j)
        negated.push_back(-condCst.atIneq64(i, j));
      // f(x) >= 0 negated to -f(x) - 1 >= 0
      negated.back() -= 1;
      // Pad to match test's column count if needed.
      while (negated.size() < test.getNumCols())
        negated.insert(negated.end() - 1, 0);
      test.addInequality(negated);
      if (!test.isEmpty())
        return false;
    }

    for (unsigned i = 0; i < numEqs; ++i) {
      // Equality: f(x) = 0. Negation: f(x) >= 1 ∨ -f(x) >= 1.
      // Check both disjuncts separately.
      SmallVector<int64_t> row;
      for (unsigned j = 0; j < condCst.getNumCols(); ++j)
        row.push_back(condCst.atEq64(i, j));

      // Disjunct 1: f(x) >= 1 → f(x) - 1 >= 0
      {
        affine::FlatAffineValueConstraints test(mergedDomain);
        test.mergeAndAlignVarsWithOther(0, &condCst);
        SmallVector<int64_t> ineq(row);
        ineq.back() -= 1;
        while (ineq.size() < test.getNumCols())
          ineq.insert(ineq.end() - 1, 0);
        test.addInequality(ineq);
        if (!test.isEmpty())
          return false;
      }

      // Disjunct 2: -f(x) >= 1 → -f(x) - 1 >= 0
      {
        affine::FlatAffineValueConstraints test(mergedDomain);
        test.mergeAndAlignVarsWithOther(0, &condCst);
        SmallVector<int64_t> ineq;
        for (auto v : row)
          ineq.push_back(-v);
        ineq.back() -= 1;
        while (ineq.size() < test.getNumCols())
          ineq.insert(ineq.end() - 1, 0);
        test.addInequality(ineq);
        if (!test.isEmpty())
          return false;
      }
    }

    return true;
  }

  /// Check if the condition is always false: domain ∧ condition is empty.
  bool isAlwaysFalse(affine::FlatAffineValueConstraints &domain,
                     affine::FlatAffineValueConstraints &condCst) {
    affine::FlatAffineValueConstraints test(domain);
    test.mergeAndAlignVarsWithOther(0, &condCst);
    test.append(condCst);
    return test.isEmpty();
  }

  /// Inline the then-block of an affine.if before the op and erase it.
  void inlineThenBlock(affine::AffineIfOp ifOp) {
    Block *thenBlock = ifOp.getThenBlock();
    // Move all ops except the terminator (affine.yield) before the affine.if.
    auto &parentBlock = *ifOp->getBlock();
    auto insertPt = Block::iterator(ifOp);
    for (auto &op : llvm::make_early_inc_range(thenBlock->without_terminator()))
      op.moveBefore(&parentBlock, insertPt);
    ifOp->erase();
  }

  /// Inline the else-block (if present) or just erase the affine.if.
  void inlineElseBlock(affine::AffineIfOp ifOp) {
    if (ifOp.hasElse()) {
      Block *elseBlock = ifOp.getElseBlock();
      auto &parentBlock = *ifOp->getBlock();
      auto insertPt = Block::iterator(ifOp);
      for (auto &op :
           llvm::make_early_inc_range(elseBlock->without_terminator()))
        op.moveBefore(&parentBlock, insertPt);
    }
    ifOp->erase();
  }
};

} // namespace mlir::transforms
