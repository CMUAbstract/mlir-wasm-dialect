//===- AffineAccumulatorPromotion.cpp - Promote affine acc patterns -*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements promotion of load-accumulate-store patterns in affine
// loops to iter_args. Unlike the SCF-level promote-loop-accumulators pass,
// this operates on affine.for/affine.load/affine.store and leverages affine
// dependence analysis to prove non-aliasing between the accumulator and other
// memory accesses. This enables promotion in cases the SCF pass cannot handle:
//
//   - Multiple loads from the same memref (grouped by access identity)
//   - Dynamic loop bounds (no trip count check needed)
//   - Index expressions that differ as SSA values but are structurally equal
//
// Example (nussinov DP kernel):
//
//   affine.for %k = ... {
//     %acc = affine.load %mem[%i, %j]          // accumulator
//     %v1  = affine.load %mem[%i, %k]          // non-aliasing read
//     %v2  = affine.load %mem[%k + 1, %j]      // non-aliasing read
//     %new = max(%acc, %v1 + %v2)
//     affine.store %new, %mem[%i, %j]           // accumulator
//   }
//
// After promotion:
//
//   %init = affine.load %mem[%i, %j]
//   %result = affine.for %k = ... iter_args(%acc = %init) -> (i32) {
//     %v1  = affine.load %mem[%i, %k]
//     %v2  = affine.load %mem[%k + 1, %j]
//     %new = max(%acc, %v1 + %v2)
//     affine.yield %new
//   }
//   affine.store %result, %mem[%i, %j]
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_AFFINEPROMOTEACCUMULATORS
#include "Transforms/TransformsPasses.h.inc"

class AffineAccumulatorPromotion
    : public impl::AffinePromoteAccumulatorsBase<AffineAccumulatorPromotion> {
public:
  using impl::AffinePromoteAccumulatorsBase<
      AffineAccumulatorPromotion>::AffinePromoteAccumulatorsBase;

  void runOnOperation() final {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Collect all AffineForOps, then process inner-first (reverse of pre-order)
    // so that inner loops are promoted before their parents are inspected.
    SmallVector<affine::AffineForOp> forOps;
    funcOp.walk([&](affine::AffineForOp forOp) { forOps.push_back(forOp); });

    for (auto forOp : llvm::reverse(forOps))
      promoteLoop(rewriter, forOp);
  }

private:
  /// A candidate load-accumulate-store pair.
  struct Candidate {
    affine::AffineLoadOp loadOp;
    affine::AffineStoreOp storeOp;
  };

  /// Check whether the store value transitively depends on the load result.
  static bool storeValueDependsOnLoad(affine::AffineStoreOp storeOp,
                                      affine::AffineLoadOp loadOp,
                                      affine::AffineForOp forOp) {
    Value storeVal = storeOp.getValueToStore();
    Value loadResult = loadOp.getResult();

    SmallVector<Value> worklist;
    DenseSet<Value> visited;
    worklist.push_back(storeVal);

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (v == loadResult)
        return true;
      if (!visited.insert(v).second)
        continue;

      Operation *defOp = v.getDefiningOp();
      if (!defOp)
        continue;
      if (!forOp.getBody()->findAncestorOpInBlock(*defOp))
        continue;
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
    return false;
  }

  /// Get the number of enclosing AffineForOps for an operation.
  static unsigned getEnclosingLoopDepth(Operation *op) {
    unsigned depth = 0;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isa<affine::AffineForOp>(parent))
        ++depth;
    }
    return depth;
  }

  /// Check if all address operands of a load are defined outside the loop.
  static bool canHoistLoad(affine::AffineLoadOp loadOp,
                           affine::AffineForOp forOp) {
    if (!forOp.isDefinedOutsideOfLoop(loadOp.getMemRef()))
      return false;
    for (Value v : loadOp.getMapOperands())
      if (!forOp.isDefinedOutsideOfLoop(v))
        return false;
    return true;
  }

  /// Check if all address operands of a store are defined outside the loop.
  static bool canSinkStore(affine::AffineStoreOp storeOp,
                           affine::AffineForOp forOp) {
    if (!forOp.isDefinedOutsideOfLoop(storeOp.getMemRef()))
      return false;
    for (Value v : storeOp.getMapOperands())
      if (!forOp.isDefinedOutsideOfLoop(v))
        return false;
    return true;
  }

  /// Try to find a preceding affine.store to the same location for
  /// store-to-load forwarding. Returns the stored value if found.
  static Value findPrecedingStoreValue(Operation *beforeOp,
                                       affine::AffineLoadOp loadOp) {
    affine::MemRefAccess loadAccess(loadOp);
    Operation *prev = beforeOp->getPrevNode();
    while (prev) {
      if (auto storeOp = dyn_cast<affine::AffineStoreOp>(prev)) {
        affine::MemRefAccess storeAccess(storeOp);
        if (storeAccess == loadAccess)
          return storeOp.getValueToStore();
        // Any other store could alias — stop searching.
        return nullptr;
      }
      if (isa<affine::AffineLoadOp>(prev)) {
        prev = prev->getPrevNode();
        continue;
      }
      if (isMemoryEffectFree(prev)) {
        prev = prev->getPrevNode();
        continue;
      }
      // Unknown side effect — stop.
      return nullptr;
    }
    return nullptr;
  }

  void promoteLoop(IRRewriter &rewriter, affine::AffineForOp forOp) {
    Block *body = forOp.getBody();

    // Collect direct-child affine.load and affine.store ops.
    // Bail if there are any non-affine memory effects in the body.
    SmallVector<affine::AffineLoadOp> loads;
    SmallVector<affine::AffineStoreOp> stores;

    for (auto &op : body->getOperations()) {
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(&op)) {
        loads.push_back(loadOp);
        continue;
      }
      if (auto storeOp = dyn_cast<affine::AffineStoreOp>(&op)) {
        stores.push_back(storeOp);
        continue;
      }
      if (isa<affine::AffineYieldOp>(&op))
        continue;
      if (isMemoryEffectFree(&op))
        continue;
      // Op with memory side effects (nested loop, function call, etc.) — bail.
      return;
    }

    if (loads.empty() || stores.empty())
      return;

    // Build MemRefAccess for all loads and stores.
    SmallVector<affine::MemRefAccess> loadAccesses, storeAccesses;
    for (auto loadOp : loads)
      loadAccesses.emplace_back(loadOp);
    for (auto storeOp : stores)
      storeAccesses.emplace_back(storeOp);

    // Step 1: Identify accumulator candidates.
    // A candidate is a (load, store) pair with:
    //   - Same access identity (MemRefAccess::operator==)
    //   - Loop-invariant address
    //   - Store value depends on load result
    //   - Exactly 1 load and 1 store with that access identity
    SmallVector<Candidate> candidates;
    DenseSet<Operation *> usedOps;

    for (auto [si, storeOp] : llvm::enumerate(stores)) {
      if (!affine::isInvariantAccess(storeOp, forOp))
        continue;
      if (!canSinkStore(storeOp, forOp))
        continue;

      // Find loads with matching access identity.
      SmallVector<unsigned> matchingLoads;
      for (auto [li, loadOp] : llvm::enumerate(loads)) {
        if (loadAccesses[li] == storeAccesses[si])
          matchingLoads.push_back(li);
      }
      if (matchingLoads.size() != 1)
        continue;

      unsigned li = matchingLoads[0];
      auto loadOp = loads[li];

      // Exactly 1 store with this access identity.
      unsigned storeCount = 0;
      for (unsigned sj = 0; sj < stores.size(); ++sj) {
        if (storeAccesses[sj] == loadAccesses[li])
          ++storeCount;
      }
      if (storeCount != 1)
        continue;

      if (!canHoistLoad(loadOp, forOp))
        continue;
      if (!storeValueDependsOnLoad(storeOp, loadOp, forOp))
        continue;
      if (usedOps.contains(loadOp.getOperation()) ||
          usedOps.contains(storeOp.getOperation()))
        continue;

      candidates.push_back({loadOp, storeOp});
      usedOps.insert(loadOp.getOperation());
      usedOps.insert(storeOp.getOperation());
    }

    if (candidates.empty())
      return;

    // Step 2: Non-aliasing check using affine dependence analysis.
    // For each candidate's store, check against every other load and store
    // in the loop body. The loop depth is the number of enclosing AffineForOps
    // (including the current one), which tells checkMemrefAccessDependence
    // to consider all loop bound constraints.
    unsigned loopDepth = getEnclosingLoopDepth(loads[0]);

    SmallVector<Candidate> safeCandidates;
    for (auto &c : candidates) {
      affine::MemRefAccess candStoreAccess(c.storeOp);
      bool safe = true;

      // Check against all other loads from the same memref.
      for (auto [li, loadOp] : llvm::enumerate(loads)) {
        if (loadOp == c.loadOp)
          continue;
        if (loadOp.getMemRef() != c.storeOp.getMemRef())
          continue;
        affine::MemRefAccess otherLoadAccess(loadOp);
        auto result = affine::checkMemrefAccessDependence(
            candStoreAccess, otherLoadAccess, loopDepth);
        if (!affine::noDependence(result)) {
          safe = false;
          break;
        }
      }

      // Check against all other stores to the same memref.
      if (safe) {
        for (auto [si, storeOp] : llvm::enumerate(stores)) {
          if (storeOp == c.storeOp)
            continue;
          if (storeOp.getMemRef() != c.storeOp.getMemRef())
            continue;
          affine::MemRefAccess otherStoreAccess(storeOp);
          auto result = affine::checkMemrefAccessDependence(
              candStoreAccess, otherStoreAccess, loopDepth);
          if (!affine::noDependence(result)) {
            safe = false;
            break;
          }
        }
      }

      if (safe)
        safeCandidates.push_back(c);
    }

    if (safeCandidates.empty())
      return;

    // Step 3: Determine initial values — either forward from a preceding
    // store (avoiding a redundant load) or hoist a load before the loop.
    rewriter.setInsertionPoint(forOp);
    SmallVector<Value> initValues;
    for (auto &c : safeCandidates) {
      Value forwarded = findPrecedingStoreValue(forOp, c.loadOp);
      if (forwarded) {
        initValues.push_back(forwarded);
      } else {
        auto hoistedLoad = affine::AffineLoadOp::create(
            rewriter, c.loadOp.getLoc(), c.loadOp.getMemRef(),
            c.loadOp.getAffineMap(), c.loadOp.getMapOperands());
        initValues.push_back(hoistedLoad);
      }
    }

    // Step 4: Transform — add iter_args via replaceWithAdditionalYields.
    unsigned numOrigIterArgs = forOp.getNumRegionIterArgs();

    SmallVector<affine::AffineStoreOp> candStores;
    SmallVector<affine::AffineLoadOp> candLoads;
    for (auto &c : safeCandidates) {
      candStores.push_back(c.storeOp);
      candLoads.push_back(c.loadOp);
    }

    auto newYieldFn =
        [&](OpBuilder &b, Location loc,
            ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
      SmallVector<Value> yieldValues;
      for (auto [i, bbArg] : llvm::enumerate(newBbArgs)) {
        rewriter.replaceAllUsesWith(candLoads[i].getResult(), bbArg);
        yieldValues.push_back(candStores[i].getValueToStore());
      }
      return yieldValues;
    };

    auto result = forOp.replaceWithAdditionalYields(
        rewriter, initValues, /*replaceInitOperandUsesInLoop=*/false,
        newYieldFn);
    if (failed(result))
      return;

    auto newForOp = cast<affine::AffineForOp>(*result);

    // Insert stores after the loop for each candidate.
    rewriter.setInsertionPointAfter(newForOp);
    for (auto [i, c] : llvm::enumerate(safeCandidates)) {
      Value loopResult = newForOp.getResult(numOrigIterArgs + i);
      affine::AffineStoreOp::create(
          rewriter, c.storeOp.getLoc(), loopResult, c.storeOp.getMemRef(),
          c.storeOp.getAffineMap(), c.storeOp.getMapOperands());
    }

    // Erase original load and store ops (now in the new loop body).
    for (auto &storeOp : candStores)
      rewriter.eraseOp(storeOp);
    for (auto &loadOp : candLoads) {
      if (loadOp->use_empty())
        rewriter.eraseOp(loadOp);
    }
  }
};

} // namespace mlir::transforms
