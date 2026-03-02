//===- LoopAccumulatorPromotion.cpp - Promote load-acc-store -------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements promotion of load-accumulate-store patterns in SCF
// loops to iter_args. This eliminates redundant memory traffic for
// accumulation patterns like:
//
//   scf.for %k = ... {
//     %val = memref.load %mem[%i, %j]
//     %new = arith.addf %val, %product
//     memref.store %new, %mem[%i, %j]
//   }
//
// After promotion:
//
//   %init = memref.load %mem[%i, %j]
//   %result = scf.for %k = ... iter_args(%acc = %init) -> (f64) {
//     %new = arith.addf %acc, %product
//     scf.yield %new
//   }
//   memref.store %result, %mem[%i, %j]
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_LOOPACCUMULATORPROMOTION
#include "Transforms/TransformsPasses.h.inc"

class LoopAccumulatorPromotion
    : public impl::LoopAccumulatorPromotionBase<LoopAccumulatorPromotion> {
public:
  using impl::LoopAccumulatorPromotionBase<
      LoopAccumulatorPromotion>::LoopAccumulatorPromotionBase;

  void runOnOperation() final {
    auto module = getOperation();
    IRRewriter rewriter(module.getContext());

    // Collect all ForOps, then process inner-first (reverse of pre-order).
    SmallVector<scf::ForOp> forOps;
    module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (auto forOp : llvm::reverse(forOps))
      promoteLoop(rewriter, forOp);
  }

private:
  /// A candidate load-accumulate-store triple.
  struct Candidate {
    memref::LoadOp loadOp;
    memref::StoreOp storeOp;
    Value storeValue; // The value being stored (the accumulated result)
  };

  /// Check whether the store value transitively depends on the load result.
  static bool storeValueDependsOnLoad(memref::StoreOp storeOp,
                                      memref::LoadOp loadOp, scf::ForOp forOp) {
    Value storeVal = storeOp.getValueToStore();
    Value loadResult = loadOp.getResult();

    // BFS/DFS through the def chain within the loop body.
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
      // Only follow ops inside the loop body.
      if (!forOp.getBody()->findAncestorOpInBlock(*defOp))
        continue;
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
    return false;
  }

  /// Check if lb < ub can be proven for constant bounds.
  static bool hasPositiveTripCount(scf::ForOp forOp) {
    auto lbConst = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
    auto ubConst = forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
    if (!lbConst || !ubConst)
      return false;

    auto lbAttr = dyn_cast<IntegerAttr>(lbConst.getValue());
    auto ubAttr = dyn_cast<IntegerAttr>(ubConst.getValue());
    if (!lbAttr || !ubAttr)
      return false;

    return lbAttr.getValue().slt(ubAttr.getValue());
  }

  /// Try to find a preceding store to the same memref+indices before
  /// `beforeOp`, with no intervening memory writes. Returns the stored
  /// value if found (enabling store-to-load forwarding), nullptr otherwise.
  static Value findPrecedingStoreValue(Operation *beforeOp, Value memref,
                                       OperandRange indices) {
    Operation *prev = beforeOp->getPrevNode();
    while (prev) {
      if (auto storeOp = dyn_cast<memref::StoreOp>(prev)) {
        if (storeOp.getMemRef() == memref) {
          auto storeIndices = storeOp.getIndices();
          if (storeIndices.size() == indices.size() &&
              std::equal(storeIndices.begin(), storeIndices.end(),
                         indices.begin()))
            return storeOp.getValueToStore();
        }
        // Store to different address — could alias, stop.
        return nullptr;
      }
      // Loads don't write memory — safe to skip.
      if (isa<memref::LoadOp>(prev)) {
        prev = prev->getPrevNode();
        continue;
      }
      // Skip memory-effect-free ops.
      if (isMemoryEffectFree(prev)) {
        prev = prev->getPrevNode();
        continue;
      }
      // Unknown side effect — stop.
      return nullptr;
    }
    return nullptr;
  }

  /// Check if a memref value originates from memref.alloc (not a function arg
  /// or other aliasable source).
  static bool isFromAlloc(Value memref) {
    return memref.getDefiningOp<memref::AllocOp>() != nullptr ||
           memref.getDefiningOp<memref::AllocaOp>() != nullptr;
  }

  void promoteLoop(IRRewriter &rewriter, scf::ForOp forOp) {
    Block *body = forOp.getBody();

    // Collect all loads and stores in the loop body (direct children only).
    SmallVector<memref::LoadOp> loads;
    SmallVector<memref::StoreOp> stores;
    bool hasInterferingSideEffects = false;

    for (auto &op : body->getOperations()) {
      if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
        loads.push_back(loadOp);
        continue;
      }
      if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
        stores.push_back(storeOp);
        continue;
      }

      // Check for ops with memory side effects that are not load/store.
      // scf.yield is always safe.
      if (isa<scf::YieldOp>(&op))
        continue;

      // For any other op, check if it's memory-effect-free.
      // Ops without MemoryEffectOpInterface are treated conservatively
      // as potentially having side effects (e.g. func.call, nested loops).
      if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(&op)) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffects.getEffects(effects);
        for (auto &effect : effects) {
          if (isa<MemoryEffects::Read>(effect.getEffect()) ||
              isa<MemoryEffects::Write>(effect.getEffect())) {
            hasInterferingSideEffects = true;
            break;
          }
        }
      } else if (!op.hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
                 !isMemoryEffectFree(&op)) {
        // Op doesn't implement MemoryEffectOpInterface — be conservative.
        // This catches func.call, nested scf.for with side effects, etc.
        hasInterferingSideEffects = true;
      }
    }

    // If there are interfering side effects, bail out entirely.
    if (hasInterferingSideEffects)
      return;

    // Build map: memref SSA Value → {loads, stores}
    DenseMap<Value, SmallVector<memref::LoadOp>> loadsByMemref;
    DenseMap<Value, SmallVector<memref::StoreOp>> storesByMemref;

    for (auto loadOp : loads)
      loadsByMemref[loadOp.getMemRef()].push_back(loadOp);
    for (auto storeOp : stores)
      storesByMemref[storeOp.getMemRef()].push_back(storeOp);

    // Find candidates: memrefs with exactly 1 load + 1 store.
    SmallVector<Candidate> candidates;

    for (auto &[memref, memLoads] : loadsByMemref) {
      // Exactly 1 load for this memref.
      if (memLoads.size() != 1)
        continue;

      auto storeIt = storesByMemref.find(memref);
      if (storeIt == storesByMemref.end())
        continue;

      auto &memStores = storeIt->second;
      // Exactly 1 store for this memref.
      if (memStores.size() != 1)
        continue;

      auto loadOp = memLoads[0];
      auto storeOp = memStores[0];

      // Check memref originates from alloc (not a function arg).
      if (!isFromAlloc(memref))
        continue;

      // Check memref value is loop-invariant.
      if (!forOp.isDefinedOutsideOfLoop(memref))
        continue;

      // Check all indices are loop-invariant and identical between load/store.
      auto loadIndices = loadOp.getIndices();
      auto storeIndices = storeOp.getIndices();
      if (loadIndices.size() != storeIndices.size())
        continue;

      bool indicesMatch = true;
      for (auto [li, si] : llvm::zip(loadIndices, storeIndices)) {
        if (li != si || !forOp.isDefinedOutsideOfLoop(li)) {
          indicesMatch = false;
          break;
        }
      }
      if (!indicesMatch)
        continue;

      // Check store value depends on load result.
      if (!storeValueDependsOnLoad(storeOp, loadOp, forOp))
        continue;

      candidates.push_back({loadOp, storeOp, storeOp.getValueToStore()});
    }

    if (candidates.empty())
      return;

    // Trip count check: for constant bounds, verify lb < ub.
    // For dynamic bounds, skip (conservative).
    if (!hasPositiveTripCount(forOp))
      return;

    // Perform the transformation for all candidates at once.
    // Step 1: Determine initial values — either forward from a preceding
    // store (avoiding a redundant load) or hoist a load before the loop.
    rewriter.setInsertionPoint(forOp);
    SmallVector<Value> initValues;
    for (auto &c : candidates) {
      Value forwarded = findPrecedingStoreValue(forOp, c.loadOp.getMemRef(),
                                                c.loadOp.getIndices());
      if (forwarded) {
        initValues.push_back(forwarded);
      } else {
        auto hoistedLoad =
            memref::LoadOp::create(rewriter, c.loadOp.getLoc(),
                                   c.loadOp.getMemRef(), c.loadOp.getIndices());
        initValues.push_back(hoistedLoad);
      }
    }

    // Step 2: Add iter_args via replaceWithAdditionalYields.
    unsigned numOrigIterArgs = forOp.getNumRegionIterArgs();

    // Build the yield function: for each new block arg, yield the accumulated
    // value (what was being stored).
    SmallVector<memref::StoreOp> storeOps;
    SmallVector<memref::LoadOp> loadOps;
    for (auto &c : candidates) {
      storeOps.push_back(c.storeOp);
      loadOps.push_back(c.loadOp);
    }

    auto newYieldFn =
        [&](OpBuilder &b, Location yieldLoc,
            ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
      SmallVector<Value> yieldValues;
      for (auto [i, bbArg] : llvm::enumerate(newBbArgs)) {
        // Replace uses of the load result with the new block arg.
        rewriter.replaceAllUsesWith(loadOps[i].getResult(), bbArg);
        // Yield the value that was being stored.
        yieldValues.push_back(storeOps[i].getValueToStore());
      }
      return yieldValues;
    };

    auto result = forOp.replaceWithAdditionalYields(
        rewriter, initValues, /*replaceInitOperandUsesInLoop=*/false,
        newYieldFn);
    if (failed(result))
      return;

    auto newForOp = cast<scf::ForOp>(*result);

    // Step 3: Insert stores after the loop for each candidate.
    rewriter.setInsertionPointAfter(newForOp);
    for (auto [i, c] : llvm::enumerate(candidates)) {
      Value loopResult = newForOp.getResult(numOrigIterArgs + i);
      memref::StoreOp::create(rewriter, c.storeOp.getLoc(), loopResult,
                              c.storeOp.getMemRef(), c.storeOp.getIndices());
    }

    // Step 4: Erase original load and store ops (now in the new loop body).
    // The storeOps/loadOps pointers still refer to ops in the new loop body
    // since replaceWithAdditionalYields moves the body.
    for (auto &storeOp : storeOps)
      rewriter.eraseOp(storeOp);
    for (auto &loadOp : loadOps) {
      if (loadOp->use_empty())
        rewriter.eraseOp(loadOp);
    }
  }
};

} // namespace mlir::transforms
