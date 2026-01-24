//===- StackificationAnalysis.cpp - Stackification analysis -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements analysis utilities for stackification.
//
//===----------------------------------------------------------------------===//

#include "wasmstack/StackificationAnalysis.h"
#include "WAMI/WAMIOps.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"

namespace mlir::wasmstack {

//===----------------------------------------------------------------------===//
// Dependency Analysis
//===----------------------------------------------------------------------===//

DepInfo queryDependencies(Operation *op) {
  DepInfo info;

  // Memory loads
  if (isa<wami::LoadOp>(op)) {
    info.readsMemory = true;
  }

  // Memory stores - both write memory AND have side effects
  // (order of stores must be preserved)
  if (isa<wami::StoreOp>(op)) {
    info.writesMemory = true;
    info.hasSideEffects = true;
  }

  // Global reads - treat like memory reads for ordering purposes
  if (isa<wasmssa::GlobalGetOp>(op)) {
    info.readsMemory = true;
  }

  // Note: WasmSSA doesn't have GlobalSetOp - mutable globals are
  // handled differently. If added later, it would have side effects.

  // Calls have side effects and may read/write memory
  if (isa<wasmssa::FuncCallOp>(op)) {
    info.hasSideEffects = true;
    info.readsMemory = true;
    info.writesMemory = true;
  }

  // Local operations don't have side effects - they're SSA values
  // LocalGetOp, LocalSetOp, LocalTeeOp are fine to reorder

  // TODO: Add stack switching operations when implemented
  // suspend, resume, switch all have significant side effects

  return info;
}

bool isSafeToMove(Operation *defOp, Operation *insertBefore) {
  // Can't move across basic block boundaries
  if (defOp->getBlock() != insertBefore->getBlock())
    return false;

  DepInfo defDeps = queryDependencies(defOp);

  // Walk operations between defOp and insertBefore
  for (Operation *op = defOp->getNextNode(); op != insertBefore;
       op = op->getNextNode()) {
    if (!op)
      return false; // Reached end of block without finding insertBefore

    DepInfo opDeps = queryDependencies(op);

    // Check memory dependencies
    if (defDeps.readsMemory && opDeps.writesMemory)
      return false; // Read-after-write hazard
    if (defDeps.writesMemory && opDeps.readsMemory)
      return false; // Write-after-read hazard
    if (defDeps.writesMemory && opDeps.writesMemory)
      return false; // Write-after-write hazard

    // Check side effects (calls, traps, etc.)
    if (defDeps.hasSideEffects || opDeps.hasSideEffects)
      return false;
  }
  return true;
}

bool shouldRematerialize(Operation *op) {
  // Constants are always cheap to rematerialize
  if (isa<wasmssa::ConstOp>(op))
    return true;

  // Local.get is cheap (in the source dialect)
  if (isa<wasmssa::LocalGetOp>(op))
    return true;

  return false;
}

void allocateLocalsForBlockArgs(Region &region, DenseSet<Value> &needsLocal) {
  for (Block &block : region) {
    // Add ALL block arguments to needsLocal (conservative but correct)
    for (BlockArgument arg : block.getArguments()) {
      needsLocal.insert(arg);
    }
    // Recursively process nested regions
    for (Operation &op : block) {
      for (Region &nestedRegion : op.getRegions()) {
        allocateLocalsForBlockArgs(nestedRegion, needsLocal);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Use Count Analysis
//===----------------------------------------------------------------------===//

void UseCountAnalysis::analyze(Block &block) {
  for (Operation &op : block) {
    for (Value operand : op.getOperands()) {
      useCounts[operand]++;
    }
    // Recursively analyze nested regions
    for (Region &region : op.getRegions()) {
      for (Block &nestedBlock : region) {
        analyze(nestedBlock);
      }
    }
  }
}

} // namespace mlir::wasmstack
