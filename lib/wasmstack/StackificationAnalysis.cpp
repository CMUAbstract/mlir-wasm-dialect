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
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::wasmstack {

//===----------------------------------------------------------------------===//
// Dependency Analysis
//===----------------------------------------------------------------------===//

DepInfo queryDependencies(Operation *op) {
  DepInfo info;

  // Cheap SSA-only operations are safe to reorder.
  if (isa<wasmssa::ConstOp, wasmssa::LocalGetOp, wasmssa::LocalSetOp,
          wasmssa::LocalTeeOp>(op)) {
    return info;
  }

  // Memory loads.
  if (isa<wami::LoadOp>(op)) {
    info.readsMemory = true;
    return info;
  }

  // Memory stores - both write memory and have side effects.
  if (isa<wami::StoreOp>(op)) {
    info.writesMemory = true;
    info.hasSideEffects = true;
    return info;
  }

  // Global reads behave like memory reads for ordering.
  if (isa<wasmssa::GlobalGetOp>(op)) {
    info.readsMemory = true;
    return info;
  }

  // Calls conservatively read/write memory and have side effects.
  if (isa<wasmssa::FuncCallOp>(op)) {
    info.hasSideEffects = true;
    info.readsMemory = true;
    info.writesMemory = true;
    return info;
  }

  // Trapping operations must not be moved across side effects.
  if (isa<wasmssa::DivSIOp, wasmssa::DivUIOp, wasmssa::RemSIOp,
          wasmssa::RemUIOp, wami::TruncSOp, wami::TruncUOp>(op)) {
    info.hasSideEffects = true;
    return info;
  }

  // Use memory effects interface when available.
  if (auto effectIface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    effectIface.getEffects(effects);
    for (const auto &effect : effects) {
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        info.readsMemory = true;
        continue;
      }
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        info.writesMemory = true;
        continue;
      }
      if (isa<MemoryEffects::Allocate, MemoryEffects::Free>(
              effect.getEffect())) {
        info.readsMemory = true;
        info.writesMemory = true;
        info.hasSideEffects = true;
        continue;
      }
      info.hasSideEffects = true;
    }
    return info;
  }

  // Unknown operations are treated conservatively.
  info.readsMemory = true;
  info.writesMemory = true;
  info.hasSideEffects = true;

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

void allocateLocalsForBlockArgsOrdered(Region &region,
                                       DenseSet<Value> &needsLocal,
                                       SmallVectorImpl<Value> &localOrder) {
  for (Block &block : region) {
    for (BlockArgument arg : block.getArguments()) {
      if (needsLocal.insert(arg).second)
        localOrder.push_back(arg);
    }
    for (Operation &op : block) {
      for (Region &nestedRegion : op.getRegions())
        allocateLocalsForBlockArgsOrdered(nestedRegion, needsLocal, localOrder);
    }
  }
}

//===----------------------------------------------------------------------===//
// Use Count Analysis
//===----------------------------------------------------------------------===//

void UseCountAnalysis::analyze(Region &region) {
  for (Block &block : region)
    analyze(block);
}

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
