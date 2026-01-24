//===- StackificationAnalysis.h - Stackification analysis -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares analysis utilities for stackification, including
// dependency analysis and use count tracking.
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_STACKIFICATIONANALYSIS_H
#define WASMSTACK_STACKIFICATIONANALYSIS_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::wasmstack {

//===----------------------------------------------------------------------===//
// Dependency Analysis
//===----------------------------------------------------------------------===//

/// Query result for operation dependencies
struct DepInfo {
  bool readsMemory = false;
  bool writesMemory = false;
  bool hasSideEffects = false;
  // Note: WebAssembly has no volatile semantics - all memory accesses
  // can be reordered unless they have data dependencies
};

/// Analyze an operation for its memory and side effect dependencies
DepInfo queryDependencies(Operation *op);

/// Check if it's safe to move defOp to immediately before insertBefore
bool isSafeToMove(Operation *defOp, Operation *insertBefore);

/// Check if an operation should be rematerialized instead of using a local
bool shouldRematerialize(Operation *op);

/// Recursively collect ALL block arguments from a region and nested regions.
/// Block arguments need locals because:
/// 1. They arrive on the stack at block entry, but their position is not
///    controlled by code motion (unlike operation results).
/// 2. Multi-use args need locals (first use consumes from stack).
/// 3. Single-use args may not be in the correct stack position for their use.
/// The conservative approach is to always use locals for block args.
void allocateLocalsForBlockArgs(Region &region, DenseSet<Value> &needsLocal);

//===----------------------------------------------------------------------===//
// Use Count Analysis
//===----------------------------------------------------------------------===//

/// Counts the number of uses for each operation's results
class UseCountAnalysis {
  DenseMap<Value, unsigned> useCounts;

public:
  UseCountAnalysis(Block &block) { analyze(block); }

  void analyze(Block &block);

  unsigned getUseCount(Value value) const {
    auto it = useCounts.find(value);
    return it != useCounts.end() ? it->second : 0;
  }

  bool hasSingleUse(Value value) const { return getUseCount(value) == 1; }
};

} // namespace mlir::wasmstack

#endif // WASMSTACK_STACKIFICATIONANALYSIS_H
