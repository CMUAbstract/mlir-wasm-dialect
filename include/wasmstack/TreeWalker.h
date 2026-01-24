//===- TreeWalker.h - Stackification tree walker ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TreeWalker class which performs LLVM-style
// stackification by building expression trees.
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_TREEWALKER_H
#define WASMSTACK_TREEWALKER_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "wasmstack/StackificationAnalysis.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::wasmstack {

/// TreeWalker performs LLVM-style stackification by building expression trees.
/// It walks operations bottom-up and tries to move definitions immediately
/// before their uses, creating a stack-based evaluation order.
class TreeWalker {
  /// Operations that have been "stackified" - their result is on the
  /// implicit value stack and will be consumed by the next operation.
  DenseSet<Operation *> stackifiedOps;

  /// Values that need to be stored in locals (all uses via local.get)
  DenseSet<Value> needsLocal;

  /// Values that should use local.tee (first use from stack, rest via
  /// local.get)
  DenseSet<Value> needsTee;

  /// Use count analysis
  const UseCountAnalysis &useCount;

public:
  TreeWalker(const UseCountAnalysis &useCount) : useCount(useCount) {}

  /// Get the set of values that need locals (all uses via local.get)
  const DenseSet<Value> &getValuesNeedingLocals() const { return needsLocal; }

  /// Get the set of values that should use tee (first use from stack)
  const DenseSet<Value> &getValuesNeedingTee() const { return needsTee; }

  /// Check if an operation has been stackified
  bool isStackified(Operation *op) const { return stackifiedOps.contains(op); }

  /// Process a block, attempting to stackify operations
  void processBlock(Block &block);

  /// Process a single operation, trying to stackify its operands
  void processOperation(Operation *op);

private:
  /// Check if a value can benefit from local.tee
  /// This is true when the value's defining op can be moved to immediately
  /// before ONE of its uses (which will consume from stack)
  bool canUseTee(Operation *defOp, Value value);

  /// Check if defOp can be stackified (moved immediately before useOp)
  bool canStackify(Operation *defOp, Operation *useOp);
};

} // namespace mlir::wasmstack

#endif // WASMSTACK_TREEWALKER_H
