//===- TreeWalker.cpp - Stackification tree walker --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TreeWalker class which performs LLVM-style
// stackification by building expression trees.
//
//===----------------------------------------------------------------------===//

#include "wasmstack/TreeWalker.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Builders.h"

namespace mlir::wasmstack {

void TreeWalker::processBlock(Block &block) {
  // Collect operations in reverse order (bottom-up processing)
  SmallVector<Operation *> ops;
  for (Operation &op : block) {
    ops.push_back(&op);
  }

  // Process from last to first
  // Skip operations already in stackifiedOps - they were processed when
  // their user was processed (e.g., add processed via block_return).
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    if (!stackifiedOps.contains(*it))
      processOperation(*it);
  }
}

void TreeWalker::processOperation(Operation *op) {
  // Recursively process nested regions first.
  // This ensures block arguments inside control flow (loop, block, if)
  // are properly analyzed for local allocation.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      processBlock(block);
    }
  }

  // Special case: branch_if with inputs (args to pass to target block).
  // WebAssembly requires stack order: [args..., condition] with condition on
  // top. If we stackify the condition's defining op, it gets emitted before
  // args, resulting in wrong order: [condition, args...]. To fix this, we
  // ensure the condition gets a local so it can be reordered during emission.
  if (auto branchIfOp = dyn_cast<wasmssa::BranchIfOp>(op)) {
    if (!branchIfOp.getInputs().empty()) {
      Value condition = branchIfOp.getCondition();
      Operation *condDefOp = condition.getDefiningOp();
      // If condition would be stackified (single-use, has defining op),
      // force it to use a local instead
      if (condDefOp && useCount.hasSingleUse(condition)) {
        needsLocal.insert(condition);
        // Don't return - still process other operands normally
      }
    }
  }

  // Process operands left-to-right so that after reordering:
  // - Left operand's definition ends up further from use (pushed first)
  // - Right operand's definition ends up immediately before use (pushed
  // second) This ensures stack order: [lhs, rhs] with rhs on top. Binary ops
  // compute: bottom op top = lhs op rhs = CORRECT
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    Operation *defOp = operand.getDefiningOp();

    // Block argument - arrives on stack at block entry.
    // Conservatively allocate a local because:
    // 1. Multi-use args need locals (first use consumes from stack)
    // 2. Single-use args may not be in correct stack position
    // Future optimization: analyze stack order to keep some args on stack.
    if (!defOp) {
      needsLocal.insert(operand);
      continue;
    }

    // Already stackified operations need special handling for additional uses
    if (stackifiedOps.contains(defOp)) {
      if (shouldRematerialize(defOp)) {
        // Clone cheap ops for additional uses (e.g., second use of same
        // const)
        OpBuilder builder(op);
        Operation *clone = builder.clone(*defOp);
        op->setOperand(i, clone->getResult(0));
        stackifiedOps.insert(clone);
        processOperation(clone);
      }
      // Non-rematerializable ops that are already stackified: the first use
      // consumes from stack, other uses need locals (handled by tee/local
      // logic when defOp was first processed)
      continue;
    }

    // Try to stackify this operand (single-use values)
    if (canStackify(defOp, op)) {
      // Move the defining operation immediately before this operation
      defOp->moveBefore(op);
      stackifiedOps.insert(defOp);

      // Recursively process the moved operation's operands
      processOperation(defOp);
    } else if (shouldRematerialize(defOp)) {
      // Rematerializable operation (constants, local.get)
      if (defOp->getBlock() == op->getBlock()) {
        // Same block - safe to move original for first use
        // (subsequent uses will hit the stackifiedOps check above and clone)
        defOp->moveBefore(op);
        stackifiedOps.insert(defOp);
        processOperation(defOp);
      } else {
        // Different block (e.g., nested region using outer value) - clone
        OpBuilder builder(op);
        Operation *clone = builder.clone(*defOp);
        op->setOperand(i, clone->getResult(0));
        stackifiedOps.insert(clone);
        processOperation(clone);
      }
    } else {
      // Can't stackify directly - check if we can use tee
      // Tee is useful when one use can consume from stack, others from local
      if (canUseTee(defOp, operand)) {
        needsTee.insert(operand);
      } else {
        needsLocal.insert(operand);
      }
    }
  }
}

bool TreeWalker::canUseTee(Operation *defOp, Value value) {
  // Must have multiple uses
  if (useCount.hasSingleUse(value))
    return false;

  // Check if any use is immediately after defOp (or can be made so)
  for (Operation *user : value.getUsers()) {
    // Check if we can move defOp immediately before this user
    if (defOp->getBlock() == user->getBlock() && isSafeToMove(defOp, user)) {
      return true; // At least one use can consume from stack
    }
  }
  return false;
}

bool TreeWalker::canStackify(Operation *defOp, Operation *useOp) {
  // Must be single-use to stackify directly
  if (defOp->getNumResults() != 1)
    return false;

  Value result = defOp->getResult(0);
  if (!useCount.hasSingleUse(result))
    return false;

  // Check if it's safe to move (no dependency hazards)
  if (!isSafeToMove(defOp, useOp))
    return false;

  return true;
}

} // namespace mlir::wasmstack
