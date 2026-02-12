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

/// Returns true if any earlier operand (0..operandIdx-1) is not guaranteed to
/// be already on stack at emission time. In that case, later operands must not
/// be pre-emitted before the operation, or operand order can be inverted.
static bool hasEarlierOnDemandOperand(Operation *op, unsigned operandIdx,
                                      const DenseSet<Operation *> &stackified) {
  for (unsigned j = 0; j < operandIdx; ++j) {
    Value prev = op->getOperand(j);
    Operation *prevDef = prev.getDefiningOp();
    if (!prevDef || !stackified.contains(prevDef))
      return true;
  }
  return false;
}

/// Returns true if operandIdx has an identical earlier operand.
static bool hasEarlierEquivalentOperand(Operation *op, unsigned operandIdx) {
  Value operand = op->getOperand(operandIdx);
  for (unsigned j = 0; j < operandIdx; ++j) {
    if (op->getOperand(j) == operand)
      return true;
  }
  return false;
}

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

  // Structured control-flow interface operands should be materialized from
  // locals rather than relying on long-lived stack values crossing region
  // boundaries.
  bool forceLocalOperands =
      isa<wasmssa::BlockOp, wasmssa::LoopOp, wasmssa::IfOp,
          wasmssa::BlockReturnOp, wasmssa::BranchIfOp>(op);

  // Process operands left-to-right so that after reordering:
  // - Left operand's definition ends up further from use (pushed first)
  // - Right operand's definition ends up immediately before use (pushed
  // second) This ensures stack order: [lhs, rhs] with rhs on top. Binary ops
  // compute: bottom op top = lhs op rhs = CORRECT
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    Operation *defOp = operand.getDefiningOp();
    bool requireOnDemand = hasEarlierOnDemandOperand(op, i, stackifiedOps);
    bool hasEarlierEquivalent = hasEarlierEquivalentOperand(op, i);

    // Block argument - arrives on stack at block entry.
    // Conservatively allocate a local because:
    // 1. Multi-use args need locals (first use consumes from stack)
    // 2. Single-use args may not be in correct stack position
    // Future optimization: analyze stack order to keep some args on stack.
    if (!defOp) {
      needsTee.erase(operand);
      needsLocal.insert(operand);
      continue;
    }

    if (forceLocalOperands) {
      needsTee.erase(operand);
      needsLocal.insert(operand);
      continue;
    }

    // If any earlier operand must be materialized on-demand, this operand must
    // also be on-demand. Otherwise this operand could be pre-emitted and appear
    // below earlier operands on the stack, breaking ordered semantics.
    //
    // Exception: repeated operands (e.g., `%x, %x`) can still use tee/local.get
    // without violating order and should keep that optimization path.
    if (requireOnDemand && !hasEarlierEquivalent) {
      needsTee.erase(operand);
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
        if (!needsLocal.contains(operand))
          needsTee.insert(operand);
      } else {
        needsTee.erase(operand);
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
