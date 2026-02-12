//===- WasmStackEmitter.h - WasmStack code emitter --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the WasmStackEmitter class which emits WasmStack
// operations from stackified WasmSSA/WAMI operations.
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_WASMSTACKEMITTER_H
#define WASMSTACK_WASMSTACKEMITTER_H

#include "WAMI/WAMIOps.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Builders.h"
#include "wasmstack/LocalAllocator.h"
#include "wasmstack/WasmStackOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::wasmstack {

/// Emits WasmStack operations from stackified WasmSSA/WAMI operations
class WasmStackEmitter {
  OpBuilder &builder;
  const LocalAllocator &allocator;
  const DenseSet<Value> &needsTee;

  /// Tracks which values have been emitted to the stack
  DenseSet<Value> emittedToStack;

  /// Sticky failure bit for fail-fast emission.
  bool failed = false;

  /// Counter for generating unique labels (member variable to avoid static)
  unsigned labelCounter = 0;

  /// Stack of labels for control flow structures (block/loop/if)
  /// Each entry is (label, isLoop) - isLoop determines branch behavior
  SmallVector<std::pair<std::string, bool>> labelStack;

public:
  WasmStackEmitter(OpBuilder &builder, const LocalAllocator &allocator,
                   const DenseSet<Value> &needsTee)
      : builder(builder), allocator(allocator), needsTee(needsTee) {}

  /// Emit a WasmStack function from a WasmSSA function
  FuncOp emitFunction(wasmssa::FuncOp srcFunc);

  /// Whether emission encountered a hard failure.
  bool hasFailed() const { return failed; }

  /// Emit a single operation
  void emitOperation(Operation *op);

  /// Emit operation and drop stack-resident results when all SSA results are
  /// unused.
  void emitOperationAndDropUnused(Operation *op);

private:
  /// Emit a constant operation
  void emitConst(wasmssa::ConstOp constOp);

  /// Emit a binary operation
  template <typename WasmStackOp>
  void emitBinaryOp(Operation *srcOp, Value lhs, Value rhs, Value result);

  /// Emit an operand value if it's not already on stack.
  /// If the value is on stack, mark it as CONSUMED (remove from emittedToStack)
  /// so that subsequent uses will fetch it via local.get.
  void emitOperandIfNeeded(Value value);

  /// Materialize SSA block arguments at control-flow region entry.
  /// Local-backed arguments are consumed from stack via local.set in reverse
  /// order; one-shot arguments remain stack-backed.
  void materializeEntryBlockArguments(Block &block);

  /// Generate a unique label for control flow structures
  std::string generateLabel(StringRef prefix);

  /// RAII guard to save/restore emittedToStack state for control flow regions
  class ScopedStackState;

  /// RAII guard for label stack management.
  class ScopedLabel;

  /// Get the branch label for a given exit level.
  std::string getLabelForExitLevel(unsigned exitLevel, Operation *contextOp);

  /// Emit a WasmSSA block operation
  void emitBlock(wasmssa::BlockOp blockOp);

  /// Emit a WasmSSA loop operation
  void emitLoop(wasmssa::LoopOp loopOp);

  /// Emit a WasmSSA if operation
  void emitIf(wasmssa::IfOp ifOp);

  /// Emit a WasmSSA branch_if operation
  void emitBranchIf(wasmssa::BranchIfOp branchIfOp);

  /// Handle a terminator operation and return the next block to process.
  Block *emitTerminatorAndGetNext(Operation *terminator, bool isInLoop);

  /// Emit a comparison operation (2 inputs, 1 i32 output)
  template <typename WasmStackOp>
  void emitCompareOp(Operation *srcOp, Value lhs, Value rhs, Value result);

  /// Emit a test operation (1 input, 1 i32 output)
  template <typename WasmStackOp>
  void emitTestOp(Operation *srcOp, Value input, Value result);

  /// Emit a unary operation (1 input, 1 output of same type)
  template <typename WasmStackOp>
  void emitUnaryOp(Operation *srcOp, Value input, Value result);

  /// Emit a signed/unsigned int-to-float conversion operation
  void emitConvertOp(Operation *srcOp, bool isSigned);

  /// Emit a promote operation (f32 to f64)
  void emitPromoteOp(wasmssa::PromoteOp promoteOp);

  /// Emit a demote operation (f64 to f32)
  void emitDemoteOp(wasmssa::DemoteOp demoteOp);

  /// Emit an i32 extension operation (i32 to i64)
  void emitExtendI32Op(Operation *srcOp, bool isSigned);

  /// Emit a wrap operation (i64 to i32)
  void emitWrapOp(wasmssa::WrapOp wrapOp);

  /// Emit a WAMI trunc operation (float to int)
  void emitTruncOp(Operation *srcOp, bool isSigned);

  /// Emit a WasmSSA local_get operation (from source dialect)
  void emitSourceLocalGet(wasmssa::LocalGetOp localGetOp);

  /// Emit a WasmSSA local_set operation (from source dialect)
  void emitSourceLocalSet(wasmssa::LocalSetOp localSetOp);

  /// Emit a WasmSSA local_tee operation (from source dialect)
  void emitSourceLocalTee(wasmssa::LocalTeeOp localTeeOp);

  /// Emit a wami.load operation
  void emitLoad(wami::LoadOp loadOp);

  /// Emit a wami.store operation
  void emitStore(wami::StoreOp storeOp);

  /// Emit a wasmssa.global_get operation
  void emitGlobalGet(wasmssa::GlobalGetOp globalGetOp);

  /// Emit a WasmSSA call operation
  void emitCall(wasmssa::FuncCallOp callOp);

  /// Materialize an operation result according to stackification policy:
  /// - `needsTee`: keep value on stack and mirror to local
  /// - local without `needsTee`: spill to local and keep stack clean
  /// - no local: keep value on stack
  void materializeResult(Location loc, Value result);

  /// Drop stack-resident results for operations whose SSA results are unused.
  void dropUnusedResults(Operation *op);

  /// Emit a hard error and mark the emitter as failed.
  void fail(Operation *op, StringRef message);
};

} // namespace mlir::wasmstack

#endif // WASMSTACK_WASMSTACKEMITTER_H
