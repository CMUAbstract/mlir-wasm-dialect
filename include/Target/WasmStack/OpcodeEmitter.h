//===- OpcodeEmitter.h - WasmStack op to binary opcode emitter --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_WASMSTACK_OPCODEEMITTER_H
#define TARGET_WASMSTACK_OPCODEEMITTER_H

#include "Target/WasmStack/BinaryWriter.h"
#include "Target/WasmStack/IndexSpace.h"
#include "Target/WasmStack/RelocationTracker.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::wasmstack {

/// Translates WasmStack operations into binary WebAssembly opcodes.
/// Manages a label stack for resolving symbolic branch targets to depth
/// indices.
class OpcodeEmitter {
public:
  OpcodeEmitter(BinaryWriter &writer, IndexSpace &indexSpace,
                RelocationTracker *tracker = nullptr,
                uint32_t sectionOffset = 0)
      : writer(writer), indexSpace(indexSpace), tracker(tracker),
        sectionOffset(sectionOffset) {}

  /// Emit all operations in a function body (excluding the function frame).
  /// Returns true on success.
  bool emitFunctionBody(Operation *funcOp);

  /// Emit a single operation. Returns true on success.
  bool emitOperation(Operation *op);

private:
  /// RAII helper to push/pop labels on the label stack.
  struct ScopedLabel {
    OpcodeEmitter &emitter;
    ScopedLabel(OpcodeEmitter &emitter, llvm::StringRef label, bool isLoop)
        : emitter(emitter) {
      emitter.labelStack.push_back({label.str(), isLoop});
    }
    ~ScopedLabel() { emitter.labelStack.pop_back(); }
  };

  /// Resolve a symbolic label to a branch depth index.
  uint32_t resolveLabelDepth(llvm::StringRef label) const;

  /// Emit block type encoding for a block/loop/if.
  bool emitBlockType(ArrayAttr paramTypes, ArrayAttr resultTypes);

  // Emit helpers for specific op categories
  bool emitConstOp(Operation *op);
  bool emitLocalOp(Operation *op);
  bool emitGlobalOp(Operation *op);
  bool emitArithmeticOp(Operation *op);
  bool emitCompareOp(Operation *op);
  bool emitConversionOp(Operation *op);
  bool emitControlFlowOp(Operation *op);
  bool emitMemoryOp(Operation *op);
  bool emitCallOp(Operation *op);
  bool emitStackSwitchingOp(Operation *op);
  bool emitMiscOp(Operation *op);

  /// Get the type-dispatched opcode for a binary/unary/compare operation.
  uint8_t getTypedOpcode(llvm::StringRef opName, mlir::Type type) const;

  BinaryWriter &writer;
  IndexSpace &indexSpace;
  RelocationTracker *tracker;
  uint32_t sectionOffset;

  /// Label stack: (label_name, is_loop) pairs.
  /// Labels are pushed when entering block/loop/if and popped on exit.
  llvm::SmallVector<std::pair<std::string, bool>> labelStack;
};

} // namespace mlir::wasmstack

#endif // TARGET_WASMSTACK_OPCODEEMITTER_H
