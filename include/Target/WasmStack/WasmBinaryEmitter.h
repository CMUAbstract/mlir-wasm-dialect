//===- WasmBinaryEmitter.h - Top-level binary emitter -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_WASMSTACK_WASMBINARYEMITTER_H
#define TARGET_WASMSTACK_WASMBINARYEMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::wasmstack {

/// Emit a WasmStack module as a WebAssembly binary (.wasm) to the given stream.
/// When relocatable is true, emits a relocatable object file (.o) with
/// linking and relocation custom sections for use with wasm-ld.
/// Returns success/failure.
mlir::LogicalResult emitWasmBinary(Operation *moduleOp,
                                   llvm::raw_ostream &output,
                                   bool relocatable = false);

} // namespace mlir::wasmstack

#endif // TARGET_WASMSTACK_WASMBINARYEMITTER_H
