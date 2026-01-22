//===- WasmStackPasses.h - WasmStack dialect passes -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes for the WasmStack dialect, including the
// stackification pass that converts WasmSSA+WAMI to WasmStack.
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_WASMSTACKPASSES_H
#define WASMSTACK_WASMSTACKPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::wasmstack {

#define GEN_PASS_DECL
#include "wasmstack/WasmStackPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "wasmstack/WasmStackPasses.h.inc"

/// Registers all WasmStack passes.
inline void registerPasses() { registerWasmStackPasses(); }

} // namespace mlir::wasmstack

#endif // WASMSTACK_WASMSTACKPASSES_H
