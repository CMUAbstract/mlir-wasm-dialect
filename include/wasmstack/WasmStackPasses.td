//===- WasmStackPasses.td - WasmStack dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines passes for the WasmStack dialect, including the
// stackification pass that converts SsaWasm+WAMI to WasmStack.
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_PASSES
#define WASMSTACK_PASSES

include "mlir/Pass/PassBase.td"

def ConvertToWasmStack : Pass<"convert-to-wasmstack", "::mlir::ModuleOp"> {
  let summary = "Convert WasmSSA+WAMI dialects to WasmStack dialect";
  let description = [{
    This pass converts operations from the WasmSSA and WAMI dialects to the
    WasmStack dialect. It performs LLVM-style stackification:

    1. **Register Stackification**: Reorders operations to create single-use
       expression trees where definitions nest directly before their uses,
       minimizing the need for local variables.

    2. **CFG Stackification**: Converts SSA control flow to WebAssembly's
       structured control flow (block, loop, if/else).

    The pass implements three strategies:
    - **Single-use movement**: Move definitions immediately before their single use
    - **Rematerialization**: Clone cheap operations (constants) instead of using locals
    - **Tee insertion**: For multi-use values, use local.tee to duplicate values

    Reference: LLVM's WebAssemblyRegStackify.cpp and WebAssemblyCFGStackify.cpp
  }];
  let dependentDialects = [
    "wasmstack::WasmStackDialect",
    "wasmssa::WasmSSADialect",
    "wami::WAMIDialect"
  ];
}

#endif // WASMSTACK_PASSES
