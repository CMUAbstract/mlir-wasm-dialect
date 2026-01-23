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

def VerifyWasmStack : Pass<"verify-wasmstack", "::mlir::ModuleOp"> {
  let summary = "Verify WasmStack dialect operations have valid stack semantics";
  let description = [{
    Performs WebAssembly-spec compliant validation of stack operations:

    1. **Stack balance**: Operations consume/produce correct number of values
    2. **Type checking**: Operand types match expected types
    3. **Control flow validation**: Blocks/loops have balanced stacks
    4. **Function signature validation**: Returns match declared types
    5. **Branch target validation**: br/br_if target valid labels with correct types

    The pass implements WebAssembly's polymorphic typing rules:
    - After `unreachable`, `br`, `return`, or `br_table`, the stack becomes
      polymorphic and can match any expected type sequence
    - Block entry consumes parameters, block exit produces results
    - Loop branches go to loop start (using param_types), block branches
      go to block end (using result_types)
  }];
  let dependentDialects = ["wasmstack::WasmStackDialect"];
}

#endif // WASMSTACK_PASSES
