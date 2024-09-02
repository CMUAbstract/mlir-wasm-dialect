//===- WasmPass.td - Wasm dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASM_PASS
#define WASM_PASS

include "mlir/Pass/PassBase.td"

def ConvertToWasm : Pass<"convert-to-wasm", "ModuleOp"> {
  let summary = "Converts Arith dialect to Wasm dialect";
  let description = [{
    Converts Arith dialect to Wasm dialect.
  }];
  let dependentDialects = [
    "wasm::WasmDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
  // TODO: Add scf and memref dialects
  ];
}

def WasmFinalize : Pass<"wasm-finalize", "func::FuncOp"> {
  let summary = "Finalizes Wasm dialect";
  let description = [{
    Transforms temporary local variable accesses into index accesses.
  }];
  let dependentDialects = [
    "wasm::WasmDialect",
  ];
}

#endif // WASM_PASS
