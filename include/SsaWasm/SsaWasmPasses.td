//===- WasmPass.td - Wasm dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SSAWASM_PASS
#define SSAWASM_PASS

include "mlir/Pass/PassBase.td"

def ConvertToSsaWasm : Pass<"convert-to-ssawasm", "ModuleOp"> {
  let summary = "Converts to SsaWasm dialect";
  let dependentDialects = [
    "ssawasm::SsaWasmDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
  ];
}

def IntroduceLocals : Pass<"introduce-locals", "ModuleOp"> {
  let summary = "Introduces locals";
  let description = [{
    This pass introduces locals for operations with multiple users or operations
    that are not stackifiable.
    This is a preparatory pass for stackification.
  }];

  let dependentDialects = [
    "ssawasm::SsaWasmDialect",
  ];
}

def ConvertSsaWasmGlobalToWasm : Pass<"convert-ssawasm-global-to-wasm", "ModuleOp"> {
  let summary = "Converts module-level operations such as Data and GetData to the wasm dialect";
  let dependentDialects = [
    "ssawasm::SsaWasmDialect",
    "wasm::WasmDialect",
  ];
}

def ConvertSsaWasmToWasm : Pass<"convert-ssawasm-to-wasm", "ModuleOp"> {
  let summary = "Converts SsaWasm dialect to Wasm dialect";
  let dependentDialects = [
    "ssawasm::SsaWasmDialect",
    "wasm::WasmDialect",
  ];
}

#endif // SSAWASM_PASS
