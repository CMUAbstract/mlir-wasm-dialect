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

def ReplaceMultiUseOpsWithLocals : Pass<"replace-multi-use-ops-with-locals", "ModuleOp"> {
  let description = [{
    This pass replaces operations with multiple users with locals.
    This is a preparatory pass for stackification.
  }];

  let dependentDialects = [
    "ssawasm::SsaWasmDialect",
  ];
}

#endif // SSAWASM_PASS
