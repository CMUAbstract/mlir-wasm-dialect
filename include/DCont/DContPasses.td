//===- DContPass.td - DCont dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DCONT_DCONT_PASS
#define DCONT_DCONT_PASS

include "mlir/Pass/PassBase.td"

def ConvertDContToSsaWasm : Pass<"convert-dcont-to-ssawasm", "ModuleOp"> {
  let summary = "Convert delimited continuations to SSA Wasm";
  let description = [{
    This pass converts delimited continuations to SSA Wasm.
  }];
  let dependentDialects = [
    "dcont::DContDialect",
    "ssawasm::SsaWasmDialect",
  ];
}

#endif // DCONT_DCONT_PASS
