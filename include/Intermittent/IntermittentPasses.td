//===- IntermittentPass.td - Wasm dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INTERMITTENT_PASS
#define INTERMITTENT_PASS

include "mlir/Pass/PassBase.td"

def PrepareForIntermittent : Pass<"prepare-for-intermittent", "ModuleOp"> {
  let summary = "Add preludes";
  let dependentDialects = [
    "intermittent::IntermittentDialect",
    "wasm::WasmDialect"
  ];
}

def ConvertIntermittentTaskToWasm : Pass<"convert-intermittent-task-to-wasm", "ModuleOp"> {
  let summary = "Converts Intermittent dialect to Wasm dialect";
  let dependentDialects = [
    "intermittent::IntermittentDialect",
    "wasm::WasmDialect"
  ];
}


#endif // INTERMITTENT_PASS
