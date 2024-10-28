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

def ConvertIntermittentToWasm : Pass<"convert-intermittent-to-wasm", "ModuleOp"> {
  let summary = "Converts Intermittent dialect to Wasm dialect";
  let dependentDialects = [
    "intermittent::IntermittentDialect",
  ];
}

#endif // INTERMITTENT_PASS
