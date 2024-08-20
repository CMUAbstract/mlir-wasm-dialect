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

def ArithToWasmPass : Pass<"arith-to-wasm"> {
  let summary = "Converts Arith dialect to Wasm dialect";
  let description = [{
    Converts Arith dialect to Wasm dialect.
  }];
}

#endif // WASM_PASS
