//===- WAMIOps.td - WAMI dialect operations -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_OPS
#define WAMI_OPS

include "WAMI/WAMITypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class WAMI_Op<string mnemonic, list<Trait> traits = []>
    : Op<WAMI_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Core WebAssembly Operations
//===----------------------------------------------------------------------===//

def WAMI_SelectOp : WAMI_Op<"select", [
    AllTypesMatch<["true_value", "false_value", "result"]>,
    Pure
]> {
  let summary = "Select between two values based on condition";
  let description = [{
    WebAssembly select instruction. Returns true_value if condition is non-zero,
    false_value otherwise.

    This corresponds to the WebAssembly `select` instruction which provides
    a ternary conditional without control flow.
  }];
  let arguments = (ins
    AnyType:$true_value,
    AnyType:$false_value,
    I32:$condition
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Stack Switching Operations (WebAssembly Stack Switching Proposal)
//
// These operations will be added incrementally as needed:
// - wami.tag           : Tag declaration for continuations
// - wami.cont_new      : Create continuation from function ref
// - wami.resume        : Resume a continuation
// - wami.suspend       : Suspend current continuation
// - wami.null_cont_ref : Create null continuation reference
//===----------------------------------------------------------------------===//

// Placeholder: Operations will be added in Phase 6

#endif // WAMI_OPS
