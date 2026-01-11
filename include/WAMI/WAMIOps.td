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
// Memory Operations
//===----------------------------------------------------------------------===//

def WAMI_LoadOp : WAMI_Op<"load", [Pure]> {
  let summary = "Load value from linear memory";
  let description = [{
    Load a value from WebAssembly linear memory at the given address.
    The address is a 32-bit integer (i32) representing a byte offset
    into linear memory.

    Example:
    ```mlir
    %value = wami.load %addr : i32
    %float = wami.load %addr : f32
    ```
  }];
  let arguments = (ins I32:$address);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$address attr-dict `:` type($result)";
}

def WAMI_StoreOp : WAMI_Op<"store", []> {
  let summary = "Store value to linear memory";
  let description = [{
    Store a value to WebAssembly linear memory at the given address.
    The address is a 32-bit integer (i32) representing a byte offset
    into linear memory.

    Example:
    ```mlir
    wami.store %addr, %value : i32
    wami.store %addr, %float : f32
    ```
  }];
  let arguments = (ins I32:$address, AnyType:$value);
  let assemblyFormat = "$address `,` $value attr-dict `:` type($value)";
}

def WAMI_DataOp : WAMI_Op<"data", [Symbol]> {
  let summary = "Declare initialized data section";
  let description = [{
    Declares a data section with initial values at a specific offset
    in linear memory. This corresponds to WebAssembly's data segment.

    Example:
    ```mlir
    wami.data @arr_data = dense<[1, 2, 3, 4]> : tensor<4xi32> at 1024
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ElementsAttr:$value,
    I32Attr:$offset
  );
  let assemblyFormat = "$sym_name `=` $value `at` $offset attr-dict";
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
