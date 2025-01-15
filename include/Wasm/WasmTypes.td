//===- WasmTypes.td - Wasm dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASM_TYPES
#define WASM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "Wasm/WasmDialect.td"

//===----------------------------------------------------------------------===//
// Wasm type definitions
//===----------------------------------------------------------------------===//

class Wasm_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Wasm_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Define a container type that holds elements of a specified type.
def Local : Wasm_Type<"Local", "local"> {
   let parameters = (ins AnyType:$inner);
   let assemblyFormat = "`<` $inner `>`";
}

def Global : Wasm_Type<"Global", "global"> {
   let parameters = (ins AnyType:$inner);
   let assemblyFormat = "`<` $inner `>`";
}

def RefType : Type<CPred<"true">, "reference type">;

def Continuation : Wasm_Type<"Continuation", "continuation"> {
  let parameters = (ins "StringAttr":$name,
                           "StringAttr":$funcId);
  let assemblyFormat = [{
   $name `(` `cont` $funcId `)`
  }];

  let builders = [
   TypeBuilder<(ins "StringRef":$name, "StringRef":$funcId), [{
   return $_get($_ctxt, 
      StringAttr::get($_ctxt, name), 
      StringAttr::get($_ctxt, funcId));
   }]>
  ];
}


#endif // WASM_TYPES