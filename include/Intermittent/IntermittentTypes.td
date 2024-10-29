//===- IntermittentTypes.td - Wasm dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INTERMITTENT_TYPES
#define INTERMITTENT_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "Intermittent/IntermittentDialect.td"

//===----------------------------------------------------------------------===//
// Intermittent type definitions
//===----------------------------------------------------------------------===//

class Intermittent_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Intermittent_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def NonVolatile : Intermittent_Type<"NonVolatile", "nonvolatile"> {
    let parameters = (ins AnyType:$elementType);
    let assemblyFormat = "`<` $elementType `>`";
    // let builders = [TypeBuilder<(ins "Type":$elementType)>];
}


#endif // INTERMITTENT_TYPES