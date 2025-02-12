//===- DContTypes.td - DCont dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DCONT_DCONT_TYPES
#define DCONT_DCONT_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "DCont/DContDialect.td"

/// Base type for DCont types.
class DCont_Type<string name, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<DCont_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def DCont_ContType : DCont_Type<"Cont", "cont"> {
    let parameters = (ins AnyType:$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}


#endif // DCONT_DCONT_TYPES