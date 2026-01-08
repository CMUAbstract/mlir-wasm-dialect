//===- WAMITypes.td - WAMI dialect types ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_TYPES
#define WAMI_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "WAMI/WAMIDialect.td"

/// Base class for WAMI types.
class WAMI_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WAMI_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Continuation Types (for WebAssembly Stack Switching proposal)
//===----------------------------------------------------------------------===//

def WAMI_ContType : WAMI_Type<"Cont", "cont"> {
  let summary = "Continuation type for stack switching";
  let description = [{
    Represents a continuation type in the WAMI dialect.
    Continuations are used for the WebAssembly Stack Switching proposal.
    The type is parametrized by a type ID string that refers to a
    declared continuation type.
  }];
  let parameters = (ins "StringAttr":$id);
  let assemblyFormat = "$id";
}

#endif // WAMI_TYPES
