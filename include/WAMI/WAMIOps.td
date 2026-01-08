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
