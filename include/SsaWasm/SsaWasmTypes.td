//===- SsaWasmTypes.td - SsaWasm dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SSAWASM_TYPES
#define SSAWASM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "SsaWasm/SsaWasmDialect.td"

/// Base type for SSA Wasm types.
class SsaWasm_Type<string name, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<SsaWasm_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// We treat this as i32 type with meta information.

/// SSA Wasm numeric type constraint, representing the union of integer and
/// floating-point types. 
def SsaWasm_WasmNumericLike : TypeConstraint<
    Or<[I<32>.predicate,
        I<64>.predicate,
        F<32>.predicate,
        F<64>.predicate,
        AnyMemRef.predicate]>,
    "ssawasm integer or float type">;

def SsaWasm_WasmIntegerLike : TypeConstraint<
    Or<[I<32>.predicate,
        I<64>.predicate]>,
    "ssawasm integer type">;

def SsaWasm_WasmFloatLike : TypeConstraint<
    Or<[F<32>.predicate,
        F<64>.predicate]>,
    "ssawasm float type">;

def SsaWasm_WasmPointerLike : TypeConstraint<
    Or<[AnyMemRef.predicate,
        I<32>.predicate,
        I<64>.predicate]>,
    "ssawasm memref or integer type">;

def SsaWasm_WasmFuncRefType : SsaWasm_Type<"WasmFuncRef", "funcref"> {
    let description = [{
        Represents a function reference type in the SSA Wasm dialect.
    }];
}

// stack switching related types


def SsaWasm_WasmContinuationType : SsaWasm_Type<"WasmContinuation", "cont"> {
    let description = [{
        Represents a continuation type in the SSA Wasm dialect.
    }];
    // TODO: Ideally we should use FunctionType here, but we use
    // StringAttr for now to allow recursive types.
    let parameters = (ins "StringAttr":$id);
    let assemblyFormat = "$id";
}

def SsaWasm_LocalLike : TypeConstraint<
    Or<[SsaWasm_WasmNumericLike.predicate,
        SsaWasm_WasmContinuationType.predicate]>,
    "ssawasm numeric or continuation type">;

#endif // SSAWASM_TYPES