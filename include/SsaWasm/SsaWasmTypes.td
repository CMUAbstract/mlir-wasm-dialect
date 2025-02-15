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

/// SSA Wasm integer type. The parameter 'bitWidth' should be set to 32 or 64.
def SsaWasm_WasmIntegerType : SsaWasm_Type<"WasmInteger", "integer"> {
    let description = [{
        Represents an integer type in the SSA Wasm dialect.
    }];
    let parameters = (ins "unsigned": $bitWidth);
    // NOTE: assemblyFormat = "`i`$bitWidth" prints `i 32` instead of `i32` so
    let assemblyFormat = "$bitWidth";
}

/// SSA Wasm floating-point type. The parameter 'bitWidth' should be 32, 64.
def SsaWasm_WasmFloatType : SsaWasm_Type<"WasmFloat", "float"> {
    let description = [{
        Represents a floating-point type in the SSA Wasm dialect.
    }];
    let parameters = (ins "unsigned": $bitWidth);
    // NOTE: assemblyFormat = "`f`$bitWidth" prints `f 32` instead of `f32` so
    // we use the following format
    let assemblyFormat = "$bitWidth";
}

// We treat this as i32 type with meta information.
def SsaWasm_WasmMemRefType : SsaWasm_Type<"WasmMemRef", "memref"> {
    let description = [{
        Represents a memory reference type in the SSA Wasm dialect.
        This is an i32 type with meta information.
    }];
    let parameters = (ins AnyMemRef:$memRefType);
    let assemblyFormat = "$memRefType";
}

/// SSA Wasm numeric type constraint, representing the union of integer and
/// floating-point types. 
def SsaWasm_WasmNumericLike : TypeConstraint<
    Or<[SsaWasm_WasmIntegerType.predicate,
        SsaWasm_WasmFloatType.predicate,
        SsaWasm_WasmMemRefType.predicate]>,
    "ssawasm integer or float type">;

def SsaWasm_WasmPointerLike : TypeConstraint<
    Or<[SsaWasm_WasmMemRefType.predicate,
        SsaWasm_WasmIntegerType.predicate]>,
    "ssawasm memref or integer type">;

// stack switching related types


def SsaWasm_WasmContinuationType : SsaWasm_Type<"WasmContinuation", "cont"> {
    let description = [{
        Represents a continuation type in the SSA Wasm dialect.
    }];
    // TODO: Ideally we should use FunctionType here, but we use
    // StringAttr for now to allow recursive types.
    let parameters = (ins "StringAttr":$function_name);
    let assemblyFormat = "$function_name";
}

#endif // SSAWASM_TYPES