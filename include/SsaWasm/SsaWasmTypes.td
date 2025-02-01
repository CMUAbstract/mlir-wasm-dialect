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
class SsaWasmType<string name, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<SsaWasm_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

/// SSA Wasm integer type. The parameter 'bitWidth' should be set to 32 or 64.
def SsaWasmIntegerType : SsaWasmType<"Integer", "integer"> {
    let description = [{
        Represents an integer type in the SSA Wasm dialect.
    }];
    let parameters = (ins "unsigned": $bitWidth);
    let hasCustomAssemblyFormat = 1;
}

/// SSA Wasm floating-point type. The parameter 'bitWidth' should be 32, 64.
def SsaWasmFloatType : SsaWasmType<"Float", "float"> {
    let description = [{
        Represents a floating-point type in the SSA Wasm dialect.
    }];
    let parameters = (ins "unsigned": $bitWidth);
    let hasCustomAssemblyFormat = 1;
}

/// SSA Wasm numeric type constraint, representing the union of integer and
/// floating-point types. 
def SsaWasmNumericLike : TypeConstraint<
    Or<[SsaWasmIntegerType.predicate,
        SsaWasmFloatType.predicate]>,
    "ssawasm integer or float type">;

#endif // SSAWASM_TYPES