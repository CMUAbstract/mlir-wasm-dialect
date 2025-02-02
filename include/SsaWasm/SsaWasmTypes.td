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
    let assemblyFormat = "`i` $bitWidth";
}

/// SSA Wasm floating-point type. The parameter 'bitWidth' should be 32, 64.
def SsaWasm_WasmFloatType : SsaWasm_Type<"WasmFloat", "float"> {
    let description = [{
        Represents a floating-point type in the SSA Wasm dialect.
    }];
    let parameters = (ins "unsigned": $bitWidth);
    let assemblyFormat = "`f` $bitWidth";
}

/// SSA Wasm numeric type constraint, representing the union of integer and
/// floating-point types. 
def SsaWasm_WasmNumericLike : TypeConstraint<
    Or<[SsaWasm_WasmIntegerType.predicate,
        SsaWasm_WasmFloatType.predicate]>,
    "ssawasm integer or float type">;

#endif // SSAWASM_TYPES