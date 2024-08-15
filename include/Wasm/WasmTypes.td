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
include "Wasm/WasmDialect.td"

//===----------------------------------------------------------------------===//
// Wasm type definitions
//===----------------------------------------------------------------------===//

class Wasm_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Wasm_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}


class Wasm_IntType<string name, string mnemonic> 
    : Wasm_Type<name, mnemonic, /*traits=*/[]> {
}

def Wasm_I32 : Wasm_IntType<"i32", "i32"> {
    let summary = "Wasm i32 type";
    let description = "i32 type in wasm dialect";
}

def Wasm_I64 : Wasm_IntType<"i64", "i64"> {
    let summary = "Wasm i64 type";
    let description = "i64 type in wasm dialect";
}


class Wasm_FloatType<string name, string mnemonic> 
    : Wasm_Type<name, mnemonic, /*traits=*/[]> {
}

def Wasm_F32 : Wasm_FloatType<"f32", "f32"> {
    let summary = "Wasm f32 type";
    let description = "f32 type in wasm dialect";
}

def Wasm_F64 : Wasm_FloatType<"f64", "f64"> {
    let summary = "Wasm f64 type";
    let description = "f64 type in wasm dialect";
}

def AnyIntType : TypeConstraint<Or<[Wasm_I32.predicate, Wasm_I64.predicate]>,
    "any integer">;

def AnyFloatType : TypeConstraint<Or<[Wasm_F32.predicate, Wasm_F64.predicate]>,
    "any float">;

def AnyNumericType : TypeConstraint<Or<[Wasm_I32.predicate,
    Wasm_I64.predicate, Wasm_F32.predicate, Wasm_F64.predicate]>,
    "any numeric">;


#endif // WASM_TYPES