//===- WasmOps.td - Wasm dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASM_OPS
#define WASM_OPS

include "Wasm/WasmTypes.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// constant operator
class Wasm_ConstantOp<string typeName> 
    : Wasm_Op< typeName # ".constant", [Pure]> {
    let summary = "Constant operation";
    let description = [{
        The `wasm.constant` operation represents a constant value.
    }];

    let assemblyFormat = [{ $value attr-dict }];
}

def Wasm_I32ConstantOp : Wasm_ConstantOp<"i32"> {
    let arguments = (ins I32Type:$value);
}

def Wasm_I64ConstantOp : Wasm_ConstantOp<"i64"> {
    let arguments = (ins I64Type:$value);
}

def Wasm_F32ConstantOp : Wasm_ConstantOp<"f32"> {
    let arguments = (ins F32Type:$value);
}

def Wasm_F64ConstantOp : Wasm_ConstantOp<"f64"> {
    let arguments = (ins F64Type:$value);
}

class Wasm_UnaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits>;

class Wasm_BinaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits>;

// integer unary operations

// clz
// ctz
// popcnt

// integer binary operations

def Wasm_IAddOp : Wasm_BinaryOp<"i32.add", []> {}
def Wasm_ISubOp : Wasm_BinaryOp<"i32.sub", []> {}
def Wasm_IMulOp : Wasm_BinaryOp<"i32.mul", []> {}
def Wasm_IDivSOp : Wasm_BinaryOp<"i32.div_s", []> {}
def Wasm_IDivUOp : Wasm_BinaryOp<"i32.div_u", []> {}
def Wasm_IRemSOp : Wasm_BinaryOp<"i32.rem_s", []> {}
def Wasm_IRemUOp : Wasm_BinaryOp<"i32.rem_u", []> {}
def Wasm_IAndOp : Wasm_BinaryOp<"i32.and", []> {}
def Wasm_IOrOp : Wasm_BinaryOp<"i32.or", []> {}
def Wasm_IXorOp : Wasm_BinaryOp<"i32.xor", []> {}
def Wasm_IShlOp : Wasm_BinaryOp<"i32.shl", []> {}
def Wasm_IShrSOp : Wasm_BinaryOp<"i32.shr_s", []> {}
def Wasm_IShrUOp : Wasm_BinaryOp<"i32.shr_u", []> {}
def Wasm_IRotlOp : Wasm_BinaryOp<"i32.rotl", []> {}
def Wasm_IRotrOp : Wasm_BinaryOp<"i32.rotr", []> {}

// floating-point unary operations

// abs

// neg

// sqrt

// ceil

// floor

// trunc

// nearest

// floating-point binary operations

// add

// sub

// mul

// div

// min

// max

// copysign

// reference instructions

// parametric instructions

// variable instructions

// table instructions

// memory instructions

// constrol instructions


#endif // WASM_OPS
