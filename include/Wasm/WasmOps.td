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


def Wasm_ConstantOp : Wasm_Op<"constant", [ConstantLike, Pure]> {
    let summary = "Constant operation";
    let description = [{
        The `wasm.constant` operation represents a constant value.
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$res);

    let assemblyFormat = [{
        `(` $value `)` attr-dict `:` type($res)
    }];
}

class Wasm_UnaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits>;

class Wasm_BinaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits>;

// integer unary operations
class Wasm_IntegerUnaryOp<string mnemonic, list<Trait> traits = []>
    : Wasm_UnaryOp<mnemonic, traits> {
    let arguments = (ins AnyIntType:$input);
    let results = (outs AnyIntType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

// integer binary operations

class Wasm_IntegerBinaryOp<string mnemonic, list<Trait> traits = []> 
: Wasm_BinaryOp<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
    // TODO: How to ensure that LHS and RHS are of the same type?
    let arguments = (ins AnyIntType:$lhs, AnyIntType:$rhs);
    let results = (outs AnyIntType:$res);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs)
    }];
}

def Wasm_IAddOp : Wasm_IntegerBinaryOp<"add", [Commutative]> {
    let summary = "Add two integer values.";
    let description = [{
        The `wasm.add` operation adds two integer values.
    }];
}

def Wasm_ISubOp : Wasm_IntegerBinaryOp<"sub", []> {
    let summary = "Subtract a numeric value by another.";
    let description = [{
        The `wasm.sub` operation subtracts a numeric value by another.
    }];
}

def Wasm_IMulOp : Wasm_IntegerBinaryOp<"mul", [Commutative]> {
    let summary = "Multiply two integer values.";
    let description = [{
        The `wasm.mul` operation multiplies two integer values.
    }];
}

def Wasm_IDivSOp : Wasm_IntegerBinaryOp<"div_s", []> {
    let summary = "Signed division of two integer values.";
    let description = [{
        The `wasm.div_s` operation divides two integer values.
    }];
}

def Wasm_IDivUOp : Wasm_IntegerBinaryOp<"div_u", []> {
    let summary = "Unsigned division of two integer values.";
    let description = [{
        The `wasm.div_u` operation divides two integer values.
    }];
}

// floating-point unary operations

// floating-point binary operations

#endif // WASM_OPS
