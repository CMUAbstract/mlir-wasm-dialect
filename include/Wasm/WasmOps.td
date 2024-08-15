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

class Wasm_BinOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits>;

def Wasm_AddOp : Wasm_BinOp<"add", [Pure, SameOperandsAndResultType, Commutative]> {
    let summary = "Add two integers.";
    let description = [{
        The `wasm.add` operation adds two numeric values.
    }];

    // TODO: How to ensure that LHS and RHS are of the same type?
    let arguments = (ins AnyNumericType:$lhs, AnyNumericType:$rhs);
    let results = (outs AnyNumericType:$res);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs)
    }];
}

#endif // WASM_OPS
