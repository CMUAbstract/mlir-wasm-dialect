
//===- DContOps.td - DCont dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DCONT_OPS
#define DCONT_OPS

include "DCont/DContTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/Constraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class DCont_Op<string mnemonic, list<Trait> traits = []>
    : Op<DCont_Dialect, mnemonic, traits>;

def DCont_NewOp : DCont_Op<"new"> {
    let summary = "create a new delimited continuation";
    let description = [{
        The `new` operation creates a new delimited continuation.
    }];
    
}

#endif // DCONT_OPS