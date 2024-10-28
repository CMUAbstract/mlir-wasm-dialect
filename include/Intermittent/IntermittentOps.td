//===- IntermittentOps.td - Wasm dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INTERMITTENT_OPS
#define INTERMITTENT_OPS

include "Intermittent/IntermittentTypes.td"

include "mlir/Bytecode/BytecodeOpInterface.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// high level ops

// intermittent.task
// intermittent.task.transition_to

// mid level ops

// intermittent.task.idempotent
// intermittent.var.local

// low level ops

def NonVolatileNewOp : Intermittent_Op<"nonvolatile.new", [
    Pure]> { 
    let arguments = (ins TypeAttr:$type);
    let results = (outs NonVolatile:$result);
    let assemblyFormat = "`<` type($result) `>` attr-dict";
}

def NonVolatileLoadOp : Intermittent_Op<"nonvolatile.load", [
    Pure,
    TypesMatchWith<"result type matches element type of 'var'",
                     "var", "result",
                     "::llvm::cast<NonVolatileType>($_self).getElementType()">,
]> { 
    let arguments = (ins NonVolatile:$var);
    let results = (outs AnyType:$result); 
    let assemblyFormat = "$var attr-dict `:` type($var)";
}

def NonVolatileStoreOp : Intermittent_Op<"nonvolatile.store", [
    TypesMatchWith<"type of 'value' matches element type of 'var'",
                     "var", "value", 
                     "::llvm::cast<NonVolatileType>($_self).getElementType()">,
]> {
    let arguments = (ins NonVolatile:$var, AnyType:$value);
      let assemblyFormat = [{
        $value `,` $var attr-dict `:` type($var)
    }];

}

// intermittent.commit_vars

#endif // INTERMITTENT_OPS
