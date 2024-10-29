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

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// high level ops

// intermittent.task
// intermittent.task.transition_to

// mid level ops

def IdempotentTaskOp : Intermittent_Op<"task.idempotent", [
    AffineScope, AutomaticAllocationScope, IsolatedFromAbove]> {
    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region AnyRegion:$body);
    let hasCustomAssemblyFormat = 1;
}

def TransitionToOp : Intermittent_Op<"task.transition_to", [
    Pure, HasParent<"IdempotentTaskOp">, Terminator]> {
    let arguments 
        = (ins OptionalAttr<FlatSymbolRefAttr>:$next_task, 
            Variadic<NonVolatile>:$vars_to_store);
    let assemblyFormat = [{
       ( $next_task^ )?
       ( `(` $vars_to_store^ `:` type($vars_to_store) `)` )?
       attr-dict
    }];
}

// low level ops

def NonVolatileNewOp : Intermittent_Op<"nonvolatile.new", [
    Pure]> { 
    let arguments = (ins TypeAttr:$inner);
    let results = (outs NonVolatile:$result);
    let hasCustomAssemblyFormat = 1;
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
