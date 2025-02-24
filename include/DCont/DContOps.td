
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
    let arguments = (ins FlatSymbolRefAttr:$function_type_name);
    let results = (outs DCont_ContType:$cont);
    let assemblyFormat = "attr-dict $function_type_name `:` type($cont)";
}

def DCont_NullContOp : DCont_Op<"null_cont"> {
    let summary = "null delimited continuation";
    let description = [{
        The `null_cont` operation represents a null delimited continuation.
        Resume a `null_cont` operation is a no-op.
    }];
    let results = (outs DCont_ContType:$cont);
}

def DCont_ResumeOp : DCont_Op<"resume"> {
    let summary = "resume a delimited continuation";
    let description = [{
        The `resume` operation resumes a delimited continuation.
    }];
    let arguments = (ins DCont_ContType:$cont, Variadic<AnyType>:$args);
    let results = (outs Variadic<AnyType>:$results);
    // TODO: We should support suspend handler with multiple blocks
    let regions = (region SizedRegion<1>:$suspend_handler);
}

def DCont_SuspendHandlerTerminator : DCont_Op<"suspend_handler_terminator", [Terminator]>{
    let summary = "terminator for suspend handler";
}


def DCont_SuspendOp : DCont_Op<"suspend"> {
    let summary = "suspend a delimited continuation";
    let description = [{
        The `suspend` operation suspends a delimited continuation.
    }];
    let arguments = (ins Variadic<AnyType>:$args);
    let results = (outs Variadic<AnyType>:$results);
    let assemblyFormat = [{
        `(` $args `)` attr-dict `:` functional-type($args, $results)
  }];
  
}

// FIXME: This is not a good modeling
def DCont_ResumeSwitchOp : DCont_Op<"resume_switch"> {
    let summary = "resume a delimited continuation so that childs can switch to each other";
    let description = [{
        The `resume_switch` operation resumes a delimited continuation.
        The resumed continuation can switch to other continuations.
    }];
    let arguments = (ins DCont_ContType:$cont, Variadic<AnyType>:$args);
    let results = (outs DCont_ContType:$returned_cont, Variadic<AnyType>:$results);
    // TODO: add assembly format
}

// FIXME: This is not a good modeling
def DCont_SwitchOp : DCont_Op<"switch"> {
    let summary = "switch to a different delimited continuation";
    let description = [{
        The `switch` operation switches to a different delimited continuation.
    }];
    let arguments = (ins DCont_ContType:$cont, Variadic<AnyType>:$args);
    let results = (outs DCont_ContType:$returned_cont, Variadic<AnyType>:$results);
    // TODO: add assembly format
}

#endif // DCONT_OPS