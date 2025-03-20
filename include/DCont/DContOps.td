
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
    let arguments = (ins FlatSymbolRefAttr:$function_name);
    let results = (outs DCont_ContType:$cont);
    let assemblyFormat = "attr-dict $function_name `:` type($cont)";
}

def DCont_StorageOp : DCont_Op<"storage"> {
    let summary = "create a storage for a delimited continuation";
    let description = [{
        The `storage` operation creates a storage for a delimited continuation.
    }];
    let results = (outs DCont_StorageType:$storage);
    let assemblyFormat = "attr-dict `:` type($storage)";
}

def DCont_LoadOp : DCont_Op<"load"> {
    let summary = "load a value from a storage";
    let description = [{
        The `load` operation loads a value from a storage.
    }];
    let arguments = (ins DCont_StorageType:$storage);
    let results = (outs DCont_ContType:$cont);
    let assemblyFormat = "attr-dict $storage `:` type($storage) `->` type($cont)";
}

def DCont_StoreOp : DCont_Op<"store"> {
    let summary = "store a value to a storage";
    let description = [{
        The `store` operation stores a value to a storage.
    }];
    let arguments = (ins DCont_StorageType:$storage, DCont_ContType:$cont);
    let assemblyFormat = "attr-dict $storage `,` $cont `:` type($cont) `->` type($storage)";
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

    // TODO: We should support suspend handler with multiple blocks
    let regions = (region SizedRegion<1>:$suspend_handler);
}

def DCont_SuspendHandlerReturnOp : DCont_Op<"suspend_handler_return", [Terminator]>{
    let summary = "return from suspend handler";
    let assemblyFormat = "attr-dict";
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

#endif // DCONT_OPS