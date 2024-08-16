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
include "mlir/IR/CommonAttrConstraints.td"

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
    let arguments = (ins I32Attr:$value);
}

def Wasm_I64ConstantOp : Wasm_ConstantOp<"i64"> {
    let arguments = (ins I64Attr:$value);
}

def Wasm_F32ConstantOp : Wasm_ConstantOp<"f32"> {
    let arguments = (ins F32Attr:$value);
}

def Wasm_F64ConstantOp : Wasm_ConstantOp<"f64"> {
    let arguments = (ins F64Attr:$value);
}

class Wasm_UnaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits> {
    let assemblyFormat = [{ attr-dict }];
}

class Wasm_BinaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits> {
    let assemblyFormat = [{ attr-dict }];
}

// integer unary operations

// clz
// ctz
// popcnt

// integer binary operations

// TODO: Add i64, f32, f64
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

def Wasm_RefNullOp : Wasm_Op<"ref.null", []> {
    let summary = "Reference null operation";
    let description = [{
        The `ref.null` operation creates a null reference.
    }];

    let arguments = (ins RefType:$ref);
}

def Wasm_RefIsNullOp : Wasm_Op<"ref.is_null", []> {
    let summary = "Reference is null operation";
    let description = [{
        The `ref.is_null` operation checks if a reference is null.
    }];
}

def Wasm_RefFuncOp : Wasm_Op<"ref.func", []> {
    let summary = "Reference function operation";
    let description = [{
        The `ref.func` operation creates a reference to a function.
    }];
    let arguments = (ins FuncIdxType:$idx);
}

// parametric instructions

def Wasm_DropOp : Wasm_Op<"drop", []> {
    let summary = "Drop operation";
    let description = [{
        The `drop` operation removes a value from the stack.
    }];
}

def Wasm_SelectOp : Wasm_Op<"select", []> {
    let summary = "Select operation";
    let description = [{
        The `select` operation selects between two values based on a condition.
    }];

    let arguments = (ins AnyType:$arg); // TODO
}

// variable instructions

def Wasm_LocalGetOp : Wasm_Op<"local.get", []> {
    let summary = "Local get operation";
    let description = [{
        The `local.get` operation reads a local variable.
    }];
    let arguments = (ins LocalIdxType:$idx);
}

def Wasm_LocalSetOp : Wasm_Op<"local.set", []> {
    let summary = "Local set operation";
    let description = [{
        The `local.set` operation writes a local variable.
    }];
    let arguments = (ins LocalIdxType:$idx);
}

def Wasm_LocalTeeOp : Wasm_Op<"local.tee", []> {
    let summary = "Local tee operation";
    let description = [{
        The `local.tee` operation writes a local variable and returns the value.
    }];
    let arguments = (ins LocalIdxType:$idx);
}

def Wasm_GlobalGetOp : Wasm_Op<"global.get", []> {
    let summary = "Global get operation";
    let description = [{
        The `global.get` operation reads a global variable.
    }];
    let arguments = (ins GlobalIdxType:$idx);
}

def Wasm_GlobalSetOp : Wasm_Op<"global.set", []> {
    let summary = "Global set operation";
    let description = [{
        The `global.set` operation writes a global variable.
    }];
    let arguments = (ins GlobalIdxType:$idx);
}

// table instructions

def Wasm_TableGetOp : Wasm_Op<"table.get", []> {
    let summary = "Table get operation";
    let description = [{
        The `table.get` operation reads a table element.
    }];
    let arguments = (ins TableIdxType:$idx);
}

def Wasm_TableSetOp : Wasm_Op<"table.set", []> {
    let summary = "Table set operation";
    let description = [{
        The `table.set` operation writes a table element.
    }];
    let arguments = (ins TableIdxType:$idx);
}

def Wasm_TableSizeOp : Wasm_Op<"table.size", []> {
    let summary = "Table size operation";
    let description = [{
        The `table.size` operation returns the size of a table.
    }];
}

def Wasm_TableGrowOp : Wasm_Op<"table.grow", []> {
    let summary = "Table grow operation";
    let description = [{
        The `table.grow` operation grows a table.
    }];
}

def Wasm_TableFillOp : Wasm_Op<"table.fill", []> {
    let summary = "Table fill operation";
    let description = [{
        The `table.fill` operation fills a range of a table.
    }];
}

def Wasm_TableCopyOp : Wasm_Op<"table.copy", []> {
    let summary = "Table copy operation";
    let description = [{
        The `table.copy` operation copies a range of a table.
    }];
}

def Wasm_TableInitOp : Wasm_Op<"table.init", []> {
    let summary = "Table init operation";
    let description = [{
        The `table.init` operation initializes a range of a table.
    }];
}

def Wasm_ElemDropOp : Wasm_Op<"elem.drop", []> {
    let summary = "Element drop operation";
    let description = [{
        The `elem.drop` operation drops a range of elements.
    }];
}

// memory instructions

// load

// store

// constrol instructions

def Wasm_Nop : Wasm_Op<"nop", []> {
    let summary = "No operation";
    let description = [{
        The `nop` operation does nothing.
    }];
}

def Wasm_Unreachable : Wasm_Op<"unreachable", []> {
    let summary = "Unreachable operation";
    let description = [{
        The `unreachable` operation is used to indicate that the program
        has reached a state that should never be reached.
    }];
}

// TODO: control flow instructions

#endif // WASM_OPS
