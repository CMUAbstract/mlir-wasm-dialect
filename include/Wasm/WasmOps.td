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
include "mlir/IR/EnumAttr.td"
include "mlir/IR/Constraints.td"

def Wasm_ConstantOp : Wasm_Op<"constant", [Pure]> {
    let arguments = (ins AnyAttr:$value);
    let assemblyFormat = "attr-dict $value";
    let hasVerifier = 1;
}

class Wasm_UnaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits> {
    let assemblyFormat = [{ attr-dict }];
}

class Wasm_BinaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits> {
    // wasm binary operations do not explicitly take arguments.
    // Arguments are implicitly taken from the stack
    let arguments = (ins TypeAttr:$type);
    let assemblyFormat = [{ attr-dict `:` $type }];
    // TODO: Add verifier
    // probably we should split this into two classes: one for integer and one
    // for floating-point
}

// integer unary operations

// clz
// ctz
// popcnt

// integer binary operations

def Wasm_AddOp : Wasm_BinaryOp<"add">;
def Wasm_SubOp : Wasm_BinaryOp<"sub">;
def Wasm_MulOp : Wasm_BinaryOp<"mul">;
def Wasm_DivSOp : Wasm_BinaryOp<"div_s">;
def Wasm_DivUOp : Wasm_BinaryOp<"div_u">;
def Wasm_RemSOp : Wasm_BinaryOp<"rem_s">;
def Wasm_RemUOp : Wasm_BinaryOp<"rem_u">;
def Wasm_AndOp : Wasm_BinaryOp<"and">;
def Wasm_OrOp : Wasm_BinaryOp<"or">;
def Wasm_XorOp : Wasm_BinaryOp<"xor">;
def Wasm_ShlOp : Wasm_BinaryOp<"shl">;
def Wasm_ShrSOp : Wasm_BinaryOp<"shr_s">;
def Wasm_ShrUOp : Wasm_BinaryOp<"shr_u">;
def Wasm_RotlOp : Wasm_BinaryOp<"rotl">;
def Wasm_RotrOp : Wasm_BinaryOp<"rotr">;

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
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
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

def Wasm_LocalOp : Wasm_Op<"local", []> {
    let summary = "Local declaration operation";
    let description = [{
        The `local` operation declares local variables.
    }];
    // TODO: arguments should be a list of types
}

def Wasm_LocalGetOp : Wasm_Op<"local.get", []> {
    let summary = "Local get operation";
    let description = [{
        The `local.get` operation reads a local variable.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_LocalSetOp : Wasm_Op<"local.set", []> {
    let summary = "Local set operation";
    let description = [{
        The `local.set` operation writes a local variable.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_LocalTeeOp : Wasm_Op<"local.tee", []> {
    let summary = "Local tee operation";
    let description = [{
        The `local.tee` operation writes a local variable and returns the value.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_GlobalGetOp : Wasm_Op<"global.get", []> {
    let summary = "Global get operation";
    let description = [{
        The `global.get` operation reads a global variable.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_GlobalSetOp : Wasm_Op<"global.set", []> {
    let summary = "Global set operation";
    let description = [{
        The `global.set` operation writes a global variable.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

// table instructions

def Wasm_TableGetOp : Wasm_Op<"table.get", []> {
    let summary = "Table get operation";
    let description = [{
        The `table.get` operation reads a table element.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_TableSetOp : Wasm_Op<"table.set", []> {
    let summary = "Table set operation";
    let description = [{
        The `table.set` operation writes a table element.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
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
def Wasm_LoadOp : Wasm_Op<"i32.load", []> {
    let summary = "Memory load operation";
    let description = [{
        The `memory.load` operation reads from memory.
    }];
    let arguments = (ins I32Attr:$offset, I32Attr:$align);
    let assemblyFormat = [{ 
        attr-dict 
        `{` `offset` `=` $offset `,`  `align` `=` $align `}`
    }];
}

def Wasm_StoreOp : Wasm_Op<"i32.store", []> {
    let summary = "Memory store operation";
    let description = [{
        The `memory.store` operation writes to memory.
    }];
    let arguments = (ins I32Attr:$offset, I32Attr:$align);
    let assemblyFormat = [{ 
        attr-dict 
        `{` `offset` `=` $offset `,`  `align` `=` $align `}`
    }];
}

def Wasm_MemorySizeOp : Wasm_Op<"memory.size", []> {
    let summary = "Memory size operation";
    let description = [{
        The `memory.size` operation returns the size of memory.
    }];
    let assemblyFormat = [{ attr-dict }];
}

def Wasm_MemoryGrowOp : Wasm_Op<"memory.grow", []> {
    let summary = "Memory grow operation";
    let description = [{
        The `memory.grow` operation grows memory.
    }];
    let assemblyFormat = [{ attr-dict }];
}

def Wasm_MemoryFillOp : Wasm_Op<"memory.fill", []> {
    let summary = "Memory fill operation";
    let description = [{
        The `memory.fill` operation fills a range of memory.
    }];
    let assemblyFormat = [{ attr-dict }];
}

def Wasm_MemoryCopyOp : Wasm_Op<"memory.copy", []> {
    let summary = "Memory copy operation";
    let description = [{
        The `memory.copy` operation copies a range of memory.
    }];
    let assemblyFormat = [{ attr-dict }];
}

def Wasm_MemoryInitOp : Wasm_Op<"memory.init", []> {
    let summary = "Memory init operation";
    let description = [{
        The `memory.init` operation initializes a range of memory.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_DataDropOp : Wasm_Op<"data.drop", []> {
    let summary = "Data drop operation";
    let description = [{
        The `data.drop` operation drops a range of data.
    }];
    let arguments = (ins I32Attr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

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
