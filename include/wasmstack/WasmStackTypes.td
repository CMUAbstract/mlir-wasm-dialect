//===- WasmStackTypes.td - WasmStack dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_TYPES
#define WASMSTACK_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "wasmstack/WasmStackDialect.td"

//===----------------------------------------------------------------------===//
// WasmStack Type Base Class
//===----------------------------------------------------------------------===//

/// Base type for WasmStack types.
class WasmStack_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WasmStack_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Value Type Constraints (reuse MLIR builtins)
//===----------------------------------------------------------------------===//

def WasmStack_I32 : Type<CPred<"$_self.isSignlessInteger(32)">, "i32">;
def WasmStack_I64 : Type<CPred<"$_self.isSignlessInteger(64)">, "i64">;
def WasmStack_F32 : Type<CPred<"$_self.isa<::mlir::Float32Type>()">, "f32">;
def WasmStack_F64 : Type<CPred<"$_self.isa<::mlir::Float64Type>()">, "f64">;

def WasmStack_ValueType : AnyTypeOf<[WasmStack_I32, WasmStack_I64,
                                     WasmStack_F32, WasmStack_F64],
                                    "wasm value type">;

def WasmStack_IntegerType : AnyTypeOf<[WasmStack_I32, WasmStack_I64],
                                      "wasm integer type">;

def WasmStack_FloatType : AnyTypeOf<[WasmStack_F32, WasmStack_F64],
                                    "wasm float type">;

//===----------------------------------------------------------------------===//
// Reference Types (custom types for Wasm-specific semantics)
//===----------------------------------------------------------------------===//

def WasmStack_FuncRefType : WasmStack_Type<"FuncRef", "funcref"> {
    let summary = "WebAssembly function reference type";
    let description = [{
        Represents a reference to a WebAssembly function. Used with
        ref.func, call_indirect, and table operations.
    }];
    let parameters = (ins OptionalParameter<"::mlir::FlatSymbolRefAttr">:$type_name);
    let assemblyFormat = "(`<` $type_name^ `>`)?";
}

def WasmStack_ContRefType : WasmStack_Type<"ContRef", "contref"> {
    let summary = "WebAssembly continuation reference type";
    let description = [{
        Represents a reference to a continuation (stack switching proposal).
        Used with cont.new, resume, suspend, and switch operations.
    }];
    let parameters = (ins "::mlir::FlatSymbolRefAttr":$type_name);
    let assemblyFormat = "`<` $type_name `>`";
}

def WasmStack_ExternRefType : WasmStack_Type<"ExternRef", "externref"> {
    let summary = "WebAssembly external reference type";
    let description = [{
        Represents an opaque reference to an external object.
    }];
}

//===----------------------------------------------------------------------===//
// Aggregate Type Constraints
//===----------------------------------------------------------------------===//

def WasmStack_RefType : AnyTypeOf<[WasmStack_FuncRefType,
                                   WasmStack_ContRefType,
                                   WasmStack_ExternRefType],
                                  "wasm reference type">;

def WasmStack_AnyType : AnyTypeOf<[WasmStack_ValueType, WasmStack_RefType],
                                  "any wasm type">;

#endif // WASMSTACK_TYPES
