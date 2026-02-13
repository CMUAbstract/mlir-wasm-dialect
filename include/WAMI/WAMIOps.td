//===- WAMIOps.td - WAMI dialect operations -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_OPS
#define WAMI_OPS

include "WAMI/WAMITypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class WAMI_Op<string mnemonic, list<Trait> traits = []>
    : Op<WAMI_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Core WebAssembly Operations
//===----------------------------------------------------------------------===//

def WAMI_SelectOp : WAMI_Op<"select", [
    AllTypesMatch<["true_value", "false_value", "result"]>,
    Pure
]> {
  let summary = "Select between two values based on condition";
  let description = [{
    WebAssembly select instruction. Returns true_value if condition is non-zero,
    false_value otherwise.

    This corresponds to the WebAssembly `select` instruction which provides
    a ternary conditional without control flow.
  }];
  let arguments = (ins
    AnyType:$true_value,
    AnyType:$false_value,
    I32:$condition
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def WAMI_LoadOp : WAMI_Op<"load", [Pure]> {
  let summary = "Load value from linear memory";
  let description = [{
    Load a value from WebAssembly linear memory at the given address.
    The address is a 32-bit integer (i32) representing a byte offset
    into linear memory.

    Example:
    ```mlir
    %value = wami.load %addr : i32
    %float = wami.load %addr : f32
    ```
  }];
  let arguments = (ins I32:$address);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$address attr-dict `:` type($result)";
}

def WAMI_StoreOp : WAMI_Op<"store", []> {
  let summary = "Store value to linear memory";
  let description = [{
    Store a value to WebAssembly linear memory at the given address.
    The address is a 32-bit integer (i32) representing a byte offset
    into linear memory.

    Example:
    ```mlir
    wami.store %addr, %value : i32
    wami.store %addr, %float : f32
    ```
  }];
  let arguments = (ins I32:$address, AnyType:$value);
  let assemblyFormat = "$address `,` $value attr-dict `:` type($value)";
}

def WAMI_DataOp : WAMI_Op<"data", [Symbol]> {
  let summary = "Declare initialized data section";
  let description = [{
    Declares a data section with initial values at a specific offset
    in linear memory. This corresponds to WebAssembly's data segment.

    Example:
    ```mlir
    wami.data @arr_data = dense<[1, 2, 3, 4]> : tensor<4xi32> at 1024
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ElementsAttr:$value,
    I32Attr:$offset
  );
  let assemblyFormat = "$sym_name `=` $value `at` $offset attr-dict";
}

//===----------------------------------------------------------------------===//
// Type Conversion Operations
//===----------------------------------------------------------------------===//

def WAMI_TruncSOp : WAMI_Op<"trunc_s", [Pure]> {
  let summary = "Truncate float to signed integer";
  let description = [{
    Truncate a floating-point value to a signed integer.
    This corresponds to WebAssembly instructions like `i32.trunc_f32_s`,
    `i32.trunc_f64_s`, `i64.trunc_f32_s`, `i64.trunc_f64_s`.

    The truncation is towards zero. If the value is NaN or out of range,
    the behavior is a trap in standard WebAssembly.

    Example:
    ```mlir
    %i32_val = wami.trunc_s %f32_val : f32 to i32
    %i64_val = wami.trunc_s %f64_val : f64 to i64
    ```
  }];
  let arguments = (ins AnyFloat:$input);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

def WAMI_TruncUOp : WAMI_Op<"trunc_u", [Pure]> {
  let summary = "Truncate float to unsigned integer";
  let description = [{
    Truncate a floating-point value to an unsigned integer.
    This corresponds to WebAssembly instructions like `i32.trunc_f32_u`,
    `i32.trunc_f64_u`, `i64.trunc_f32_u`, `i64.trunc_f64_u`.

    The truncation is towards zero. If the value is negative, NaN, or out
    of range, the behavior is a trap in standard WebAssembly.

    Example:
    ```mlir
    %i32_val = wami.trunc_u %f32_val : f32 to i32
    %i64_val = wami.trunc_u %f64_val : f64 to i64
    ```
  }];
  let arguments = (ins AnyFloat:$input);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

//===----------------------------------------------------------------------===//
// Stack Switching Operations (WebAssembly Stack Switching Proposal)
//===----------------------------------------------------------------------===//

def WAMI_TypeFuncOp : WAMI_Op<"type.func", [Symbol]> {
  let summary = "Declare a function type";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$type
  );
  let assemblyFormat = "$sym_name `=` $type attr-dict";
}

def WAMI_TypeContOp : WAMI_Op<"type.cont", [Symbol]> {
  let summary = "Declare a continuation type";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$func_type
  );
  let assemblyFormat = "$sym_name `=` `cont` $func_type attr-dict";
  let hasVerifier = 1;
}

def WAMI_TagOp : WAMI_Op<"tag", [Symbol]> {
  let summary = "Declare a suspension tag";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$type,
    OptionalAttr<StrAttr>:$export_name
  );
  let assemblyFormat = "$sym_name `:` $type (`export` $export_name^)? attr-dict";
}

def WAMI_RefFuncOp : WAMI_Op<"ref.func", [Pure]> {
  let summary = "Create function reference";
  let arguments = (ins FlatSymbolRefAttr:$func);
  let results = (outs WAMI_FuncRefType:$result);
  let assemblyFormat = "$func attr-dict `:` type($result)";
}

def WAMI_RefNullOp : WAMI_Op<"ref.null", [Pure]> {
  let summary = "Create null reference";
  let results = (outs WAMI_RefType:$result);
  let assemblyFormat = "`:` type($result) attr-dict";
  let hasVerifier = 1;
}

def WAMI_ContNewOp : WAMI_Op<"cont.new"> {
  let summary = "Create continuation from function reference";
  let arguments = (ins
    WAMI_FuncRefType:$func,
    FlatSymbolRefAttr:$cont_type
  );
  let results = (outs WAMI_ContType:$result);
  let assemblyFormat = "$func `:` type($func) `as` $cont_type attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def WAMI_ContBindOp : WAMI_Op<"cont.bind"> {
  let summary = "Bind arguments to continuation";
  let arguments = (ins
    WAMI_ContType:$cont,
    Variadic<AnyType>:$bound_args,
    FlatSymbolRefAttr:$src_cont_type,
    FlatSymbolRefAttr:$dst_cont_type
  );
  let results = (outs WAMI_ContType:$result);
  let hasVerifier = 1;
}

def WAMI_SuspendOp : WAMI_Op<"suspend"> {
  let summary = "Suspend current continuation";
  let arguments = (ins
    FlatSymbolRefAttr:$tag,
    Variadic<AnyType>:$payload
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
}

def WAMI_HandlerYieldOp : WAMI_Op<"handler.yield", [Terminator]> {
  let summary = "Yield values from resume handler";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "$values attr-dict `:` type($values)";
}

def WAMI_ResumeOp : WAMI_Op<"resume"> {
  let summary = "Resume continuation with typed handlers";
  let arguments = (ins
    WAMI_ContType:$cont,
    Variadic<AnyType>:$args,
    FlatSymbolRefAttr:$cont_type,
    ArrayAttr:$handler_tags
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$handlers);
  let hasVerifier = 1;
}

def WAMI_ResumeThrowOp : WAMI_Op<"resume_throw"> {
  let summary = "Resume continuation and throw on suspension";
  let arguments = (ins
    WAMI_ContType:$cont,
    Variadic<AnyType>:$args,
    FlatSymbolRefAttr:$cont_type,
    ArrayAttr:$handler_tags
  );
  let regions = (region AnyRegion:$handlers);
  let hasVerifier = 1;
}

def WAMI_BarrierYieldOp : WAMI_Op<"barrier.yield", [Terminator]> {
  let summary = "Yield values from barrier";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "$values attr-dict `:` type($values)";
}

def WAMI_BarrierOp : WAMI_Op<"barrier", [SingleBlock, NoTerminator]> {
  let summary = "Barrier for stack switching boundary";
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

#endif // WAMI_OPS
