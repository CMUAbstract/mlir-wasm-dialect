//===- WasmStackOps.td - WasmStack dialect ops -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_OPS
#define WASMSTACK_OPS

include "wasmstack/WasmStackTypes.td"
include "wasmstack/WasmStackInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

// Base class for all WasmStack ops - all are side-effectful (no Pure trait)
// to prevent unwanted transformations by standard MLIR passes.
class WasmStack_Op<string mnemonic, list<Trait> traits = []>
    : Op<WasmStack_Dialect, mnemonic,
         !listconcat([DeclareOpInterfaceMethods<StackEffectOpInterface>], traits)>;

// Base class for binary arithmetic operations (2 inputs, 1 output)
class WasmStack_BinaryOp<string mnemonic, list<Trait> traits = []>
    : WasmStack_Op<mnemonic, traits> {
  let arguments = (ins TypeAttr:$type);
  let assemblyFormat = "`:` $type attr-dict";
  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 2; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType(), getType()};
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getType()};
    }
  }];
}

// Base class for unary operations (1 input, 1 output)
class WasmStack_UnaryOp<string mnemonic, list<Trait> traits = []>
    : WasmStack_Op<mnemonic, traits> {
  let arguments = (ins TypeAttr:$type);
  let assemblyFormat = "`:` $type attr-dict";
  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType()};
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getType()};
    }
  }];
}

// Base class for comparison operations (2 inputs, 1 i32 output)
class WasmStack_CompareOp<string mnemonic, list<Trait> traits = []>
    : WasmStack_Op<mnemonic, traits> {
  let arguments = (ins TypeAttr:$type);
  let assemblyFormat = "`:` $type attr-dict";
  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 2; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType(), getType()};
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32)};
    }
  }];
}

// Base class for test operations (1 input, 1 i32 output)
class WasmStack_TestOp<string mnemonic, list<Trait> traits = []>
    : WasmStack_Op<mnemonic, traits> {
  let arguments = (ins TypeAttr:$type);
  let assemblyFormat = "`:` $type attr-dict";
  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType()};
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32)};
    }
  }];
}

//===----------------------------------------------------------------------===//
// Module Structure Operations
//===----------------------------------------------------------------------===//

def WasmStack_ModuleOp : WasmStack_Op<"module", [
    IsolatedFromAbove,
    SymbolTable,
    SingleBlock,
    NoTerminator
]> {
  let summary = "WebAssembly module";
  let description = [{
    Container for all WebAssembly definitions (functions, memories, globals, etc.)
  }];

  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $body";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_FuncOp : WasmStack_Op<"func", [
    IsolatedFromAbove,
    Symbol,
    SingleBlock,
    NoTerminator
]> {
  let summary = "WebAssembly function";
  let description = [{
    Defines a WebAssembly function with explicit parameter and result types.
    The function body uses implicit stack operations.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$func_type,
    OptionalAttr<StrAttr>:$export_name
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }

    // Helper methods
    ::mlir::TypeRange getArgumentTypes() {
      return getFuncType().getInputs();
    }
    ::mlir::TypeRange getResultTypes() {
      return getFuncType().getResults();
    }
    unsigned getNumArguments() {
      return getFuncType().getNumInputs();
    }
    unsigned getNumResults() {
      return getFuncType().getNumResults();
    }
  }];

  let hasVerifier = 1;
}

def WasmStack_MemoryOp : WasmStack_Op<"memory", [Symbol]> {
  let summary = "WebAssembly linear memory declaration";
  let description = [{
    Declares a WebAssembly linear memory with minimum and optional maximum size
    in pages (64KiB each).
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I32Attr:$min_pages,
    OptionalAttr<I32Attr>:$max_pages,
    OptionalAttr<StrAttr>:$export_name
  );

  let assemblyFormat = "$sym_name `min` `=` $min_pages (`max` `=` $max_pages^)? (`export` $export_name^)? attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_GlobalOp : WasmStack_Op<"global", [
    Symbol,
    SingleBlock,
    NoTerminator
]> {
  let summary = "WebAssembly global variable";
  let description = [{
    Declares a WebAssembly global variable with type, mutability, and
    optional initializer. Type must be a valid WebAssembly type
    (i32, i64, f32, f64, or reference type).
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    OptionalAttr<StrAttr>:$export_name
  );
  let regions = (region SizedRegion<1>:$init);

  let assemblyFormat = "$sym_name `:` $type (`mutable` $is_mutable^)? (`export` $export_name^)? attr-dict-with-keyword $init";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_DataOp : WasmStack_Op<"data"> {
  let summary = "WebAssembly data segment";
  let description = [{
    Declares a data segment to be placed in linear memory at a specified offset.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$memory,
    I32Attr:$offset,
    StrAttr:$data
  );

  let assemblyFormat = "$memory `offset` `=` $offset $data attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

//===----------------------------------------------------------------------===//
// Local Variable Operations
//===----------------------------------------------------------------------===//

def WasmStack_LocalOp : WasmStack_Op<"local"> {
  let summary = "Local variable declaration";
  let description = [{
    Declares a local variable within a function. Local indices are assigned
    after function parameters.
  }];

  let arguments = (ins
    I32Attr:$index,
    TypeAttr:$type
  );

  let assemblyFormat = "$index `:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_LocalGetOp : WasmStack_Op<"local.get"> {
  let summary = "Get local variable value";
  let description = [{
    Pushes the value of the local variable at the given index onto the stack.
  }];

  let arguments = (ins I32Attr:$index, TypeAttr:$type);
  let assemblyFormat = "$index `:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getType()};
    }
  }];
}

def WasmStack_LocalSetOp : WasmStack_Op<"local.set"> {
  let summary = "Set local variable value";
  let description = [{
    Pops a value from the stack and stores it in the local variable at the
    given index.
  }];

  let arguments = (ins I32Attr:$index, TypeAttr:$type);
  let assemblyFormat = "$index `:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }
    unsigned getStackOutputs() { return 0; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType()};
    }
  }];
}

def WasmStack_LocalTeeOp : WasmStack_Op<"local.tee"> {
  let summary = "Set local and return value";
  let description = [{
    Pops a value, stores it in the local variable, and pushes it back.
    Equivalent to local.set + local.get but more efficient.
  }];

  let arguments = (ins I32Attr:$index, TypeAttr:$type);
  let assemblyFormat = "$index `:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType()};
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getType()};
    }
  }];
}

def WasmStack_GlobalGetOp : WasmStack_Op<"global.get"> {
  let summary = "Get global variable value";
  let arguments = (ins FlatSymbolRefAttr:$global, TypeAttr:$type);
  let assemblyFormat = "$global `:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getType()};
    }
  }];
}

def WasmStack_GlobalSetOp : WasmStack_Op<"global.set"> {
  let summary = "Set global variable value";
  let arguments = (ins FlatSymbolRefAttr:$global, TypeAttr:$type);
  let assemblyFormat = "$global `:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }
    unsigned getStackOutputs() { return 0; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType()};
    }
  }];
}

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

def WasmStack_I32ConstOp : WasmStack_Op<"i32.const"> {
  let summary = "Push i32 constant";
  let arguments = (ins I32Attr:$value);
  let assemblyFormat = "$value attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32)};
    }
  }];
}

def WasmStack_I64ConstOp : WasmStack_Op<"i64.const"> {
  let summary = "Push i64 constant";
  let arguments = (ins I64Attr:$value);
  let assemblyFormat = "$value attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {::mlir::IntegerType::get(getContext(), 64)};
    }
  }];
}

def WasmStack_F32ConstOp : WasmStack_Op<"f32.const"> {
  let summary = "Push f32 constant";
  let arguments = (ins F32Attr:$value);
  let assemblyFormat = "$value attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {::mlir::Float32Type::get(getContext())};
    }
  }];
}

def WasmStack_F64ConstOp : WasmStack_Op<"f64.const"> {
  let summary = "Push f64 constant";
  let arguments = (ins F64Attr:$value);
  let assemblyFormat = "$value attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {::mlir::Float64Type::get(getContext())};
    }
  }];
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

// Integer arithmetic
def WasmStack_AddOp : WasmStack_BinaryOp<"add"> {
  let summary = "Add two values";
}

def WasmStack_SubOp : WasmStack_BinaryOp<"sub"> {
  let summary = "Subtract two values";
}

def WasmStack_MulOp : WasmStack_BinaryOp<"mul"> {
  let summary = "Multiply two values";
}

def WasmStack_DivSOp : WasmStack_BinaryOp<"div_s"> {
  let summary = "Signed division";
}

def WasmStack_DivUOp : WasmStack_BinaryOp<"div_u"> {
  let summary = "Unsigned division";
}

def WasmStack_RemSOp : WasmStack_BinaryOp<"rem_s"> {
  let summary = "Signed remainder";
}

def WasmStack_RemUOp : WasmStack_BinaryOp<"rem_u"> {
  let summary = "Unsigned remainder";
}

// Bitwise operations
def WasmStack_AndOp : WasmStack_BinaryOp<"and"> {
  let summary = "Bitwise AND";
}

def WasmStack_OrOp : WasmStack_BinaryOp<"or"> {
  let summary = "Bitwise OR";
}

def WasmStack_XorOp : WasmStack_BinaryOp<"xor"> {
  let summary = "Bitwise XOR";
}

def WasmStack_ShlOp : WasmStack_BinaryOp<"shl"> {
  let summary = "Shift left";
}

def WasmStack_ShrSOp : WasmStack_BinaryOp<"shr_s"> {
  let summary = "Signed shift right";
}

def WasmStack_ShrUOp : WasmStack_BinaryOp<"shr_u"> {
  let summary = "Unsigned shift right";
}

def WasmStack_RotlOp : WasmStack_BinaryOp<"rotl"> {
  let summary = "Rotate left";
}

def WasmStack_RotrOp : WasmStack_BinaryOp<"rotr"> {
  let summary = "Rotate right";
}

// Unary integer operations
def WasmStack_ClzOp : WasmStack_UnaryOp<"clz"> {
  let summary = "Count leading zeros";
}

def WasmStack_CtzOp : WasmStack_UnaryOp<"ctz"> {
  let summary = "Count trailing zeros";
}

def WasmStack_PopcntOp : WasmStack_UnaryOp<"popcnt"> {
  let summary = "Population count (number of 1 bits)";
}

// Float arithmetic
def WasmStack_FAbsOp : WasmStack_UnaryOp<"abs"> {
  let summary = "Absolute value (float)";
}

def WasmStack_FNegOp : WasmStack_UnaryOp<"neg"> {
  let summary = "Negate (float)";
}

def WasmStack_FCeilOp : WasmStack_UnaryOp<"ceil"> {
  let summary = "Ceiling (float)";
}

def WasmStack_FFloorOp : WasmStack_UnaryOp<"floor"> {
  let summary = "Floor (float)";
}

def WasmStack_FTruncOp : WasmStack_UnaryOp<"trunc"> {
  let summary = "Truncate towards zero (float)";
}

def WasmStack_FNearestOp : WasmStack_UnaryOp<"nearest"> {
  let summary = "Round to nearest even (float)";
}

def WasmStack_FSqrtOp : WasmStack_UnaryOp<"sqrt"> {
  let summary = "Square root (float)";
}

def WasmStack_FDivOp : WasmStack_BinaryOp<"div"> {
  let summary = "Float division";
}

def WasmStack_FMinOp : WasmStack_BinaryOp<"min"> {
  let summary = "Minimum (float)";
}

def WasmStack_FMaxOp : WasmStack_BinaryOp<"max"> {
  let summary = "Maximum (float)";
}

def WasmStack_FCopysignOp : WasmStack_BinaryOp<"copysign"> {
  let summary = "Copy sign (float)";
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def WasmStack_EqzOp : WasmStack_TestOp<"eqz"> {
  let summary = "Equal to zero test";
}

def WasmStack_EqOp : WasmStack_CompareOp<"eq"> {
  let summary = "Equal comparison";
}

def WasmStack_NeOp : WasmStack_CompareOp<"ne"> {
  let summary = "Not equal comparison";
}

def WasmStack_LtSOp : WasmStack_CompareOp<"lt_s"> {
  let summary = "Signed less than";
}

def WasmStack_LtUOp : WasmStack_CompareOp<"lt_u"> {
  let summary = "Unsigned less than";
}

def WasmStack_GtSOp : WasmStack_CompareOp<"gt_s"> {
  let summary = "Signed greater than";
}

def WasmStack_GtUOp : WasmStack_CompareOp<"gt_u"> {
  let summary = "Unsigned greater than";
}

def WasmStack_LeSOp : WasmStack_CompareOp<"le_s"> {
  let summary = "Signed less than or equal";
}

def WasmStack_LeUOp : WasmStack_CompareOp<"le_u"> {
  let summary = "Unsigned less than or equal";
}

def WasmStack_GeSOp : WasmStack_CompareOp<"ge_s"> {
  let summary = "Signed greater than or equal";
}

def WasmStack_GeUOp : WasmStack_CompareOp<"ge_u"> {
  let summary = "Unsigned greater than or equal";
}

// Float comparisons (same structure but for floats)
// WebAssembly uses lt, gt, le, ge for floats (no signed/unsigned distinction)
def WasmStack_FLtOp : WasmStack_CompareOp<"lt"> {
  let summary = "Float less than";
}

def WasmStack_FGtOp : WasmStack_CompareOp<"gt"> {
  let summary = "Float greater than";
}

def WasmStack_FLeOp : WasmStack_CompareOp<"le"> {
  let summary = "Float less than or equal";
}

def WasmStack_FGeOp : WasmStack_CompareOp<"ge"> {
  let summary = "Float greater than or equal";
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def WasmStack_BlockOp : WasmStack_Op<"block", [
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<WasmLabelOpInterface>
]> {
  let summary = "Structured block";
  let description = [{
    A structured control flow block. Branch targets exit to the end of the block.
    Uses symbolic labels for readable IR; the emitter computes depth indices.

    Supports WebAssembly multi-value: blocks can have input parameters (consumed
    from stack when entering) and result types (left on stack when exiting).
  }];

  let arguments = (ins
    SymbolNameAttr:$label,
    TypeArrayAttr:$param_types,
    TypeArrayAttr:$result_types
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$label `:` `(` $param_types `)` `->` $result_types attr-dict-with-keyword $body";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return getParamTypes().size(); }
    unsigned getStackOutputs() { return getResultTypes().size(); }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      ::llvm::SmallVector<::mlir::Type> types;
      for (auto attr : getParamTypes()) {
        types.push_back(::mlir::cast<::mlir::TypeAttr>(attr).getValue());
      }
      return types;
    }
  }];
}

def WasmStack_LoopOp : WasmStack_Op<"loop", [
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<WasmLabelOpInterface, ["isLoop"]>
]> {
  let summary = "Structured loop";
  let description = [{
    A structured loop block. Branch targets jump to the start of the loop.
    Unlike block, branching to a loop re-enters from the beginning.

    Supports WebAssembly multi-value: loops can have input parameters (consumed
    from stack when entering) and result types (left on stack when exiting).
    Note: When branching to a loop, the branch provides values matching param_types.
  }];

  let arguments = (ins
    SymbolNameAttr:$label,
    TypeArrayAttr:$param_types,
    TypeArrayAttr:$result_types
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$label `:` `(` $param_types `)` `->` $result_types attr-dict-with-keyword $body";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return getParamTypes().size(); }
    unsigned getStackOutputs() { return getResultTypes().size(); }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      ::llvm::SmallVector<::mlir::Type> types;
      for (auto attr : getParamTypes()) {
        types.push_back(::mlir::cast<::mlir::TypeAttr>(attr).getValue());
      }
      return types;
    }
  }];
}

def WasmStack_IfOp : WasmStack_Op<"if", [
    SingleBlock,
    NoTerminator
]> {
  let summary = "Conditional block";
  let description = [{
    Conditional execution. Pops an i32 condition; if non-zero executes then_body,
    otherwise executes else_body (if present).

    Supports WebAssembly multi-value: in addition to the condition, the if can
    consume additional parameters from the stack (param_types) and produce
    results (result_types). The condition is always consumed first (implicit i32).
  }];

  let arguments = (ins
    TypeArrayAttr:$param_types,
    TypeArrayAttr:$result_types
  );
  let regions = (region SizedRegion<1>:$then_body, AnyRegion:$else_body);

  let assemblyFormat = "`:` `(` $param_types `)` `->` $result_types attr-dict-with-keyword `then` $then_body (`else` $else_body^)?";

  let extraClassDeclaration = [{
    // +1 for the implicit i32 condition
    unsigned getStackInputs() { return 1 + getParamTypes().size(); }
    unsigned getStackOutputs() { return getResultTypes().size(); }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      ::llvm::SmallVector<::mlir::Type> types;
      // First input is always the i32 condition
      types.push_back(::mlir::IntegerType::get(getContext(), 32));
      // Then additional parameters
      for (auto attr : getParamTypes()) {
        types.push_back(::mlir::cast<::mlir::TypeAttr>(attr).getValue());
      }
      return types;
    }
  }];
}

def WasmStack_BrOp : WasmStack_Op<"br"> {
  let summary = "Unconditional branch";
  let description = [{
    Unconditional branch to a block/loop label. NOT a terminator in MLIR sense
    since wasmstack uses NoTerminator regions.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target);
  let assemblyFormat = "$target attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_BrIfOp : WasmStack_Op<"br_if"> {
  let summary = "Conditional branch";
  let description = [{
    Conditional branch. Pops an i32; if non-zero, branches to target.
    NOT a terminator - execution can fall through if condition is false.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target);
  let assemblyFormat = "$target attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }  // condition
    unsigned getStackOutputs() { return 0; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32)};
    }
  }];
}

def WasmStack_BrTableOp : WasmStack_Op<"br_table"> {
  let summary = "Branch table";
  let description = [{
    Branch table for switch-like control flow. Pops an i32 index and branches
    to targets[index] or default_target if index is out of bounds.
  }];

  let arguments = (ins
    ArrayAttr:$targets,
    FlatSymbolRefAttr:$default_target
  );
  let assemblyFormat = "$targets `default` $default_target attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }  // index
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_ReturnOp : WasmStack_Op<"return"> {
  let summary = "Return from function";
  let description = [{
    Returns from the current function. Stack must have the function's
    return type values on top. The number and types of values are validated
    against the enclosing function's signature.
  }];

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }  // Determined by function signature
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_UnreachableOp : WasmStack_Op<"unreachable"> {
  let summary = "Unreachable trap";
  let description = [{
    Trap immediately. Used to signal unreachable code.
  }];

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_NopOp : WasmStack_Op<"nop"> {
  let summary = "No operation";
  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

//===----------------------------------------------------------------------===//
// Call Operations
//===----------------------------------------------------------------------===//

def WasmStack_CallOp : WasmStack_Op<"call"> {
  let summary = "Direct function call";
  let description = [{
    Calls a function by reference. Arguments are popped from stack in reverse
    order, results are pushed to stack.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    TypeAttrOf<FunctionType>:$callee_type
  );

  let assemblyFormat = "$callee `:` $callee_type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() {
      return getCalleeType().getNumInputs();
    }
    unsigned getStackOutputs() {
      return getCalleeType().getNumResults();
    }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return llvm::to_vector(getCalleeType().getInputs());
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return llvm::to_vector(getCalleeType().getResults());
    }
  }];
}

def WasmStack_CallIndirectOp : WasmStack_Op<"call_indirect"> {
  let summary = "Indirect function call through table";
  let description = [{
    Calls a function through a table. The table index is popped from stack,
    then arguments are popped, then results are pushed.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$table,
    TypeAttrOf<FunctionType>:$callee_type
  );

  let assemblyFormat = "$table `:` $callee_type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() {
      return getCalleeType().getNumInputs() + 1;  // +1 for table index
    }
    unsigned getStackOutputs() {
      return getCalleeType().getNumResults();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

// Load operations pop address, push value
class WasmStack_LoadOp<string mnemonic> : WasmStack_Op<mnemonic> {
  let arguments = (ins
    I32Attr:$offset,
    I32Attr:$align,
    TypeAttr:$type
  );
  let assemblyFormat = "`offset` `=` $offset `align` `=` $align `:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }  // address
    unsigned getStackOutputs() { return 1; }  // value
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32)};  // i32 address
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getType()};
    }
  }];
}

// Store operations pop value and address, push nothing
class WasmStack_StoreOp<string mnemonic> : WasmStack_Op<mnemonic> {
  let arguments = (ins
    I32Attr:$offset,
    I32Attr:$align,
    TypeAttr:$type
  );
  let assemblyFormat = "`offset` `=` $offset `align` `=` $align `:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 2; }  // address, value
    unsigned getStackOutputs() { return 0; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32), getType()};
    }
  }];
}

def WasmStack_I32LoadOp : WasmStack_LoadOp<"i32.load"> {
  let summary = "Load i32 from memory";
}

def WasmStack_I64LoadOp : WasmStack_LoadOp<"i64.load"> {
  let summary = "Load i64 from memory";
}

def WasmStack_F32LoadOp : WasmStack_LoadOp<"f32.load"> {
  let summary = "Load f32 from memory";
}

def WasmStack_F64LoadOp : WasmStack_LoadOp<"f64.load"> {
  let summary = "Load f64 from memory";
}

def WasmStack_I32Load8SOp : WasmStack_LoadOp<"i32.load8_s"> {
  let summary = "Load i8 and sign-extend to i32";
}

def WasmStack_I32Load8UOp : WasmStack_LoadOp<"i32.load8_u"> {
  let summary = "Load i8 and zero-extend to i32";
}

def WasmStack_I32Load16SOp : WasmStack_LoadOp<"i32.load16_s"> {
  let summary = "Load i16 and sign-extend to i32";
}

def WasmStack_I32Load16UOp : WasmStack_LoadOp<"i32.load16_u"> {
  let summary = "Load i16 and zero-extend to i32";
}

def WasmStack_I64Load8SOp : WasmStack_LoadOp<"i64.load8_s"> {
  let summary = "Load i8 and sign-extend to i64";
}

def WasmStack_I64Load8UOp : WasmStack_LoadOp<"i64.load8_u"> {
  let summary = "Load i8 and zero-extend to i64";
}

def WasmStack_I64Load16SOp : WasmStack_LoadOp<"i64.load16_s"> {
  let summary = "Load i16 and sign-extend to i64";
}

def WasmStack_I64Load16UOp : WasmStack_LoadOp<"i64.load16_u"> {
  let summary = "Load i16 and zero-extend to i64";
}

def WasmStack_I64Load32SOp : WasmStack_LoadOp<"i64.load32_s"> {
  let summary = "Load i32 and sign-extend to i64";
}

def WasmStack_I64Load32UOp : WasmStack_LoadOp<"i64.load32_u"> {
  let summary = "Load i32 and zero-extend to i64";
}

def WasmStack_I32StoreOp : WasmStack_StoreOp<"i32.store"> {
  let summary = "Store i32 to memory";
}

def WasmStack_I64StoreOp : WasmStack_StoreOp<"i64.store"> {
  let summary = "Store i64 to memory";
}

def WasmStack_F32StoreOp : WasmStack_StoreOp<"f32.store"> {
  let summary = "Store f32 to memory";
}

def WasmStack_F64StoreOp : WasmStack_StoreOp<"f64.store"> {
  let summary = "Store f64 to memory";
}

def WasmStack_I32Store8Op : WasmStack_StoreOp<"i32.store8"> {
  let summary = "Wrap i32 to i8 and store";
}

def WasmStack_I32Store16Op : WasmStack_StoreOp<"i32.store16"> {
  let summary = "Wrap i32 to i16 and store";
}

def WasmStack_I64Store8Op : WasmStack_StoreOp<"i64.store8"> {
  let summary = "Wrap i64 to i8 and store";
}

def WasmStack_I64Store16Op : WasmStack_StoreOp<"i64.store16"> {
  let summary = "Wrap i64 to i16 and store";
}

def WasmStack_I64Store32Op : WasmStack_StoreOp<"i64.store32"> {
  let summary = "Wrap i64 to i32 and store";
}

def WasmStack_MemorySizeOp : WasmStack_Op<"memory.size"> {
  let summary = "Get current memory size in pages";
  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32)};
    }
  }];
}

def WasmStack_MemoryGrowOp : WasmStack_Op<"memory.grow"> {
  let summary = "Grow memory by specified pages";
  let description = [{
    Grows the memory by the specified number of pages (popped from stack).
    Returns the previous size in pages, or -1 on failure.
  }];
  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }  // delta pages
    unsigned getStackOutputs() { return 1; }  // previous size or -1
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32)};
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {::mlir::IntegerType::get(getContext(), 32)};
    }
  }];
}

//===----------------------------------------------------------------------===//
// Stack Manipulation Operations
//===----------------------------------------------------------------------===//

def WasmStack_DropOp : WasmStack_Op<"drop"> {
  let summary = "Drop top value from stack";
  let arguments = (ins TypeAttr:$type);
  let assemblyFormat = "`:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }
    unsigned getStackOutputs() { return 0; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType()};
    }
  }];
}

def WasmStack_SelectOp : WasmStack_Op<"select"> {
  let summary = "Select one of two values based on condition";
  let description = [{
    Pops condition (i32), then two values of the same type.
    If condition is non-zero, pushes the first value; otherwise the second.
  }];

  let arguments = (ins TypeAttr:$type);
  let assemblyFormat = "`:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 3; }  // val1, val2, condition
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getType(), getType(), ::mlir::IntegerType::get(getContext(), 32)};
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getType()};
    }
  }];
}

//===----------------------------------------------------------------------===//
// Conversion Operations
//===----------------------------------------------------------------------===//

class WasmStack_ConversionOp<string mnemonic> : WasmStack_Op<mnemonic> {
  let arguments = (ins TypeAttr:$src_type, TypeAttr:$dst_type);
  let assemblyFormat = "`:` $src_type `->` $dst_type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackInputTypes() {
      return {getSrcType()};
    }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getDstType()};
    }
  }];
}

def WasmStack_I32WrapI64Op : WasmStack_ConversionOp<"i32.wrap_i64"> {
  let summary = "Wrap i64 to i32";
}

def WasmStack_I64ExtendI32SOp : WasmStack_ConversionOp<"i64.extend_i32_s"> {
  let summary = "Sign-extend i32 to i64";
}

def WasmStack_I64ExtendI32UOp : WasmStack_ConversionOp<"i64.extend_i32_u"> {
  let summary = "Zero-extend i32 to i64";
}

def WasmStack_I32TruncF32SOp : WasmStack_ConversionOp<"i32.trunc_f32_s"> {
  let summary = "Truncate f32 to signed i32";
}

def WasmStack_I32TruncF32UOp : WasmStack_ConversionOp<"i32.trunc_f32_u"> {
  let summary = "Truncate f32 to unsigned i32";
}

def WasmStack_I32TruncF64SOp : WasmStack_ConversionOp<"i32.trunc_f64_s"> {
  let summary = "Truncate f64 to signed i32";
}

def WasmStack_I32TruncF64UOp : WasmStack_ConversionOp<"i32.trunc_f64_u"> {
  let summary = "Truncate f64 to unsigned i32";
}

def WasmStack_I64TruncF32SOp : WasmStack_ConversionOp<"i64.trunc_f32_s"> {
  let summary = "Truncate f32 to signed i64";
}

def WasmStack_I64TruncF32UOp : WasmStack_ConversionOp<"i64.trunc_f32_u"> {
  let summary = "Truncate f32 to unsigned i64";
}

def WasmStack_I64TruncF64SOp : WasmStack_ConversionOp<"i64.trunc_f64_s"> {
  let summary = "Truncate f64 to signed i64";
}

def WasmStack_I64TruncF64UOp : WasmStack_ConversionOp<"i64.trunc_f64_u"> {
  let summary = "Truncate f64 to unsigned i64";
}

def WasmStack_F32ConvertI32SOp : WasmStack_ConversionOp<"f32.convert_i32_s"> {
  let summary = "Convert signed i32 to f32";
}

def WasmStack_F32ConvertI32UOp : WasmStack_ConversionOp<"f32.convert_i32_u"> {
  let summary = "Convert unsigned i32 to f32";
}

def WasmStack_F32ConvertI64SOp : WasmStack_ConversionOp<"f32.convert_i64_s"> {
  let summary = "Convert signed i64 to f32";
}

def WasmStack_F32ConvertI64UOp : WasmStack_ConversionOp<"f32.convert_i64_u"> {
  let summary = "Convert unsigned i64 to f32";
}

def WasmStack_F64ConvertI32SOp : WasmStack_ConversionOp<"f64.convert_i32_s"> {
  let summary = "Convert signed i32 to f64";
}

def WasmStack_F64ConvertI32UOp : WasmStack_ConversionOp<"f64.convert_i32_u"> {
  let summary = "Convert unsigned i32 to f64";
}

def WasmStack_F64ConvertI64SOp : WasmStack_ConversionOp<"f64.convert_i64_s"> {
  let summary = "Convert signed i64 to f64";
}

def WasmStack_F64ConvertI64UOp : WasmStack_ConversionOp<"f64.convert_i64_u"> {
  let summary = "Convert unsigned i64 to f64";
}

def WasmStack_F32DemoteF64Op : WasmStack_ConversionOp<"f32.demote_f64"> {
  let summary = "Demote f64 to f32";
}

def WasmStack_F64PromoteF32Op : WasmStack_ConversionOp<"f64.promote_f32"> {
  let summary = "Promote f32 to f64";
}

def WasmStack_I32ReinterpretF32Op : WasmStack_ConversionOp<"i32.reinterpret_f32"> {
  let summary = "Reinterpret f32 bits as i32";
}

def WasmStack_I64ReinterpretF64Op : WasmStack_ConversionOp<"i64.reinterpret_f64"> {
  let summary = "Reinterpret f64 bits as i64";
}

def WasmStack_F32ReinterpretI32Op : WasmStack_ConversionOp<"f32.reinterpret_i32"> {
  let summary = "Reinterpret i32 bits as f32";
}

def WasmStack_F64ReinterpretI64Op : WasmStack_ConversionOp<"f64.reinterpret_i64"> {
  let summary = "Reinterpret i64 bits as f64";
}

//===----------------------------------------------------------------------===//
// Stack Switching Operations (Typed Continuations Proposal)
//===----------------------------------------------------------------------===//

def WasmStack_TypeFuncOp : WasmStack_Op<"type.func", [Symbol]> {
  let summary = "Declare a function type";
  let description = [{
    Declares a named function type for use with call_indirect and continuations.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$type
  );

  let assemblyFormat = "$sym_name `=` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_TypeContOp : WasmStack_Op<"type.cont", [Symbol]> {
  let summary = "Declare a continuation type";
  let description = [{
    Declares a named continuation type referencing a function type.
    Used with cont.new, resume, suspend, and switch.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$func_type
  );

  let assemblyFormat = "$sym_name `=` `cont` $func_type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_TagOp : WasmStack_Op<"tag", [Symbol]> {
  let summary = "Declare a tag for exceptions/suspensions";
  let description = [{
    Declares a tag that can be used with suspend and resume handlers.
    Tags have a type signature for the values passed when suspending.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$type,
    OptionalAttr<StrAttr>:$export_name
  );

  let assemblyFormat = "$sym_name `:` $type (`export` $export_name^)? attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_RefFuncOp : WasmStack_Op<"ref.func"> {
  let summary = "Get function reference";
  let description = [{
    Pushes a reference to the named function onto the stack.
  }];

  let arguments = (ins FlatSymbolRefAttr:$func);
  let assemblyFormat = "$func attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
  }];
}

def WasmStack_RefNullOp : WasmStack_Op<"ref.null"> {
  let summary = "Get null reference";
  let arguments = (ins TypeAttr:$type);
  let assemblyFormat = "`:` $type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 1; }
    ::llvm::SmallVector<::mlir::Type> getStackOutputTypes() {
      return {getType()};
    }
  }];
}

def WasmStack_ContNewOp : WasmStack_Op<"cont.new"> {
  let summary = "Create new continuation";
  let description = [{
    Pops a function reference and pushes a new continuation reference.
    The continuation captures the function to be invoked later via resume.
  }];

  let arguments = (ins FlatSymbolRefAttr:$cont_type);
  let assemblyFormat = "$cont_type attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }  // func_ref
    unsigned getStackOutputs() { return 1; }  // cont_ref
  }];
}

def WasmStack_ContBindOp : WasmStack_Op<"cont.bind"> {
  let summary = "Bind arguments to continuation";
  let description = [{
    Binds arguments to a continuation, creating a new continuation with
    fewer expected arguments.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$src_cont_type,
    FlatSymbolRefAttr:$dst_cont_type
  );
  let assemblyFormat = "$src_cont_type `->` $dst_cont_type attr-dict";

  let extraClassDeclaration = [{
    // Stack effects depend on the type difference
    unsigned getStackInputs() { return 1; }  // TODO: + bound args
    unsigned getStackOutputs() { return 1; }
  }];
}

def WasmStack_ResumeOp : WasmStack_Op<"resume"> {
  let summary = "Resume a continuation";
  let description = [{
    Resumes a continuation with the given arguments. Handlers specify how
    to handle suspensions from the continuation.

    Handlers is an array of (tag, label) pairs. When the continuation
    suspends with a tag, control transfers to the corresponding label.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$cont_type,
    ArrayAttr:$handlers
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Stack effects depend on continuation type
    unsigned getStackInputs() { return 1; }  // cont_ref + args
    unsigned getStackOutputs() { return 0; }  // results from cont type
  }];
}

def WasmStack_SuspendOp : WasmStack_Op<"suspend"> {
  let summary = "Suspend current continuation";
  let description = [{
    Suspends the current continuation with the given tag.
    Values on stack matching the tag type are passed to the handler.
  }];

  let arguments = (ins FlatSymbolRefAttr:$tag);
  let assemblyFormat = "$tag attr-dict";

  let extraClassDeclaration = [{
    // Stack effects depend on tag type
    unsigned getStackInputs() { return 0; }
    unsigned getStackOutputs() { return 0; }
  }];
}

def WasmStack_SwitchOp : WasmStack_Op<"switch"> {
  let summary = "Switch to another continuation";
  let description = [{
    Direct switch to another continuation, suspending the current one.
    More efficient than suspend+resume for symmetric coroutines.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$cont_type,
    FlatSymbolRefAttr:$tag
  );
  let assemblyFormat = "$cont_type `(` `tag` `:` $tag `)` attr-dict";

  let extraClassDeclaration = [{
    unsigned getStackInputs() { return 1; }  // cont_ref
    unsigned getStackOutputs() { return 0; }
  }];
}

#endif // WASMSTACK_OPS
