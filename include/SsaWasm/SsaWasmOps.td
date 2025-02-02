
//===- SsaWasmOps.td - SsaWasm dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SSAWASM_OPS
#define SSAWASM_OPS

include "SsaWasm/SsaWasmTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/Constraints.td"

class SsaWasm_Op<string mnemonic, list<Trait> traits = []>
    : Op<SsaWasm_Dialect, mnemonic, traits>;


///////////////////////////////////////////////////////////////////////////
// Base class for numeric binary operations.
// All numeric binary ops in SSA form explicitly list their two operands
// and their single result.
///////////////////////////////////////////////////////////////////////////

class SsaWasm_WasmNumericBinaryOp<string mnemonic, list<Trait> traits = []> 
    : SsaWasm_Op<mnemonic, traits # [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "SSA-based numeric binary operation";
  let description = [{
    This is the base class for numeric binary operations in the SSA Wasm dialect.
    Both operands and the result are explicit, allowing for easier analysis and optimization.
  }];

  // Two explicit operands and one result.
  let arguments = (ins SsaWasm_WasmNumericLike:$lhs, SsaWasm_WasmNumericLike:$rhs);
  let results = (outs SsaWasm_WasmNumericLike:$result);

  // Assembly format shows the two operands followed by the type of the result.
  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($result)";
}

///////////////////////////////////////////////////////////////////////////
// Addition Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_AddOp : SsaWasm_WasmNumericBinaryOp<"add"> {
  let summary = "SSA-based addition operation";
  let description = [{
    Adds the left-hand-side operand to the right-hand-side operand and
    produces an explicit result.
  }];
}

///////////////////////////////////////////////////////////////////////////
// Subtraction Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_SubOp : SsaWasm_WasmNumericBinaryOp<"sub"> {
  let summary = "SSA-based subtraction operation";
  let description = [{
    Subtracts the right-hand-side operand from the left-hand-side operand
    and returns the result.
  }];
}

///////////////////////////////////////////////////////////////////////////
// Multiplication Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_MulOp : SsaWasm_WasmNumericBinaryOp<"mul"> {
  let summary = "SSA-based multiplication operation";
  let description = [{
    Multiplies two operands explicitly and yields the product.
  }];
}

///////////////////////////////////////////////////////////////////////////
// Function Operations
///////////////////////////////////////////////////////////////////////////

def SsaWasm_FuncOp : SsaWasm_Op<"func", [FunctionOpInterface]> {
    let summary = "SSA-based function operation";
    let description = [{
        This is the base class for function operations in the SSA Wasm dialect.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        Region *getCallableRegion() { return &getBody(); }
    }];
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def SsaWasm_ReturnOp : SsaWasm_Op<"return", [Terminator, HasParent<"FuncOp">, ReturnLike]> {
    let summary = "SSA-based return operation";
    let description = [{
        This is the base class for return operations in the SSA Wasm dialect.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [OpBuilder<(ins), [{
        build($_builder, $_state, std::nullopt);
    }]>];

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // SSAWASM_OPS
