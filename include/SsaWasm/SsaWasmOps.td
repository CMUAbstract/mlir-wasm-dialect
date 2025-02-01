
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

class SsaWasm_NumericBinaryOp<string mnemonic, list<Trait> traits = []> 
    : SsaWasm_Op<mnemonic, traits # [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "SSA-based numeric binary operation";
  let description = [{
    This is the base class for numeric binary operations in the SSA Wasm dialect.
    Both operands and the result are explicit, allowing for easier analysis and optimization.
  }];

  // Two explicit operands and one result.
  let arguments = (ins SsaWasmNumericLike:$lhs, SsaWasmNumericLike:$rhs);
  let results = (outs SsaWasmNumericLike:$result);

  // Assembly format shows the two operands followed by the type of the result.
  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($result)";
}

///////////////////////////////////////////////////////////////////////////
// Addition Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_AddOp : SsaWasm_NumericBinaryOp<"add"> {
  let summary = "SSA-based addition operation";
  let description = [{
    Adds the left-hand-side operand to the right-hand-side operand and
    produces an explicit result.
  }];
}

///////////////////////////////////////////////////////////////////////////
// Subtraction Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_SubOp : SsaWasm_NumericBinaryOp<"sub"> {
  let summary = "SSA-based subtraction operation";
  let description = [{
    Subtracts the right-hand-side operand from the left-hand-side operand
    and returns the result.
  }];
}

///////////////////////////////////////////////////////////////////////////
// Multiplication Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_MulOp : SsaWasm_NumericBinaryOp<"mul"> {
  let summary = "SSA-based multiplication operation";
  let description = [{
    Multiplies two operands explicitly and yields the product.
  }];
}


#endif // SSAWASM_OPS
