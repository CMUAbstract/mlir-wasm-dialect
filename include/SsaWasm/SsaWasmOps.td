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
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/Constraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

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
// Constant Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_ConstantOp : SsaWasm_Op<"constant", [Pure]> {
  let summary = "SSA-based constant operation";
  let description = [{
    The `ssawasm.constant` operation produces an SSA value equal to some
    constant specified by an attribute. This constant can be an integer
    or floating-point value.
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs SsaWasm_WasmNumericLike:$result);

  // FIXME: Make this pretty
  let assemblyFormat = "$value type($result) attr-dict";

  let builders = [
    OpBuilder<(ins "mlir::TypedAttr":$value), [{
      auto type = value.getType();
      if (type.isInteger(32)) {
        build($_builder, $_state, WasmIntegerType::get($_builder.getContext(), 32), value);
      } else if (type.isInteger(64)) {
        build($_builder, $_state, WasmIntegerType::get($_builder.getContext(), 64), value);
      } else if (type.isF32()) {
        build($_builder, $_state, WasmFloatType::get($_builder.getContext(), 32), value);
      } else if (type.isF64()) {
        build($_builder, $_state, WasmFloatType::get($_builder.getContext(), 64), value);
      } else {
        // TODO: handle error
      }
    }]>
  ];
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


def SsaWasm_CallOp : SsaWasm_Op<"call",
    [CallOpInterface]> {
    let summary = "call operation";

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>);

    let builders = [
      OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
        $_state.addOperands(operands);
        $_state.addAttribute("callee", SymbolRefAttr::get(callee));
        $_state.addTypes(callee.getFunctionType().getResults());
      }]>,
      OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
        CArg<"ValueRange", "{}">:$operands), [{
        $_state.addOperands(operands);
        $_state.addAttribute("callee", callee);
        $_state.addTypes(results);
      }]>,
      OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
        CArg<"ValueRange", "{}">:$operands), [{
        build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
      }]>,
      OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
        CArg<"ValueRange", "{}">:$operands), [{
        build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
              results, operands);
      }]>];

    let extraClassDeclaration = [{
      FunctionType getCalleeType();

      /// Get the argument operands to the called function.
      operand_range getArgOperands() {
        return {arg_operand_begin(), arg_operand_end()};
      }

      MutableOperandRange getArgOperandsMutable() {
        return getOperandsMutable();
      }

      operand_iterator arg_operand_begin() { return operand_begin(); }
      operand_iterator arg_operand_end() { return operand_end(); }

      /// Return the callee of this operation.
      CallInterfaceCallable getCallableForCallee() {
        return (*this)->getAttrOfType<SymbolRefAttr>("callee");
      }

      /// Set the callee for this operation.
      void setCalleeFromCallable(CallInterfaceCallable callee) {
        (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
      }
    }];

    let assemblyFormat = [{
      $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];
}

///////////////////////////////////////////////////////////////////////////
// Memory Operations
///////////////////////////////////////////////////////////////////////////

def MemRefTypeAttr
    : TypeAttrBase<"::mlir::MemRefType", "memref type attribute"> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

def SsaWasm_GlobalOp : SsaWasm_Op<"global", [Symbol]> {
   let arguments = (ins SymbolNameAttr:$sym_name,
                       MemRefTypeAttr:$type,
                       StrAttr:$initial_value,
                       UnitAttr:$constant,
                       I32Attr:$base_addr);

  let assemblyFormat = [{
       (`constant` $constant^)?
       $sym_name `:`
       custom<GlobalMemrefOpTypeAndInitialValue>($type, $initial_value)
       `at` $base_addr
       attr-dict
  }];

  let extraClassDeclaration = [{
     bool isExternal() { return false; } // TODO: support external globals
     bool isUninitialized() {
        return false; // TODO: support uninitialized globals
     }
  }];
}

def SsaWasm_GetGlobalOp : SsaWasm_Op<"get_global", [Pure]> {
  // TODO: Add DeclareOpInterfaceMethods<SymbolUserOpInterface> with a
  // verifier
  let summary = "similar to memref::GetGlobalOp";

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs SsaWasm_WasmMemRefType:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

def SsaWasm_LoadOp : SsaWasm_Op<"load"> {
  let arguments = (ins SsaWasm_WasmIntegerType:$addr);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def SsaWasm_StoreOp : SsaWasm_Op<"store"> {
  let arguments = (ins SsaWasm_WasmIntegerType:$addr, AnyType:$value);
  let hasCustomAssemblyFormat = 1;
}

///////////////////////////////////////////////////////////////////////////
// Local Operations
///////////////////////////////////////////////////////////////////////////
def SsaWasm_LocalOp : SsaWasm_Op<"local", [Pure, SameOperandsAndResultType]> {
    let summary = "SSA-based local variable operation";
    let description = [{
        This is the base class for local variable operations in the SSA Wasm dialect.
    }];

    let arguments = (ins SsaWasm_WasmNumericLike:$value);
    let results = (outs SsaWasm_WasmNumericLike:$result);

    let assemblyFormat = "attr-dict $value `:` type($result)";
}

#endif // SSAWASM_OPS
