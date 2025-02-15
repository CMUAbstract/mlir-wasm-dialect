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
    : SsaWasm_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
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

def SsaWasm_AddOp : SsaWasm_WasmNumericBinaryOp<"add", [Commutative, AlwaysSpeculatableImplTrait]> {
  let summary = "SSA-based addition operation";
  let description = [{
    Adds the left-hand-side operand to the right-hand-side operand and
    produces an explicit result.
  }];

  let hasFolder = 1;
}

///////////////////////////////////////////////////////////////////////////
// Subtraction Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_SubOp : SsaWasm_WasmNumericBinaryOp<"sub", [AlwaysSpeculatableImplTrait]> {
  let summary = "SSA-based subtraction operation";
  let description = [{
    Subtracts the right-hand-side operand from the left-hand-side operand
    and returns the result.
  }];

  let hasFolder = 1;
}

///////////////////////////////////////////////////////////////////////////
// Multiplication Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_MulOp : SsaWasm_WasmNumericBinaryOp<"mul", [Commutative, AlwaysSpeculatableImplTrait]> {
  let summary = "SSA-based multiplication operation";
  let description = [{
    Multiplies two operands explicitly and yields the product.
  }];

  let hasFolder = 1;
}

def SsaWasm_ILeUOp : SsaWasm_WasmNumericBinaryOp<"ileu", [AlwaysSpeculatableImplTrait]> {
  let summary = "SSA-based integer less than or equal to unsigned operation";
  let description = [{
    Compares two operands explicitly and yields the result of the comparison.
  }];
}

def SsaWasm_MinOp : SsaWasm_WasmNumericBinaryOp<"min", [AlwaysSpeculatableImplTrait]> {
  let summary = "SSA-based minimum operation";
  let description = [{
    Compares two operands explicitly and yields the minimum of the two.
  }];

  let hasFolder = 1;
}

def SsaWasm_MaxOp : SsaWasm_WasmNumericBinaryOp<"max", [AlwaysSpeculatableImplTrait]> {
  let summary = "SSA-based maximum operation";
  let description = [{
    Compares two operands explicitly and yields the maximum of the two.
  }];

  let hasFolder = 1;
}

/////////////////////////////////////////////////////////////////////////// 
// Constant Operation
///////////////////////////////////////////////////////////////////////////

def SsaWasm_ConstantOp : SsaWasm_Op<"constant", [Pure, ConstantLike, AlwaysSpeculatableImplTrait]> {
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
      if (type.isInteger(32) || type.isIndex()) {
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

  let hasFolder = 1;
}

///////////////////////////////////////////////////////////////////////////
// Function Operations
///////////////////////////////////////////////////////////////////////////

def SsaWasm_FuncOp : SsaWasm_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
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

def SsaWasm_DataOp : SsaWasm_Op<"data", [Symbol]> {
   let arguments = (ins SymbolNameAttr:$sym_name,
                       MemRefTypeAttr:$type,
                       AnyAttr:$initial_value,
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

def SsaWasm_GetDataOp : SsaWasm_Op<"get_data", [Pure]> {
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

def SsaWasm_AsPointerOp : SsaWasm_Op<"as_pointer"> {
  let arguments = (ins SsaWasm_WasmPointerLike:$value);
  let results = (outs SsaWasm_WasmIntegerType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";

  let builders = [OpBuilder<(ins "Value":$value), [{
    $_state.addOperands(value);
    $_state.addTypes(WasmIntegerType::get($_builder.getContext(), 32));
  }]>];
}

def SsaWasm_AsMemRefOp : SsaWasm_Op<"as_memref"> {
  let arguments = (ins SsaWasm_WasmPointerLike:$value);
  let results = (outs SsaWasm_WasmMemRefType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
}

///////////////////////////////////////////////////////////////////////////
// Local Operations
///////////////////////////////////////////////////////////////////////////
def SsaWasm_LocalOp : SsaWasm_Op<"local", [SameOperandsAndResultType]> {
    let summary = "SSA-based local variable operation";
    let description = [{
        This is the base class for local variable operations in the SSA Wasm dialect.
    }];

    let arguments = (ins SsaWasm_WasmNumericLike:$value);
    let results = (outs SsaWasm_WasmNumericLike:$result);

    let assemblyFormat = "attr-dict $value `:` type($result)";
}

// TODO: Enforce that local and value have the same type
def SsaWasm_LocalSetOp : SsaWasm_Op<"local_set"> {
  let summary = "local variable set operation";
  let description = [{
    This operation sets the value of a local variable.
    The definingOp of `local` must be LocalOp.
  }];
  // TODO: Type-level enforcement of the above constraint
  let arguments = (ins SsaWasm_WasmNumericLike:$local, SsaWasm_WasmNumericLike:$value);
  let assemblyFormat = "$local `,` $value `:` type($local) `->` type($value) attr-dict";
}

// TODO: Enforce that local and value have the same type
def SsaWasm_LocalGetOp : SsaWasm_Op<"local_get"> {
  let summary = "local variable get operation";
  let description = [{
    This operation gets the value of a local variable.
    The definingOp of `local` must be LocalOp.
  }];
  // TODO: Type-level enforcement of the above constraint
  let arguments = (ins SsaWasm_WasmNumericLike:$local);
  let results = (outs SsaWasm_WasmNumericLike:$result);
  let assemblyFormat = "$local `:` type($local) `->` type($result) attr-dict";
}

def SsaWasm_BlockLoopOp : SsaWasm_Op<"blockloop"> {
  let description = [{
    This is a temporary operation used to model 
    a loop inside a block.
  }];
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    // TODO: Convert this to a builder
    std::tuple<Block *, Block *, Block *> initialize(OpBuilder &builder);
    Block *getEntryBlock();
    Block *getExitBlock();
  }];
}


def SsaWasm_BlockLoopTerminatorOp : SsaWasm_Op<"blockloop.terminator", [Terminator]> {
  let description = [{
    This is a temporary operation to be put at the exit block of a BlockLoopOp.
  }];
}

def SsaWasm_BlockLoopBranchOp : SsaWasm_Op<"blockloop.br", 
  [Terminator, HasParent<"BlockLoopOp">]> {
  let successors = (successor AnySuccessor:$dest);
  let extraClassDeclaration = [{
    bool isBranchingToBegin();
  }];
}

def SsaWasm_BlockLoopCondBranchOp : SsaWasm_Op<"blockloop.cond_br", 
  [Terminator, HasParent<"BlockLoopOp">]> {
  let arguments = (ins SsaWasm_WasmIntegerType:$condition);
  let successors = (successor AnySuccessor:$dest, AnySuccessor:$falseDest);
  let extraClassDeclaration = [{
    bool isBranchingToBegin();
  }];
}

def SsaWasm_TempBranchOp : SsaWasm_Op<"temp_branch", [Terminator]> {
  let description = [{
    This is a temporary operation used to model branch within a BlockOp or
    LoopOp.
    Must branch to the next block in the region.
    Should be removed when lowering to Wasm.
  }];
  let successors = (successor AnySuccessor:$dest);
}

// stack switching related ops

def SsaWasm_TagOp : SsaWasm_Op<"tag", [Symbol]> {
  let description = [{
    This operation declares a tag.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "attr-dict $sym_name";
}

def SsaWasm_ContNewOp : SsaWasm_Op<"cont_new"> {
  let description = [{
    This operation creates a new continuation.
  }];
  let arguments = (ins FlatSymbolRefAttr:$function_name);
  let results = (outs SsaWasm_WasmContinuationType:$result);
  let assemblyFormat = "attr-dict $function_name `->` type($result)";
}

def SsaWasm_ContNullOp : SsaWasm_Op<"cont_null"> {
  let description = [{
    This operation creates a null continuation.
  }];
  let results = (outs SsaWasm_WasmContinuationType:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def SsaWasm_ResumeSwitchOp : SsaWasm_Op<"resume_switch"> {
  let description = [{
    This corresponds to the `resume` operation in Wasm with a switch handler.
  }];
  let arguments = (ins FlatSymbolRefAttr:$tag, SsaWasm_WasmContinuationType:$cont, Variadic<AnyType>:$args);
  let results = (outs SsaWasm_WasmContinuationType:$returned_cont, Variadic<AnyType>:$results);
}

def SsaWasm_SwitchOp : SsaWasm_Op<"switch"> {
  let description = [{
    This corresponds to the `switch` operation in Wasm.
  }];
  let arguments = (ins SsaWasm_WasmContinuationType:$cont, Variadic<AnyType>:$args);
  let results = (outs SsaWasm_WasmContinuationType:$returned_cont, Variadic<AnyType>:$results);
}

#endif // SSAWASM_OPS
