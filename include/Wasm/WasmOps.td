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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/Constraints.td"

def MemRefTypeAttr
    : TypeAttrBase<"::mlir::MemRefType", "memref type attribute"> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

def Wasm_ConstantOp : Wasm_Op<"constant", [Pure]> {
    let arguments = (ins AnyAttr:$value);
    let assemblyFormat = "attr-dict $value";
    let hasVerifier = 1;
}

class Wasm_IntegerOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits> {
    let summary = "operations on i32 and i64 types"; 
    let arguments = (ins TypeAttr:$type);
    let assemblyFormat = [{ attr-dict `:` $type }];
}

class Wasm_FloatOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits> {
    let summary = "operations on f32 and f64 types"; 
    let arguments = (ins TypeAttr:$type);
    let assemblyFormat = [{ attr-dict `:` $type }];
}

// note that there is no wasm instruction that are unary and accept both integer
// and float types
class Wasm_NumericBinaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_Op<mnemonic, traits> {
    let summary = "operations on i32, i64, f32, and f64 types"; 
    let arguments = (ins TypeAttr:$type);
    let assemblyFormat = [{ attr-dict `:` $type }];
}

class Wasm_IntegerUnaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_IntegerOp<mnemonic, traits>;

class Wasm_IntegerBinaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_IntegerOp<mnemonic, traits>;

class Wasm_FloatUnaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_FloatOp<mnemonic, traits>;

class Wasm_FloatBinaryOp<string mnemonic, list<Trait> traits = []> 
    : Wasm_FloatOp<mnemonic, traits>;

// numeric binary operations
def Wasm_AddOp : Wasm_NumericBinaryOp<"add">;
def Wasm_SubOp : Wasm_NumericBinaryOp<"sub">;
def Wasm_MulOp : Wasm_NumericBinaryOp<"mul">;

// integer unary operations

def Wasm_IClzOp : Wasm_IntegerUnaryOp<"clz">;
def Wasm_ICtzOp : Wasm_IntegerUnaryOp<"ctz">;
def Wasm_IPopcntOp : Wasm_IntegerUnaryOp<"popcnt">;

// integer binary operations

def Wasm_IDivSOp : Wasm_IntegerBinaryOp<"div_s">;
def Wasm_IDivUOp : Wasm_IntegerBinaryOp<"div_u">;
def Wasm_IRemSOp : Wasm_IntegerBinaryOp<"rem_s">;
def Wasm_IRemUOp : Wasm_IntegerBinaryOp<"rem_u">;
def Wasm_IAndOp : Wasm_IntegerBinaryOp<"and">;
def Wasm_IOrOp : Wasm_IntegerBinaryOp<"or">;
def Wasm_IXorOp : Wasm_IntegerBinaryOp<"xor">;
def Wasm_IShlOp : Wasm_IntegerBinaryOp<"shl">;
def Wasm_IShrSOp : Wasm_IntegerBinaryOp<"shr_s">;
def Wasm_IShrUOp : Wasm_IntegerBinaryOp<"shr_u">;
def Wasm_IRotlOp : Wasm_IntegerBinaryOp<"rotl">;
def Wasm_IRotrOp : Wasm_IntegerBinaryOp<"rotr">;

def Wasm_ILeUOp : Wasm_IntegerBinaryOp<"le_u">;

// floating-point unary operations

def Wasm_FAbsOp : Wasm_FloatUnaryOp<"abs">;
def Wasm_FNegOp : Wasm_FloatUnaryOp<"neg">;
def Wasm_FSqrtOp : Wasm_FloatUnaryOp<"sqrt">;
def Wasm_FCeilOp : Wasm_FloatUnaryOp<"ceil">;
def Wasm_FFloorOp : Wasm_FloatUnaryOp<"floor">;
def Wasm_FTruncOp : Wasm_FloatUnaryOp<"trunc">;
def Wasm_FNearesetOp : Wasm_FloatUnaryOp<"nearest">;

// floating-point binary operations

def Wasm_FDivOp : Wasm_FloatBinaryOp<"div">;
def Wasm_FMinOp : Wasm_FloatBinaryOp<"min">;
def Wasm_FMaxOp : Wasm_FloatBinaryOp<"max">;
def Wasm_FCopysignOp : Wasm_FloatBinaryOp<"copysign">;

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

// temporary instructions

def Wasm_TempLocalOp : Wasm_Op<"temp.local", []> {
    let arguments = (ins TypeAttr:$type);
    let results = (outs Local:$local);
    // TODO: Improve assembly format
    let assemblyFormat = [{ `<` $type `>` attr-dict `:` type(results) }];

    let builders = [
        OpBuilder<(ins "Type": $inner)>,
    ];
}

def Wasm_TempLocalGetOp : Wasm_Op<"temp.local.get", []> {
    let arguments = (ins Local:$local);
    let hasCustomAssemblyFormat = 1;
}

def Wasm_TempLocalSetOp : Wasm_Op<"temp.local.set", []> {
    let arguments = (ins Local:$local);
    let hasCustomAssemblyFormat = 1;
}

// local instructions

def Wasm_LocalOp : Wasm_Op<"local", []> {
    let summary = "Local declaration operation";
    let description = [{
        The `local` operation declares local variables.
    }];
    let arguments = (ins TypeArrayAttr:$types);
    let assemblyFormat = [{ attr-dict `:` $types }];
    // TODO: arguments should be a list of types
}

def Wasm_LocalGetOp : Wasm_Op<"local.get", []> {
    let summary = "Local get operation";
    let description = [{
        The `local.get` operation reads a local variable.
    }];
    let arguments = (ins IndexAttr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_LocalSetOp : Wasm_Op<"local.set", []> {
    let summary = "Local set operation";
    let description = [{
        The `local.set` operation writes a local variable.
    }];
    let arguments = (ins IndexAttr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_LocalTeeOp : Wasm_Op<"local.tee", []> {
    let summary = "Local tee operation";
    let description = [{
        The `local.tee` operation writes a local variable and returns the value.
    }];
    let arguments = (ins IndexAttr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_GlobalGetOp : Wasm_Op<"global.get", []> {
    let summary = "Global get operation";
    let description = [{
        The `global.get` operation reads a global variable.
    }];
    let arguments = (ins IndexAttr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_GlobalSetOp : Wasm_Op<"global.set", []> {
    let summary = "Global set operation";
    let description = [{
        The `global.set` operation writes a global variable.
    }];
    let arguments = (ins IndexAttr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

// table instructions

def Wasm_TableGetOp : Wasm_Op<"table.get", []> {
    let summary = "Table get operation";
    let description = [{
        The `table.get` operation reads a table element.
    }];
    let arguments = (ins IndexAttr:$idx);
    let assemblyFormat = [{ $idx attr-dict }];
}

def Wasm_TableSetOp : Wasm_Op<"table.set", []> {
    let summary = "Table set operation";
    let description = [{
        The `table.set` operation writes a table element.
    }];
    let arguments = (ins IndexAttr:$idx);
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
    let arguments = (ins TypeAttr : $type);
    let assemblyFormat = [{ attr-dict `:` $type }];
}

def Wasm_StoreOp : Wasm_Op<"store", []> {
    let arguments = (ins TypeAttr : $type);
    let assemblyFormat = [{ attr-dict `:` $type }];
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

def Wasm_DataOp : Wasm_Op<"data"> {
    let arguments = (ins SymbolNameAttr:$sym_name, 
        I32Attr:$offset, StrAttr:$bytes, OptionalAttr<MemRefTypeAttr>:$type); 
    let assemblyFormat = [{ 
        attr-dict $sym_name `:`
        `{` `offset` `=` $offset `,`  `bytes` `=` $bytes `}`
    }];
}

def Wasm_TempGetGlobalOp : Wasm_Op<"temp.get_global", 
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$name);
    let results = (outs AnyStaticShapeMemRef:$result);
    let assemblyFormat = [{ $name `:` type($result) attr-dict }];
}

def Wasm_CallOp : Wasm_Op<"call"> {
    let arguments = (ins FlatSymbolRefAttr:$name);
    let assemblyFormat = [{ $name attr-dict }];

    let extraClassDeclaration = [{
        StringRef getCallee() { return getName(); }
    }];
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

def Wasm_LoopOp: Wasm_Op<"loop"> {
    // TODO: Handle loop-carried values
    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins)>,
    ];
}

def Wasm_LoopEndOp: Wasm_Op<"loop_end", [Terminator]> {
    // TODO: HasParent<LoopOp>
}

def Wasm_BranchOp: Wasm_Op<"br", [Terminator]> {
    let successors = (successor AnySuccessor:$dest);
}

def Wasm_CondBranchOp: Wasm_Op<"cond_br", [Terminator]> {
    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
}

def Wasm_NoOp : Wasm_Op<"noop", []> {
    let summary = "No operation";
    let description = [{
        The `noop` operation does nothing.
    }];
}

def Wasm_WasmFuncOp : Wasm_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {

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

def Wasm_WasmReturnOp : Wasm_Op<"return", [Pure, Terminator, HasParent<"WasmFuncOp">, ReturnLike]> {

    // NOTE: WasmReturnOp does not have any operands
    // The return operation only emits the input in the format if it is present.
    let assemblyFormat = "attr-dict";
}


def WasmFunctionTypeOp : Wasm_Op<"type.func", [Pure]> {
    let summary = "Declare a function type";
    let description = [{
        Declares a function type with specified parameter and result types.
    }];
    
    // TODO: Add typechecking
    let arguments = (ins SymbolNameAttr:$name, TypeAttrOf<FunctionType>:$function_type);
    let assemblyFormat = [{
        $name $function_type attr-dict
    }];
}

def WasmContinuationTypeOp : Wasm_Op<"type.ct", [Pure]> {
  let summary = "Declares a continuation type";
  let description = [{
    Declares a control tag with specified parameter and result types.
  }];
  let arguments = (ins SymbolNameAttr:$name, FlatSymbolRefAttr:$func);
  let assemblyFormat = "`type` $name `(` `cont` $func `)` attr-dict";
}

def WasmTagOp : Wasm_Op<"tag", [Pure]> {
  let summary = "Declares a control tag";
  let description = [{
    Declares a control tag with specified parameter and result types.
  }];
  let arguments = (ins SymbolNameAttr:$name);
  let assemblyFormat = "`tag` $name attr-dict";
}

def WasmFuncRefOp : Wasm_Op<"func.ref"> {
  let summary = "Creates a function reference";
  let description = [{
    Creates a function reference and pushes it onto the stack.
  }];
  let arguments = (ins FlatSymbolRefAttr:$func);
  let assemblyFormat = "`ref` $func attr-dict";
}

def WasmContNewOp : Wasm_Op<"cont.new"> {
  let summary = "Creates a suspended continuation from a function";
  let description = [{
    Creates a suspended continuation from a function reference and pushes it onto the stack.
    Assumes that a function reference is on the top of the stack.
  }];
  let arguments = (ins FlatSymbolRefAttr:$ct);
  // NOTE: We cannot use cont.new because it includes a dot
  let assemblyFormat = "`cont` `_` `new` $ct attr-dict";
}

// TODO: Ideally, `resume` operation should
// take a variadic list of handlers. 
// For now, we are assuming that there is only one handler
// of form (on $e switch)
def WasmResumeSwitchOp : Wasm_Op<"resume.switch"> {
  let summary = "Resume a continuation under a handler";
  let description = [{
    Resumes a continuation under a specified handler.
    Assumes that a continuation is on the top of the stack.
  }];
  let arguments = (ins FlatSymbolRefAttr:$ct, FlatSymbolRefAttr:$tag); 
  let assemblyFormat = "`resume` $ct `(` `on` $tag `switch` `)` attr-dict";
}

def WasmSwitchOp : Wasm_Op<"switch"> {
  let summary = "Switch to a continuation";
  let description = [{
    Switches to a continuation under a specified handler.
    Assumes that a continuation is on the top of the stack.
  }];
  let arguments = (ins FlatSymbolRefAttr:$ct, FlatSymbolRefAttr:$tag); 
  let assemblyFormat = "`switch` $ct `(` `on` $tag `)` attr-dict";
}



#endif // WASM_OPS
