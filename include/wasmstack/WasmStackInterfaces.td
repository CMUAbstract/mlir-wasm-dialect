//===- WasmStackInterfaces.td - WasmStack interfaces -----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_INTERFACES
#define WASMSTACK_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// StackEffectOpInterface
//===----------------------------------------------------------------------===//

def StackEffectOpInterface : OpInterface<"StackEffectOpInterface"> {
  let description = [{
    Interface for operations with stack effects in the wasmstack dialect.

    All wasmstack operations manipulate an implicit value stack. This interface
    provides methods to query the stack effects of each operation, enabling
    stack balance verification.

    Stack effects are expressed as:
    - stackInputs: Number of values consumed from the stack
    - stackOutputs: Number of values produced to the stack
    - stackInputTypes: Types of values consumed
    - stackOutputTypes: Types of values produced

    For polymorphic operations (e.g., select), types may depend on context.
  }];

  let cppNamespace = "::mlir::wasmstack";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get number of values consumed from the stack",
      /*retTy=*/"unsigned",
      /*methodName=*/"getStackInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return 0;"
    >,
    InterfaceMethod<
      /*desc=*/"Get number of values produced to the stack",
      /*retTy=*/"unsigned",
      /*methodName=*/"getStackOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return 0;"
    >,
    InterfaceMethod<
      /*desc=*/"Get input value types consumed from the stack",
      /*retTy=*/"::llvm::SmallVector<::mlir::Type>",
      /*methodName=*/"getStackInputTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {};"
    >,
    InterfaceMethod<
      /*desc=*/"Get output value types produced to the stack",
      /*retTy=*/"::llvm::SmallVector<::mlir::Type>",
      /*methodName=*/"getStackOutputTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {};"
    >,
  ];

  let verify = [{
    // No generic verification - ops implement their own verifiers
    return ::mlir::success();
  }];
}

//===----------------------------------------------------------------------===//
// WasmLabelOpInterface
//===----------------------------------------------------------------------===//

def WasmLabelOpInterface : OpInterface<"WasmLabelOpInterface"> {
  let description = [{
    Interface for operations that define branch targets in the wasmstack dialect.

    Block and loop operations define labels that can be targeted by br/br_if
    operations. This interface allows branch verifiers to find valid targets.
  }];

  let cppNamespace = "::mlir::wasmstack";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the label name for this block/loop",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getLabelName",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Check if this is a loop (branch goes to start, not end)",
      /*retTy=*/"bool",
      /*methodName=*/"isLoop",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<
      /*desc=*/"Get result types when branching to this target",
      /*retTy=*/"::mlir::TypeRange",
      /*methodName=*/"getBranchResultTypes",
      /*args=*/(ins)
    >,
  ];
}

#endif // WASMSTACK_INTERFACES
