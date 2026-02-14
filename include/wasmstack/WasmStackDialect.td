//===- WasmStackDialect.td - WasmStack dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_DIALECT
#define WASMSTACK_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// WasmStack dialect definition.
//===----------------------------------------------------------------------===//

def WasmStack_Dialect : Dialect {
    let name = "wasmstack";
    let summary = "Stack-based WebAssembly dialect for binary emission";
    let description = [{
        The wasmstack dialect represents WebAssembly instructions in their native
        stack-based form. Unlike SSA-form dialects, operations in wasmstack have
        implicit operands and results that come from/go to the value stack.

        Key design principles:
        - Implicit operands: True stack machine, no SSA values for stack ops
        - All ops are side-effectful: Prevents unwanted pass transformations
        - NoTerminator regions: Opts out of MLIR's CFG analysis
        - Symbolic labels: Readable branch targets, emitter computes depth

        This dialect is the final lowering target before binary emission.
    }];
    let cppNamespace = "::mlir::wasmstack";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

#endif // WASMSTACK_DIALECT
