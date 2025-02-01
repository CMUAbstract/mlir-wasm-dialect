//===- SsaWasmDialect.td - SsaWasm dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SSAWASM_DIALECT
#define SSAWASM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SsaWasm dialect definition.
//===----------------------------------------------------------------------===//

def SsaWasm_Dialect : Dialect {
    let name = "ssawasm";
    let summary = "ssawasm dialect";
    let description = [{
        The ssaWasm dialect is a dialect for the WebAssembly language. It is a
        dialect that uses SSA form, and later will be converted to the Wasm
        dialect through stackification.
    }];
    let cppNamespace = "::mlir::ssawasm";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let dependentDialects = [
        "arith::ArithDialect",
        "func::FuncDialect",
        "scf::SCFDialect",
        "memref::MemRefDialect",
    ];
}


#endif // SSAWASM_DIALECT
