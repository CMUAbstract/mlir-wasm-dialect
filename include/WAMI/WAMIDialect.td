//===- WAMIDialect.td - WAMI dialect definition --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_DIALECT
#define WAMI_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// WAMI dialect definition.
//===----------------------------------------------------------------------===//

def WAMI_Dialect : Dialect {
    let name = "wami";
    let summary = "WAMI dialect for WebAssembly extensions";
    let description = [{
        The WAMI dialect extends the upstream WasmSSA dialect with additional
        operations for WebAssembly proposals (e.g., stack switching) and
        WAMI-specific features. It operates at the same SSA abstraction level
        as WasmSSA and both dialects lower together to stack-based WebAssembly.
    }];
    let cppNamespace = "::mlir::wami";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let dependentDialects = [
        "mlir::wasmssa::WasmSSADialect",
    ];
}

#endif // WAMI_DIALECT
