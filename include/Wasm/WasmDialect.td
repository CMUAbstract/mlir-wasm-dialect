//===- WasmDialect.td - Wasm dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASM_DIALECT
#define WASM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Wasm dialect definition.
//===----------------------------------------------------------------------===//

def Wasm_Dialect : Dialect {
    let name = "wasm";
    let summary = "wasm dialect";
    let description = [{
        This is an MLIR wasm dialect. TODO
    }];
    let cppNamespace = "::mlir::wasm";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base wasm operation definition.
//===----------------------------------------------------------------------===//

class Wasm_Op<string mnemonic, list<Trait> traits = []> :
        Op<Wasm_Dialect, mnemonic, traits>;

#endif // WASM_DIALECT
