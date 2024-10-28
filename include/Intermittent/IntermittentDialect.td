//===- IntermittentDialect.td - Wasm dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INTERMITTENT_DIALECT
#define INTERMITTENT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Intermittent dialect definition.
//===----------------------------------------------------------------------===//

def Intermittent_Dialect : Dialect {
    let name = "intermittent";
    let summary = "intermittent dialect";
    let description = [{
        A dialect for task-based intermittent computing.
    }];
    let cppNamespace = "::mlir::intermittent";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let dependentDialects = [
    ];
}

//===----------------------------------------------------------------------===//
// Base wasm operation definition.
//===----------------------------------------------------------------------===//

class Wasm_Op<string mnemonic, list<Trait> traits = []> :
        Op<Intermittent_Dialect, mnemonic, traits>;

#endif // INTERMITTENT_DIALECT
