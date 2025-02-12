//===- DContDialect.td - DCont dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DCONT_DIALECT
#define DCONT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DCont dialect definition.
//===----------------------------------------------------------------------===//

def DCont_Dialect : Dialect {
    let name = "dcont";
    let summary = "dcont dialect";
    let description = [{
        This dialect models delimited continuations,
        which can be used to model non-local control flow.
    }];
    let cppNamespace = "::mlir::dcont";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let dependentDialects = [
        "func::FuncDialect",
    ];
}


#endif // DCONT_DIALECT
