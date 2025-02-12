//===- DContPass.td - DCont dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DCONT_DCONT_PASS
#define DCONT_DCONT_PASS

include "mlir/Pass/PassBase.td"


def ConvertIntermittentToDCont : Pass<"convert-intermittent-to-dcont", "ModuleOp"> {
  let summary = "Convert intermittent computing control flow to delimited continuations";
  let description = [{
    This pass converts intermittent computing control flow to delimited continuations.
  }];

  let dependentDialects = [
    "dcont::DContDialect",
    "intermittent::IntermittentDialect",
  ];
}

#endif // DCONT_DCONT_PASS
