//===- TransformsPasses.td - Dialect-independent passes -*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines dialect-independent transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_PASSES
#define TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def StrengthReduce : Pass<"strength-reduce", "ModuleOp"> {
  let summary = "Strength-reduce induction variable multiplies in SCF loops";
  let description = [{
    Replaces muli(cast(IV), loop-invariant) patterns inside scf.for loops
    with accumulator iter_args that increment by a fixed stride each iteration.
    This converts multiplications into additions (strength reduction).
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect"
  ];
}

#endif // TRANSFORMS_PASSES
