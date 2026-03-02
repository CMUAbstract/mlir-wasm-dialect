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

    By default, only applies when the induction variable has no remaining uses
    in the loop body after replacement (no register pressure increase).
    Use aggressive=true to always apply the transformation.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect"
  ];
  let options = [
    Option<"aggressive", "aggressive", "bool", /*default=*/"false",
           "Apply strength reduction even when the induction variable "
           "remains live (increases register pressure)">
  ];
}

#endif // TRANSFORMS_PASSES
