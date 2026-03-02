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

def ScfLoopUnroll : Pass<"scf-loop-unroll", "ModuleOp"> {
  let summary = "Unroll SCF for loops by a given factor";
  let description = [{
    Unrolls scf.for loops by the specified factor using MLIR's
    loopUnrollByFactor utility. This is equivalent to affine-loop-unroll
    but operates on SCF loops instead of affine loops.
  }];
  let options = [
    Option<"unrollFactor", "unroll-factor", "unsigned", "4",
           "Loop unroll factor">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect"
  ];
}

def LoopAccumulatorPromotion : Pass<"promote-loop-accumulators", "ModuleOp"> {
  let summary = "Promote memory load-accumulate-store patterns to iter_args";
  let description = [{
    Detects load-accumulate-store patterns inside scf.for loops where a memref
    location is loaded, accumulated with an arithmetic operation, and stored
    back every iteration. Promotes such patterns to use iter_args, hoisting the
    initial load before the loop and the final store after the loop.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect"
  ];
}

#endif // TRANSFORMS_PASSES
