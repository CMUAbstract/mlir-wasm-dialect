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
    Folds muli(cast(IV), K) patterns inside scf.for loops into the loop bounds,
    replacing lb/ub/step with lb*K/ub*K/step*K and eliminating the multiply.

    Only applies when ALL multiply candidates in a loop share the same factor K
    and the induction variable has no uses outside the multiply chain.
    Never introduces additional loop-carried variables.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect"
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
