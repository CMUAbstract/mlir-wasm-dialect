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
    Eliminates muli(cast(IV), K) and shli(cast(IV), N) patterns inside scf.for
    loops using two strategies:

    1. Bounds-based (preferred): Folds the multiply factor into the loop bounds,
       replacing lb/ub/step with lb*K/ub*K/step*K. Requires all candidates to
       share the same factor and the IV to have no surviving uses.

    2. Accumulator-based (fallback): Converts each multiply into a loop-carried
       accumulator that increments by K*step each iteration. Handles multiple
       different factors in the same loop and cases where the IV is still alive.
       Only applies when the transformation does not increase register pressure.
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

def AffinePromoteAccumulators : Pass<"affine-promote-accumulators", "func::FuncOp"> {
  let summary = "Promote affine load-accumulate-store patterns to iter_args";
  let description = [{
    Identifies loop-invariant affine.load/affine.store pairs inside affine.for
    loops where the store value depends on the load, and promotes them to
    iter_args. Uses affine dependence analysis to prove non-aliasing with
    other memory accesses in the loop.
  }];
  let dependentDialects = [
    "affine::AffineDialect"
  ];
}

def RemainderToAnd : Pass<"remainder-to-and", "ModuleOp"> {
  let summary = "Convert remainder by power-of-2 to bitwise AND";
  let description = [{
    Converts unsigned and signed remainder operations by power-of-2 constants
    to bitwise AND operations:
      remui(x, 2^k) -> andi(x, 2^k - 1)  (always correct)
      remsi(x, 2^k) -> andi(x, 2^k - 1)  (when x is provably non-negative)

    Uses MLIR's IntegerRangeAnalysis to prove non-negativity for signed cases.
    Leaves remainder by 1 untouched for existing constant-folding passes.
  }];
  let dependentDialects = ["arith::ArithDialect"];
}

def HoistFromScfIf : Pass<"hoist-from-scf-if", "ModuleOp"> {
  let summary = "Hoist pure operations out of scf.if regions";
  let description = [{
    Moves pure (side-effect-free) operations whose operands are all defined
    outside the scf.if region to before the scf.if in the parent block.
    Processes bottom-up (innermost scf.if first) to handle nesting.
    Exposes redundant computations to subsequent CSE and LICM.

    This mirrors what LLVM's LICM does unconditionally for pure arithmetic
    inside conditional blocks (llvm/lib/Transforms/Scalar/LICM.cpp). LLVM's
    flat CFG makes all basic blocks visible to loop-level LICM, but MLIR's
    structured scf.if regions hide operations from its LICM. This pass
    bridges that gap. Any over-hoisting can be cleaned up by control-flow-sink.
  }];
  let dependentDialects = ["scf::SCFDialect"];
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
