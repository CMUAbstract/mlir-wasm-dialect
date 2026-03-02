//===- ScfLoopUnroll.cpp - SCF loop unrolling pass ----------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unrolling for SCF for loops.
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_SCFLOOPUNROLL
#include "Transforms/TransformsPasses.h.inc"

class ScfLoopUnroll : public impl::ScfLoopUnrollBase<ScfLoopUnroll> {
public:
  using impl::ScfLoopUnrollBase<ScfLoopUnroll>::ScfLoopUnrollBase;

  void runOnOperation() final {
    if (unrollFactor <= 1)
      return;

    auto module = getOperation();

    // Collect only innermost ForOps (those whose body contains no nested
    // scf.for). Unrolling outer loops causes massive code bloat with minimal
    // benefit since outer loop overhead is negligible relative to inner work.
    SmallVector<scf::ForOp> innermostOps;
    module.walk([&](scf::ForOp forOp) {
      bool hasNestedFor = false;
      forOp.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
      if (!hasNestedFor)
        innermostOps.push_back(forOp);
    });

    for (auto forOp : innermostOps)
      (void)loopUnrollByFactor(forOp, unrollFactor);
  }
};

} // namespace mlir::transforms
