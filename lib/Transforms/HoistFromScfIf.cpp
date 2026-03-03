//===- HoistFromScfIf.cpp - Hoist pure ops from scf.if regions ---*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that hoists pure (side-effect-free) operations
// out of scf.if regions when all their operands are defined outside the region.
//
// Motivation: After wami-convert-memref lowers memref operations, address
// arithmetic (arith.constant, arith.muli, arith.addi) is generated inline
// inside each scf.if region. MLIR's CSE cannot merge across region boundaries
// and MLIR's LICM only scans direct children of scf.for body blocks. This
// pass exposes those redundancies by moving pure ops to the parent scope.
//
// This mirrors LLVM's LICM behavior: LLVM unconditionally hoists pure
// loop-invariant operations from conditional blocks without any cost model
// (see llvm/lib/Transforms/Scalar/LICM.cpp, the three-gate check:
// hasLoopInvariantOperands, canSinkOrHoistInst,
// isSafeToExecuteUnconditionally). LLVM can do this because its flat CFG makes
// all basic blocks visible to LICM. MLIR's structured scf.if regions hide
// operations, so we need this explicit hoisting pass. Any over-hoisting is
// cleaned up by control-flow-sink.
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_HOISTFROMSCFIF
#include "Transforms/TransformsPasses.h.inc"

class HoistFromScfIf : public impl::HoistFromScfIfBase<HoistFromScfIf> {
public:
  using impl::HoistFromScfIfBase<HoistFromScfIf>::HoistFromScfIfBase;

  void runOnOperation() final {
    auto module = getOperation();

    // Collect all IfOps. walk() uses post-order (innermost first), which is
    // exactly the order we want: hoist from inner scf.if first so that ops
    // land in the outer region, then hoist from outer scf.if to move them
    // further out.
    SmallVector<scf::IfOp> ifOps;
    module.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

    for (auto ifOp : ifOps) {
      hoistFromRegion(ifOp.getThenRegion(), ifOp);
      if (!ifOp.getElseRegion().empty())
        hoistFromRegion(ifOp.getElseRegion(), ifOp);
    }
  }

private:
  /// Hoist pure operations with externally-defined operands out of a region.
  ///
  /// Operations are processed in forward order. When an early op (e.g.
  /// arith.constant) is hoisted, it leaves the region. Subsequent ops that
  /// depended on it (e.g. arith.muli) then see their operand as external,
  /// making them hoistable too. The chain unravels in a single forward scan.
  void hoistFromRegion(Region &region, Operation *insertBefore) {
    if (region.empty())
      return;

    Block &block = region.front();

    // Snapshot: collect ops to avoid iterator invalidation during moves.
    SmallVector<Operation *> ops;
    for (Operation &op : block)
      ops.push_back(&op);

    for (Operation *op : ops) {
      // Never move terminators (scf.yield).
      if (op->hasTrait<OpTrait::IsTerminator>())
        continue;

      // Only move side-effect-free operations.
      if (!isMemoryEffectFree(op))
        continue;

      // Check that all operands are defined outside this region.
      bool allExternal = llvm::all_of(op->getOperands(), [&](Value operand) {
        return !region.isAncestor(operand.getParentRegion());
      });
      if (!allExternal)
        continue;

      op->moveBefore(insertBefore);
    }
  }
};

} // namespace mlir::transforms
