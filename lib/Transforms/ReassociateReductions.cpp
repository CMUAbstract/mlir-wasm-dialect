//===- ReassociateReductions.cpp - Tree-reduce sequential chains --*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After loop unrolling, iter_arg reductions become left-folded sequential
// chains (e.g., max(max(max(acc, v0), v1), v2)). This pass restructures
// them into balanced trees (max(acc, max(max(v0, v1), max(v2, v3)))) to
// reduce loop-carried dependency depth from N to 1.
//
//===----------------------------------------------------------------------===//

#include "Transforms/TransformsPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::transforms {

#define GEN_PASS_DEF_REASSOCIATEREDUCTIONS
#include "Transforms/TransformsPasses.h.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return true if `val` transitively depends on `arg` through the SSA
/// def-chain within `arg`'s block.
static bool dependsOnArg(Value val, BlockArgument arg) {
  if (val == arg)
    return true;
  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return false;
  Block *block = arg.getOwner();
  SmallVector<Operation *, 16> worklist{defOp};
  llvm::DenseSet<Operation *> visited;
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    if (!visited.insert(op).second)
      continue;
    for (Value operand : op->getOperands()) {
      if (operand == arg)
        return true;
      if (auto *dep = operand.getDefiningOp())
        if (dep->getBlock() == block)
          worklist.push_back(dep);
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Reduction pattern matching
//===----------------------------------------------------------------------===//

/// Describes the "kind" of a reduction so that chain elements can be compared
/// and new reductions can be reconstructed.
struct ReductionKind {
  bool isCmpiSelect = false;
  arith::CmpIPredicate pred = arith::CmpIPredicate::eq;
  bool selectTrueIsCmpiLhs = true;
  StringRef directOpName;

  bool operator==(const ReductionKind &o) const {
    if (isCmpiSelect != o.isCmpiSelect)
      return false;
    if (isCmpiSelect)
      return pred == o.pred && selectTrueIsCmpiLhs == o.selectTrueIsCmpiLhs;
    return directOpName == o.directOpName;
  }
};

/// Result of matching a single value as a reduction operation.
struct ReductionMatch {
  ReductionKind kind;
  Value opA, opB;
  SmallVector<Operation *, 2> ops; // for cleanup
};

/// Try to match `val` as a commutative, associative reduction operation.
static std::optional<ReductionMatch> matchReduction(Value val) {
  // Pattern 1: select(cmpi <order-pred>, A, B), ... -> max/min
  if (auto sel = val.getDefiningOp<arith::SelectOp>()) {
    if (auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>()) {
      if (cmp.getResult().hasOneUse()) {
        auto pred = cmp.getPredicate();
        using P = arith::CmpIPredicate;
        if (pred == P::sge || pred == P::sgt || pred == P::sle ||
            pred == P::slt || pred == P::uge || pred == P::ugt ||
            pred == P::ule || pred == P::ult) {
          Value cL = cmp.getLhs(), cR = cmp.getRhs();
          Value sT = sel.getTrueValue(), sF = sel.getFalseValue();
          if (sT == cL && sF == cR)
            return ReductionMatch{{true, pred, true, {}},
                                  cL,
                                  cR,
                                  {sel.getOperation(), cmp.getOperation()}};
          if (sT == cR && sF == cL)
            return ReductionMatch{{true, pred, false, {}},
                                  cR,
                                  cL,
                                  {sel.getOperation(), cmp.getOperation()}};
        }
      }
    }
  }

  // Pattern 2: direct associative + commutative binary op
  auto *op = val.getDefiningOp();
  if (op && op->getNumOperands() == 2 && op->getNumResults() == 1 &&
      isa<arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
          arith::AddIOp, arith::MulIOp, arith::AndIOp, arith::OrIOp,
          arith::XOrIOp>(op))
    return ReductionMatch{{false, {}, true, op->getName().getStringRef()},
                          op->getOperand(0),
                          op->getOperand(1),
                          {op}};

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Tree construction
//===----------------------------------------------------------------------===//

/// Build a single reduction op matching `kind` with the given operands.
static Value buildReduction(IRRewriter &rewriter, Location loc,
                            const ReductionKind &kind, Value lhs, Value rhs) {
  if (kind.isCmpiSelect) {
    Value cmp = arith::CmpIOp::create(rewriter, loc, kind.pred, lhs, rhs);
    Value selT = kind.selectTrueIsCmpiLhs ? lhs : rhs;
    Value selF = kind.selectTrueIsCmpiLhs ? rhs : lhs;
    return arith::SelectOp::create(rewriter, loc, cmp, selT, selF);
  }
  // Direct op.
  StringRef name = kind.directOpName;
  if (name == "arith.maxsi")
    return arith::MaxSIOp::create(rewriter, loc, lhs, rhs);
  if (name == "arith.maxui")
    return arith::MaxUIOp::create(rewriter, loc, lhs, rhs);
  if (name == "arith.minsi")
    return arith::MinSIOp::create(rewriter, loc, lhs, rhs);
  if (name == "arith.minui")
    return arith::MinUIOp::create(rewriter, loc, lhs, rhs);
  if (name == "arith.addi")
    return arith::AddIOp::create(rewriter, loc, lhs, rhs);
  if (name == "arith.muli")
    return arith::MulIOp::create(rewriter, loc, lhs, rhs);
  if (name == "arith.andi")
    return arith::AndIOp::create(rewriter, loc, lhs, rhs);
  if (name == "arith.ori")
    return arith::OrIOp::create(rewriter, loc, lhs, rhs);
  assert(name == "arith.xori");
  return arith::XOrIOp::create(rewriter, loc, lhs, rhs);
}

/// Pairwise-reduce `vals` into a balanced binary tree.
static Value buildBalancedTree(IRRewriter &rewriter, Location loc,
                               const ReductionKind &kind,
                               MutableArrayRef<Value> vals) {
  assert(!vals.empty());
  while (vals.size() > 1) {
    size_t n = vals.size(), out = 0;
    for (size_t i = 0; i + 1 < n; i += 2)
      vals[out++] = buildReduction(rewriter, loc, kind, vals[i], vals[i + 1]);
    if (n & 1)
      vals[out++] = vals[n - 1];
    vals = vals.take_front(out);
  }
  return vals[0];
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class ReassociateReductions
    : public impl::ReassociateReductionsBase<ReassociateReductions> {
public:
  using ReassociateReductionsBase::ReassociateReductionsBase;

  void runOnOperation() final {
    auto module = getOperation();
    IRRewriter rewriter(module.getContext());
    SmallVector<scf::ForOp> forOps;
    module.walk([&](scf::ForOp op) { forOps.push_back(op); });
    for (auto forOp : forOps)
      processForOp(rewriter, forOp);
  }

private:
  void processForOp(IRRewriter &rewriter, scf::ForOp forOp) {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    for (unsigned idx = 0; idx < forOp.getNumRegionIterArgs(); ++idx) {
      BlockArgument iterArg = forOp.getRegionIterArg(idx);
      Value yieldVal = yieldOp.getOperand(idx);

      // ---- Extract the reduction chain ----
      SmallVector<Value> fresh;
      SmallVector<Operation *> toErase;
      std::optional<ReductionKind> kind;
      Value current = yieldVal;
      bool valid = true;

      while (valid) {
        auto m = matchReduction(current);
        if (!m) {
          valid = false;
          break;
        }
        if (!kind)
          kind = m->kind;
        else if (!(*kind == m->kind)) {
          valid = false;
          break;
        }

        // Identify chain vs. fresh operand.
        Value chain, freshVal;
        bool isBase = false;
        if (m->opA == iterArg) {
          chain = iterArg;
          freshVal = m->opB;
          isBase = true;
        } else if (m->opB == iterArg) {
          chain = iterArg;
          freshVal = m->opA;
          isBase = true;
        } else if (!dependsOnArg(m->opB, iterArg)) {
          chain = m->opA;
          freshVal = m->opB;
        } else if (!dependsOnArg(m->opA, iterArg)) {
          chain = m->opB;
          freshVal = m->opA;
        } else {
          valid = false;
          break;
        }

        fresh.push_back(freshVal);
        toErase.append(m->ops.begin(), m->ops.end());

        if (isBase)
          break;
        current = chain;
      }

      // Minimum chain length 3 (2-element tree = sequential, no benefit).
      if (!valid || fresh.size() < 3)
        continue;

      // fresh is in reverse order (yield -> base); flip to base -> yield.
      std::reverse(fresh.begin(), fresh.end());

      // ---- Build balanced tree + single acc-dependent reduction ----
      rewriter.setInsertionPoint(yieldOp);
      Location loc = forOp.getLoc();
      Value tree = buildBalancedTree(rewriter, loc, *kind, fresh);
      Value result = buildReduction(rewriter, loc, *kind, iterArg, tree);
      yieldOp.setOperand(idx, result);

      // Clean up dead chain ops.
      for (auto *op : toErase)
        if (op->use_empty())
          rewriter.eraseOp(op);
    }
  }
};

} // namespace mlir::transforms
