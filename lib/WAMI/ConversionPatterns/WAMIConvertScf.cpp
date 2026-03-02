//===- WAMIConvertScf.cpp - SCF to WasmSSA conversion ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion patterns from SCF dialect to the upstream
// WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/ConversionPatterns/WAMIConvertScf.h"
#include "WAMI/ConversionPatterns/WAMIConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"

namespace mlir::wami {

namespace {

//===----------------------------------------------------------------------===//
// IfOp Lowering
//===----------------------------------------------------------------------===//

struct IfOpLowering : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Convert condition to i32 (WasmSSA expects i32)
    Value condition = ensureI32Condition(adaptor.getCondition(), loc, rewriter);

    // Convert result types
    SmallVector<Type, 4> resultTypes;
    for (Type t : op.getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(t));
    }

    // Create a successor block for after the if
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // Add block arguments for results
    for (Type t : resultTypes) {
      afterBlock->addArgument(t, loc);
    }

    // Go back to the end of the current block to create the if
    rewriter.setInsertionPointToEnd(currentBlock);

    // Create the wasmssa.if operation with the target block
    auto ifOp = wasmssa::IfOp::create(rewriter, loc, condition,
                                      /*inputs=*/ValueRange{}, afterBlock);

    // Create and populate the 'then' region
    Block *thenBlock = ifOp.createIfBlock();
    rewriter.setInsertionPointToEnd(thenBlock);

    // Move operations from then region instead of cloning (avoids stale
    // references)
    Block *origThenBlock = &op.getThenRegion().front();

    // Get yielded values BEFORE moving operations
    SmallVector<Value, 4> thenYieldOperands;
    if (auto yieldOp = dyn_cast<scf::YieldOp>(origThenBlock->getTerminator())) {
      for (Value v : yieldOp.getOperands()) {
        thenYieldOperands.push_back(v);
      }
    }

    // Move operations from original then block
    for (auto &origOp :
         llvm::make_early_inc_range(origThenBlock->without_terminator())) {
      origOp.moveBefore(thenBlock, thenBlock->end());
    }

    // Erase the original terminator
    rewriter.eraseOp(origThenBlock->getTerminator());

    SmallVector<Value, 4> normalizedThenYieldOperands;
    if (failed(normalizeOperandsToTypes(thenYieldOperands, resultTypes, loc,
                                        rewriter, normalizedThenYieldOperands)))
      return failure();
    wasmssa::BlockReturnOp::create(rewriter, loc, normalizedThenYieldOperands);

    // Create and populate the 'else' region
    Block *elseBlock = ifOp.createElseBlock();
    rewriter.setInsertionPointToEnd(elseBlock);

    if (!op.getElseRegion().empty()) {
      Block *origElseBlock = &op.getElseRegion().front();

      // Get yielded values BEFORE moving operations
      SmallVector<Value, 4> elseYieldOperands;
      if (auto yieldOp =
              dyn_cast<scf::YieldOp>(origElseBlock->getTerminator())) {
        for (Value v : yieldOp.getOperands()) {
          elseYieldOperands.push_back(v);
        }
      }

      // Move operations from original else block
      for (auto &origOp :
           llvm::make_early_inc_range(origElseBlock->without_terminator())) {
        origOp.moveBefore(elseBlock, elseBlock->end());
      }

      // Erase the original terminator
      rewriter.eraseOp(origElseBlock->getTerminator());

      SmallVector<Value, 4> normalizedElseYieldOperands;
      if (failed(normalizeOperandsToTypes(elseYieldOperands, resultTypes, loc,
                                          rewriter,
                                          normalizedElseYieldOperands)))
        return failure();
      wasmssa::BlockReturnOp::create(rewriter, loc,
                                     normalizedElseYieldOperands);
    } else {
      wasmssa::BlockReturnOp::create(rewriter, loc, ValueRange{});
    }

    // Replace uses of the original if results with block arguments
    rewriter.replaceOp(op, afterBlock->getArguments());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ForOp Lowering
//===----------------------------------------------------------------------===//

/// Check if lb < ub can be proven for constant bounds, meaning the loop body
/// always executes at least once (no guard needed, tail-controlled is safe).
static bool hasPositiveTripCount(scf::ForOp forOp) {
  auto lbConst = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
  auto ubConst = forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
  if (!lbConst || !ubConst)
    return false;

  auto lbAttr = dyn_cast<IntegerAttr>(lbConst.getValue());
  auto ubAttr = dyn_cast<IntegerAttr>(ubConst.getValue());
  if (!lbAttr || !ubAttr)
    return false;

  return lbAttr.getValue().slt(ubAttr.getValue());
}

/// ForOp lowering with two strategies:
///
/// **Tail-controlled** (constant bounds with lb < ub):
///   block @exit
///     loop @loop
///       <body>
///       nextI = add i, step
///       continueCond = lt_si nextI, ub
///       br_if @loop                         // conditional back-edge (hot)
///       br @exit                            // unconditional exit (cold)
///     end
///   end
///   Binaryen folds br_if + br into a single br_if, eliminating 1 branch/iter.
///
/// **Header-controlled** (dynamic bounds):
///   block @exit
///     loop @loop
///       exitCond = ge_si i, ub
///       br_if @exit                         // conditional exit
///       <body>
///       nextI = add i, step
///       BlockReturn [nextI, yielded]        // br @loop (back-edge)
///     end
///   end
struct ForOpLowering : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (hasPositiveTripCount(op))
      return lowerTailControlled(op, adaptor, rewriter);
    return lowerHeaderControlled(op, adaptor, rewriter);
  }

private:
  /// Shared prologue: normalize bounds, collect init values and types,
  /// create the outer BlockOp and afterBlock.
  struct LoopPrologue {
    Value lowerBound, upperBound, step;
    SmallVector<Value, 4> normalizedInitValues;
    SmallVector<Type, 4> resultTypes;
    SmallVector<Type, 4> initTypes;
    Block *afterBlock;
    wasmssa::BlockOp blockOp;
    Block *blockEntry;
  };

  LogicalResult buildPrologue(scf::ForOp op, OpAdaptor &adaptor,
                              ConversionPatternRewriter &rewriter,
                              LoopPrologue &p) const {
    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();

    p.lowerBound = adaptor.getLowerBound();
    p.upperBound = adaptor.getUpperBound();
    p.step = adaptor.getStep();

    if (failed(normalizeToType(p.lowerBound, i32Type, loc, rewriter,
                               p.lowerBound)))
      return failure();
    if (failed(normalizeToType(p.upperBound, i32Type, loc, rewriter,
                               p.upperBound)))
      return failure();
    if (failed(normalizeToType(p.step, i32Type, loc, rewriter, p.step)))
      return failure();

    SmallVector<Value, 4> initValues;
    initValues.push_back(p.lowerBound);
    for (Value initArg : adaptor.getInitArgs())
      initValues.push_back(initArg);

    for (Type t : op.getResultTypes())
      p.resultTypes.push_back(getTypeConverter()->convertType(t));

    p.initTypes.push_back(i32Type);
    p.initTypes.append(p.resultTypes.begin(), p.resultTypes.end());

    if (failed(normalizeOperandsToTypes(initValues, p.initTypes, loc, rewriter,
                                        p.normalizedInitValues)))
      return failure();

    Block *currentBlock = rewriter.getInsertionBlock();
    p.afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    for (Type t : p.resultTypes)
      p.afterBlock->addArgument(t, loc);

    rewriter.setInsertionPointToEnd(currentBlock);
    p.blockOp = wasmssa::BlockOp::create(rewriter, loc, p.normalizedInitValues,
                                         p.afterBlock);
    p.blockEntry = p.blockOp.createBlock();
    rewriter.setInsertionPointToEnd(p.blockEntry);
    return success();
  }

  /// Move body ops from scf.for into loopBody, remap induction var and
  /// iter args, return the yielded values (normalized to resultTypes).
  LogicalResult
  moveBodyAndGetYielded(scf::ForOp op, ConversionPatternRewriter &rewriter,
                        Location loc, Block *loopBody, Value loopInductionVar,
                        ArrayRef<Value> loopIterArgs,
                        ArrayRef<Type> resultTypes,
                        SmallVector<Value, 4> &normalizedYielded) const {
    Block *origBody = op.getBody();
    rewriter.replaceAllUsesWith(op.getInductionVar(), loopInductionVar);
    for (auto [oldArg, newArg] :
         llvm::zip(op.getRegionIterArgs(), loopIterArgs))
      rewriter.replaceAllUsesWith(oldArg, newArg);

    SmallVector<Value, 4> yieldedValues;
    if (auto yieldOp = dyn_cast<scf::YieldOp>(origBody->getTerminator()))
      for (Value v : yieldOp.getOperands())
        yieldedValues.push_back(v);

    for (auto &bodyOp :
         llvm::make_early_inc_range(origBody->without_terminator()))
      bodyOp.moveBefore(loopBody, loopBody->end());
    rewriter.eraseOp(origBody->getTerminator());

    return normalizeOperandsToTypes(yieldedValues, resultTypes, loc, rewriter,
                                    normalizedYielded);
  }

  /// Tail-controlled: body first, then conditional back-edge + unconditional
  /// exit. Used when lb < ub is provable (no guard needed).
  LogicalResult lowerTailControlled(scf::ForOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    LoopPrologue p;
    if (failed(buildPrologue(op, adaptor, rewriter, p)))
      return failure();

    SmallVector<Value, 4> blockIterArgs;
    for (unsigned i = 1; i < p.blockEntry->getNumArguments(); ++i)
      blockIterArgs.push_back(p.blockEntry->getArgument(i));

    // Dummy exit block (unreachable — loop exits via wami.branch)
    Block *loopExitBlock = rewriter.createBlock(&p.blockOp.getBody());
    rewriter.setInsertionPointToEnd(loopExitBlock);
    wasmssa::BlockReturnOp::create(rewriter, loc, blockIterArgs);

    rewriter.setInsertionPointToEnd(p.blockEntry);
    auto loopOp = wasmssa::LoopOp::create(
        rewriter, loc, p.blockEntry->getArguments(), loopExitBlock);
    Block *loopBody = loopOp.createBlock();
    rewriter.setInsertionPointToEnd(loopBody);

    Value loopInductionVar = loopBody->getArgument(0);
    SmallVector<Value, 4> loopIterArgs;
    for (unsigned i = 1; i < loopBody->getNumArguments(); ++i)
      loopIterArgs.push_back(loopBody->getArgument(i));

    // Body FIRST (before exit check)
    SmallVector<Value, 4> normalizedYielded;
    if (failed(moveBodyAndGetYielded(op, rewriter, loc, loopBody,
                                     loopInductionVar, loopIterArgs,
                                     p.resultTypes, normalizedYielded)))
      return failure();

    // nextI = add i, step
    Value nextInduction =
        wasmssa::AddOp::create(rewriter, loc, loopInductionVar, p.step);

    // continueCond = lt_si nextI, ub
    Value continueCond =
        wasmssa::LtSIOp::create(rewriter, loc, nextInduction, p.upperBound);

    SmallVector<Value, 4> continueValues;
    continueValues.push_back(nextInduction);
    for (Value v : normalizedYielded)
      continueValues.push_back(v);

    // br_if @loop (exitLevel=0) — conditional back-edge
    Block *exitBlock = rewriter.createBlock(&loopOp.getBody());
    rewriter.setInsertionPointToEnd(loopBody);
    wasmssa::BranchIfOp::create(rewriter, loc, continueCond,
                                /*exitLevel=*/0, continueValues, exitBlock);

    // Unconditional exit via always-true BranchIfOp to level 1 (@exit block).
    // Binaryen folds br_if(const 1) into br, then merges br_if @loop + br @exit
    // into a single br_if @loop.
    rewriter.setInsertionPointToEnd(exitBlock);
    Value constTrue =
        wasmssa::ConstOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
    Block *dummyBlock = rewriter.createBlock(&loopOp.getBody());
    rewriter.setInsertionPointToEnd(exitBlock);
    wasmssa::BranchIfOp::create(rewriter, loc, constTrue,
                                /*exitLevel=*/1, normalizedYielded, dummyBlock);

    // Dummy block (unreachable — BranchIfOp above always taken)
    rewriter.setInsertionPointToEnd(dummyBlock);
    wasmssa::BlockReturnOp::create(
        rewriter, loc, SmallVector<Value>(loopBody->getArguments()));

    rewriter.replaceOp(op, p.afterBlock->getArguments());
    return success();
  }

  /// Header-controlled: exit check first, then body + unconditional back-edge.
  /// Used when bounds are dynamic (guard-free, always correct).
  LogicalResult
  lowerHeaderControlled(scf::ForOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    LoopPrologue p;
    if (failed(buildPrologue(op, adaptor, rewriter, p)))
      return failure();

    SmallVector<Value, 4> blockIterArgs;
    for (unsigned i = 1; i < p.blockEntry->getNumArguments(); ++i)
      blockIterArgs.push_back(p.blockEntry->getArgument(i));

    // Dummy exit block for LoopOp (unreachable — loop exits via br_if @exit).
    // Passes only iter_args (not induction var) since BlockOp results =
    // resultTypes.
    Block *loopExitBlock = rewriter.createBlock(&p.blockOp.getBody());
    rewriter.setInsertionPointToEnd(loopExitBlock);
    wasmssa::BlockReturnOp::create(rewriter, loc, blockIterArgs);

    rewriter.setInsertionPointToEnd(p.blockEntry);
    auto loopOp = wasmssa::LoopOp::create(
        rewriter, loc, p.blockEntry->getArguments(), loopExitBlock);
    Block *loopBody = loopOp.createBlock();
    rewriter.setInsertionPointToEnd(loopBody);

    Value loopInductionVar = loopBody->getArgument(0);
    SmallVector<Value, 4> loopIterArgs;
    for (unsigned i = 1; i < loopBody->getNumArguments(); ++i)
      loopIterArgs.push_back(loopBody->getArgument(i));

    // Exit check FIRST: ge_si i, ub → br_if @exit
    Value exitCond =
        wasmssa::GeSIOp::create(rewriter, loc, loopInductionVar, p.upperBound);
    Block *continueBlock = rewriter.createBlock(&loopOp.getBody());
    rewriter.setInsertionPointToEnd(loopBody);
    wasmssa::BranchIfOp::create(rewriter, loc, exitCond,
                                /*exitLevel=*/1, blockIterArgs, continueBlock);

    // Body in continueBlock
    rewriter.setInsertionPointToEnd(continueBlock);
    SmallVector<Value, 4> normalizedYielded;
    if (failed(moveBodyAndGetYielded(op, rewriter, loc, continueBlock,
                                     loopInductionVar, loopIterArgs,
                                     p.resultTypes, normalizedYielded)))
      return failure();

    // nextI = add i, step
    Value nextInduction =
        wasmssa::AddOp::create(rewriter, loc, loopInductionVar, p.step);

    // Back-edge: BlockReturnOp [nextI, yielded] → br @loop
    SmallVector<Value, 4> backEdgeValues;
    backEdgeValues.push_back(nextInduction);
    for (Value v : normalizedYielded)
      backEdgeValues.push_back(v);
    wasmssa::BlockReturnOp::create(rewriter, loc, backEdgeValues);

    rewriter.replaceOp(op, p.afterBlock->getArguments());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WhileOp Lowering
//===----------------------------------------------------------------------===//

struct WhileOpLowering : public OpConversionPattern<scf::WhileOp> {
  using OpConversionPattern<scf::WhileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get initial values
    SmallVector<Value, 4> initValues(adaptor.getInits());

    SmallVector<Type, 4> beforeTypes;
    for (Type t : op.getBefore().getArgumentTypes()) {
      beforeTypes.push_back(getTypeConverter()->convertType(t));
    }

    // Convert result types
    SmallVector<Type, 4> resultTypes;
    for (Type t : op.getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(t));
    }

    SmallVector<Value, 4> normalizedInitValues;
    if (failed(normalizeOperandsToTypes(initValues, beforeTypes, loc, rewriter,
                                        normalizedInitValues)))
      return failure();

    // Create successor block for after the loop
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    for (Type t : resultTypes) {
      afterBlock->addArgument(t, loc);
    }

    // Go back to insert the block op
    rewriter.setInsertionPointToEnd(currentBlock);

    // Create outer block
    auto blockOp = wasmssa::BlockOp::create(rewriter, loc, normalizedInitValues,
                                            afterBlock);

    // Create block entry
    Block *blockEntry = blockOp.createBlock();
    rewriter.setInsertionPointToEnd(blockEntry);

    // Create a dummy exit block within the outer block
    Block *loopExitBlock = rewriter.createBlock(&blockOp.getBody());
    rewriter.setInsertionPointToEnd(loopExitBlock);
    wasmssa::BlockReturnOp::create(
        rewriter, loc, SmallVector<Value>(blockEntry->getArguments()));

    // Go back to block entry
    rewriter.setInsertionPointToEnd(blockEntry);

    // Create inner loop
    auto loopOp = wasmssa::LoopOp::create(
        rewriter, loc, blockEntry->getArguments(), loopExitBlock);

    // Create loop body
    Block *loopBody = loopOp.createBlock();
    rewriter.setInsertionPointToEnd(loopBody);

    // Map before region arguments
    IRMapping beforeMapping;
    for (auto [oldArg, newArg] :
         llvm::zip(op.getBefore().getArguments(), loopBody->getArguments())) {
      beforeMapping.map(oldArg, newArg);
    }

    // Clone before region (condition check)
    for (auto &beforeOp : op.getBefore().front().without_terminator()) {
      rewriter.clone(beforeOp, beforeMapping);
    }

    // Handle scf.condition
    auto conditionOp =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    Value cond = beforeMapping.lookupOrDefault(conditionOp.getCondition());
    cond = ensureI32Condition(cond, loc, rewriter);

    // Get values to pass to after region
    SmallVector<Value, 4> condArgs;
    for (Value v : conditionOp.getArgs()) {
      condArgs.push_back(beforeMapping.lookupOrDefault(v));
    }

    SmallVector<Value, 4> normalizedCondArgs;
    if (failed(normalizeOperandsToTypes(condArgs, resultTypes, loc, rewriter,
                                        normalizedCondArgs)))
      return failure();

    // Create continuation block for body
    Block *bodyBlock = rewriter.createBlock(&loopOp.getBody());

    // branch_if (not cond) to level 1 (exit) with condArgs, else continue to
    // body
    rewriter.setInsertionPointToEnd(loopBody);
    Value notCond = wasmssa::EqzOp::create(rewriter, loc, cond);
    wasmssa::BranchIfOp::create(rewriter, loc, notCond,
                                /*exitLevel=*/1, normalizedCondArgs, bodyBlock);

    // In body block, execute after region
    rewriter.setInsertionPointToEnd(bodyBlock);

    // Map after region arguments to condArgs
    IRMapping afterMapping;
    for (auto [oldArg, newArg] :
         llvm::zip(op.getAfter().getArguments(), normalizedCondArgs)) {
      afterMapping.map(oldArg, newArg);
    }

    // Clone after region (loop body)
    for (auto &afterOp : op.getAfter().front().without_terminator()) {
      rewriter.clone(afterOp, afterMapping);
    }

    // Handle scf.yield
    auto yieldOp = cast<scf::YieldOp>(op.getAfter().front().getTerminator());
    SmallVector<Value, 4> yieldValues;
    for (Value v : yieldOp.getOperands()) {
      yieldValues.push_back(afterMapping.lookupOrDefault(v));
    }

    SmallVector<Value, 4> normalizedYieldValues;
    if (failed(normalizeOperandsToTypes(yieldValues, beforeTypes, loc, rewriter,
                                        normalizedYieldValues)))
      return failure();

    // Continue loop
    wasmssa::BlockReturnOp::create(rewriter, loc, normalizedYieldValues);

    // Replace the original op
    rewriter.replaceOp(op, afterBlock->getArguments());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// YieldOp Lowering (handled within parent op patterns)
//===----------------------------------------------------------------------===//

struct YieldOpLowering : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Yield is handled by parent op patterns
    // This pattern handles any remaining yields
    rewriter.replaceOpWithNewOp<wasmssa::BlockReturnOp>(op,
                                                        adaptor.getOperands());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateWAMIConvertScfPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<IfOpLowering, ForOpLowering, WhileOpLowering, YieldOpLowering>(
      typeConverter, context);
}

} // namespace mlir::wami
