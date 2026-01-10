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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"

namespace mlir::wami {

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Convert an i1 condition to i32 for WasmSSA (which expects i32 conditions).
static Value convertConditionToI32(Value cond, Location loc,
                                   ConversionPatternRewriter &rewriter) {
  if (cond.getType().isInteger(1)) {
    return arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), cond);
  }
  return cond;
}

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
    Value condition =
        convertConditionToI32(adaptor.getCondition(), loc, rewriter);

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

    // Clone the then region using IRMapping
    IRMapping thenMapping;
    for (auto &origOp : op.getThenRegion().front().without_terminator()) {
      rewriter.clone(origOp, thenMapping);
    }

    // Handle scf.yield -> wasmssa.block_return
    if (auto yieldOp = dyn_cast<scf::YieldOp>(
            op.getThenRegion().front().getTerminator())) {
      SmallVector<Value, 4> yieldOperands;
      for (Value v : yieldOp.getOperands()) {
        yieldOperands.push_back(thenMapping.lookupOrDefault(v));
      }
      wasmssa::BlockReturnOp::create(rewriter, loc, yieldOperands);
    } else {
      wasmssa::BlockReturnOp::create(rewriter, loc, ValueRange{});
    }

    // Create and populate the 'else' region
    Block *elseBlock = ifOp.createElseBlock();
    rewriter.setInsertionPointToEnd(elseBlock);

    if (!op.getElseRegion().empty()) {
      IRMapping elseMapping;
      for (auto &origOp : op.getElseRegion().front().without_terminator()) {
        rewriter.clone(origOp, elseMapping);
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(
              op.getElseRegion().front().getTerminator())) {
        SmallVector<Value, 4> yieldOperands;
        for (Value v : yieldOp.getOperands()) {
          yieldOperands.push_back(elseMapping.lookupOrDefault(v));
        }
        wasmssa::BlockReturnOp::create(rewriter, loc, yieldOperands);
      } else {
        wasmssa::BlockReturnOp::create(rewriter, loc, ValueRange{});
      }
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

struct ForOpLowering : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get loop bounds and step - convert to i32 if needed
    Value lowerBound = adaptor.getLowerBound();
    Value upperBound = adaptor.getUpperBound();
    Value step = adaptor.getStep();

    // Ensure bounds are i32
    if (!lowerBound.getType().isInteger(32)) {
      lowerBound = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI32Type(), lowerBound);
    }
    if (!upperBound.getType().isInteger(32)) {
      upperBound = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI32Type(), upperBound);
    }
    if (!step.getType().isInteger(32)) {
      step = arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(),
                                        step);
    }

    // Collect initial values: induction variable + iter_args
    SmallVector<Value, 4> initValues;
    initValues.push_back(lowerBound);
    for (Value initArg : adaptor.getInitArgs()) {
      initValues.push_back(initArg);
    }

    // Convert result types
    SmallVector<Type, 4> resultTypes;
    for (Type t : op.getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(t));
    }

    // Create successor block for after the loop
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    for (Type t : resultTypes) {
      afterBlock->addArgument(t, loc);
    }

    // Go back to insert the block op
    rewriter.setInsertionPointToEnd(currentBlock);

    // Create outer block for break-out
    auto blockOp =
        wasmssa::BlockOp::create(rewriter, loc, initValues, afterBlock);

    // Create block entry
    Block *blockEntry = blockOp.createBlock();
    rewriter.setInsertionPointToEnd(blockEntry);

    // Block arguments: induction var + iter_args
    Value inductionVar = blockEntry->getArgument(0);
    SmallVector<Value, 4> blockIterArgs;
    for (unsigned i = 1; i < blockEntry->getNumArguments(); ++i) {
      blockIterArgs.push_back(blockEntry->getArgument(i));
    }

    // Create a dummy exit block within the outer block (needed as loop target)
    Block *loopExitBlock = rewriter.createBlock(&blockOp.getBody());
    rewriter.setInsertionPointToEnd(loopExitBlock);
    wasmssa::BlockReturnOp::create(rewriter, loc, blockIterArgs);

    // Go back to block entry to create the loop
    rewriter.setInsertionPointToEnd(blockEntry);

    // Create inner loop
    auto loopOp = wasmssa::LoopOp::create(
        rewriter, loc, blockEntry->getArguments(), loopExitBlock);

    // Create loop body
    Block *loopBody = loopOp.createBlock();
    rewriter.setInsertionPointToEnd(loopBody);

    // Loop arguments
    Value loopInductionVar = loopBody->getArgument(0);
    SmallVector<Value, 4> loopIterArgs;
    for (unsigned i = 1; i < loopBody->getNumArguments(); ++i) {
      loopIterArgs.push_back(loopBody->getArgument(i));
    }

    // Check exit condition: induction >= upper_bound
    Value exitCond =
        wasmssa::GeSIOp::create(rewriter, loc, loopInductionVar, upperBound);

    // Create continuation block within loop for the body
    Block *continueBlock = rewriter.createBlock(&loopOp.getBody());

    // branch_if exitCond to level 1 (outer block) with iter_args, else continue
    rewriter.setInsertionPointToEnd(loopBody);
    wasmssa::BranchIfOp::create(rewriter, loc, exitCond,
                                /*exitLevel=*/1, loopIterArgs, continueBlock);

    // In continue block, execute body and loop back
    rewriter.setInsertionPointToEnd(continueBlock);

    // Clone the loop body operations
    IRMapping mapping;
    mapping.map(op.getInductionVar(), loopInductionVar);
    for (auto [oldArg, newArg] :
         llvm::zip(op.getRegionIterArgs(), loopIterArgs)) {
      mapping.map(oldArg, newArg);
    }

    for (auto &bodyOp : op.getBody()->without_terminator()) {
      rewriter.clone(bodyOp, mapping);
    }

    // Get yielded values and update induction variable
    SmallVector<Value, 4> yieldedValues;
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op.getBody()->getTerminator())) {
      for (Value v : yieldOp.getOperands()) {
        yieldedValues.push_back(mapping.lookupOrDefault(v));
      }
    }

    // Update induction variable
    Value nextInduction =
        wasmssa::AddOp::create(rewriter, loc, loopInductionVar, step);

    // Continue loop with updated values
    SmallVector<Value, 4> continueValues;
    continueValues.push_back(nextInduction);
    for (Value v : yieldedValues) {
      continueValues.push_back(v);
    }
    wasmssa::BlockReturnOp::create(rewriter, loc, continueValues);

    // Replace the original op with results
    rewriter.replaceOp(op, afterBlock->getArguments());
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

    // Convert result types
    SmallVector<Type, 4> resultTypes;
    for (Type t : op.getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(t));
    }

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
    auto blockOp =
        wasmssa::BlockOp::create(rewriter, loc, initValues, afterBlock);

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
    cond = convertConditionToI32(cond, loc, rewriter);

    // Get values to pass to after region
    SmallVector<Value, 4> condArgs;
    for (Value v : conditionOp.getArgs()) {
      condArgs.push_back(beforeMapping.lookupOrDefault(v));
    }

    // Create continuation block for body
    Block *bodyBlock = rewriter.createBlock(&loopOp.getBody());

    // branch_if (not cond) to level 1 (exit) with condArgs, else continue to
    // body
    rewriter.setInsertionPointToEnd(loopBody);
    Value notCond = wasmssa::EqzOp::create(rewriter, loc, cond);
    wasmssa::BranchIfOp::create(rewriter, loc, notCond,
                                /*exitLevel=*/1, condArgs, bodyBlock);

    // In body block, execute after region
    rewriter.setInsertionPointToEnd(bodyBlock);

    // Map after region arguments to condArgs
    IRMapping afterMapping;
    for (auto [oldArg, newArg] :
         llvm::zip(op.getAfter().getArguments(), condArgs)) {
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

    // Continue loop
    wasmssa::BlockReturnOp::create(rewriter, loc, yieldValues);

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
