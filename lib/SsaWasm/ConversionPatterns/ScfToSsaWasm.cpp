#include "SsaWasm/ConversionPatterns/ScfToSsaWasm.h"
#include "SsaWasm/SsaWasmOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

namespace mlir::ssawasm {

// +----------------------------------------+
// | entrypointBlock                        |
// |   <initialize induction variable>      |
// |   br ^conditionBlock                   |
// +----------------------------------------+
//                 |
//                 v
// +----------------------------------------+
// | conditionBlock                         |
// |   <evaluate loop condition>            |
// |   cond_br %cond, ^exit                 |
// +----------------------------------------+
//         |
//         | True
//         v
// +----------------------------------------+
// | bodyBlock                              |
// |   <loop body begins>                   |
// |                                        |
// |   <loop body ends>                     |
// +----------------------------------------+
//                 |
//                 v
// +----------------------------------------+
// | inductionVariableUpdateBlock           |
// |   <update induction variable>          |
// |   br ^conditionBlock                   |
// +----------------------------------------+
//                 |
//                 v
//          (Back to conditionBlock)

struct ForOpLowering : public RewritePattern {
  TypeConverter &typeConverter;

  ForOpLowering(TypeConverter &typeConverter, MLIRContext *context)
      : RewritePattern(scf::ForOp::getOperationName(), 1, context),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);
    Location loc = forOp.getLoc();

    auto lowerBound = forOp.getLowerBound();
    auto upperBound = forOp.getUpperBound();
    ValueRange initArgs = forOp.getInitArgs();
    auto step = forOp.getStep();

    // since we are not using DialectConversion, we need to manually cast the
    // types
    auto castedLowerBound =
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, typeConverter.convertType(lowerBound.getType()),
                lowerBound)
            .getResult(0);
    auto castedUpperBound =
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, typeConverter.convertType(upperBound.getType()),
                upperBound)
            .getResult(0);
    auto castedStep =
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, typeConverter.convertType(step.getType()), step)
            .getResult(0);

    auto blockLoopOp = rewriter.create<BlockLoopOp>(loc);
    auto [entryBlock, loopStartLabel, blockEndLabel] =
        blockLoopOp.initialize(rewriter);

    Block *conditionBlock = loopStartLabel;
    Block *inductionVariableUpdateBlock =
        rewriter.createBlock(&blockLoopOp.getRegion());

    assert(forOp.getRegion().getBlocks().size() == 1 &&
           "Only support scf.for op with one block");
    Block *bodyBlock = &forOp.getRegion().front();

    rewriter.inlineRegionBefore(forOp.getRegion(),
                                inductionVariableUpdateBlock);

    auto bodyArgs = bodyBlock->getArguments();

    // create locals for iteration variables before the BlockLoopOp
    rewriter.setInsertionPoint(blockLoopOp);
    SmallVector<Value, 4> iterationLocals;
    for (Value initArg : initArgs) {
      auto castedInitArg =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, typeConverter.convertType(initArg.getType()), initArg)
              .getResult(0);
      auto local = rewriter.create<LocalOp>(loc, castedInitArg).getResult();
      auto castedLocal =
          rewriter
              .create<UnrealizedConversionCastOp>(loc, initArg.getType(), local)
              .getResult(0);
      iterationLocals.push_back(castedLocal);
    }
    // create local for induction variable
    auto inductionLocal =
        rewriter.create<LocalOp>(loc, castedLowerBound).getResult();
    auto castedInductionLocal =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, lowerBound.getType(),
                                                inductionLocal)
            .getResult(0);

    // replace uses of iteration variables and induction variable with
    // locals
    rewriter.replaceAllUsesWith(bodyArgs[0], castedInductionLocal);
    for (unsigned i = 0; i < initArgs.size(); ++i) {
      rewriter.replaceAllUsesWith(bodyArgs[i + 1], iterationLocals[i]);
    }

    // Condition Block: evaluate loop condition
    rewriter.setInsertionPointToStart(conditionBlock);
    auto comparisonOp =
        rewriter.create<ILeUOp>(loc, castedUpperBound, inductionLocal);

    rewriter.create<BlockLoopCondBranchOp>(loc, comparisonOp.getResult(),
                                           blockEndLabel, bodyBlock);

    // Body End Block: update induction variable and branch to
    // inductionVariableUpdateBlock
    rewriter.setInsertionPointToEnd(bodyBlock);
    // get the old terminator
    Operation *terminator = bodyBlock->getTerminator();
    if (isa<scf::YieldOp>(terminator)) {
      // store the results of the yield op to iteration locals using LocalSetOp
      for (unsigned i = 0; i < initArgs.size(); ++i) {
        auto castedIterationLocal =
            rewriter
                .create<UnrealizedConversionCastOp>(
                    loc, iterationLocals[i].getType(), iterationLocals[i])
                .getResult(0);
        auto castedResult = rewriter
                                .create<UnrealizedConversionCastOp>(
                                    loc, iterationLocals[i].getType(),
                                    terminator->getOperand(i))
                                .getResult(0);
        rewriter.create<LocalSetOp>(loc, castedIterationLocal, castedResult);
      }
    }
    rewriter.eraseOp(terminator);
    rewriter.create<TempBranchOp>(loc, inductionVariableUpdateBlock);

    // Induction Variable Update Block
    rewriter.setInsertionPointToStart(inductionVariableUpdateBlock);
    auto addOp = rewriter.create<AddOp>(loc, inductionLocal, castedStep);
    rewriter.create<LocalSetOp>(loc, inductionLocal, addOp.getResult());
    rewriter.create<BlockLoopBranchOp>(loc, loopStartLabel);

    // Replace scf.for op with final iteration variable values
    rewriter.replaceOp(op, iterationLocals);

    // Remove block args
    while (bodyBlock->getNumArguments() > 0) {
      bodyBlock->eraseArgument(0);
    }

    return success();
  }
};

struct IfOpLowering : public RewritePattern {
  TypeConverter &typeConverter;

  IfOpLowering(TypeConverter &typeConverter, MLIRContext *context)
      : RewritePattern(scf::IfOp::getOperationName(), 1, context),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ifOp = cast<scf::IfOp>(op);

    Location loc = ifOp.getLoc();
    auto condition = ifOp.getCondition();
    auto results = ifOp.getResults();

    // Create the if-else operation
    SmallVector<Type, 4> convertedTypes;
    if (failed(
            typeConverter.convertTypes(results.getTypes(), convertedTypes))) {
      emitError(loc, "Failed to convert types");
      return failure();
    }
    auto castedCondition =
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, typeConverter.convertType(condition.getType()), condition)
            .getResult(0);

    auto ifElseOp =
        rewriter.create<IfElseOp>(loc, convertedTypes, castedCondition);

    // Move the then body
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), ifElseOp.getThenRegion(),
                                ifElseOp.getThenRegion().end());

    // Move the else body if it exists
    if (!ifOp.getElseRegion().empty()) {
      rewriter.inlineRegionBefore(ifOp.getElseRegion(),
                                  ifElseOp.getElseRegion(),
                                  ifElseOp.getElseRegion().end());
    } else {
      // create an empty block
      Block *elseBlock = rewriter.createBlock(&ifElseOp.getElseRegion());
      rewriter.setInsertionPointToEnd(elseBlock);
      rewriter.create<IfElseTerminatorOp>(loc, ValueRange());
    }

    for (auto region : ifElseOp.getRegions()) {
      while (region->front().getNumArguments() > 0) {
        region->front().eraseArgument(0);
      }
    }

    // Convert scf.yield in then block to IfElseTerminatorOp
    if (auto yieldOp = dyn_cast<scf::YieldOp>(
            ifElseOp.getThenRegion().front().getTerminator())) {
      rewriter.setInsertionPoint(yieldOp);
      if (convertedTypes.size() > 0) {
        auto convertedResults =
            rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedTypes,
                                                    yieldOp.getResults())
                .getResults();
        rewriter.replaceOpWithNewOp<IfElseTerminatorOp>(yieldOp,
                                                        convertedResults);
      } else {
        rewriter.replaceOpWithNewOp<IfElseTerminatorOp>(yieldOp,
                                                        yieldOp.getResults());
      }
    }

    // Convert scf.yield in else block to IfElseTerminatorOp
    if (auto yieldOp = dyn_cast<scf::YieldOp>(
            ifElseOp.getElseRegion().front().getTerminator())) {
      rewriter.setInsertionPoint(yieldOp);
      if (convertedTypes.size() > 0) {
        auto convertedResults =
            rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedTypes,
                                                    yieldOp.getResults())
                .getResults();
        rewriter.replaceOpWithNewOp<IfElseTerminatorOp>(yieldOp,
                                                        convertedResults);
      } else {
        rewriter.replaceOpWithNewOp<IfElseTerminatorOp>(yieldOp,
                                                        yieldOp.getResults());
      }
    }

    rewriter.replaceOp(op, ifElseOp.getResults());
    return success();
  }
};

void populateScfToSsaWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.add<ForOpLowering, IfOpLowering>(typeConverter,
                                            patterns.getContext());
}

} // namespace mlir::ssawasm