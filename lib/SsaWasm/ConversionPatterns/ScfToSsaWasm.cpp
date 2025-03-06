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

struct ForOpLowering : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto lowerBound = adaptor.getLowerBound();
    auto upperBound = adaptor.getUpperBound();
    ValueRange initArgs = adaptor.getInitArgs();
    auto step = adaptor.getStep();

    auto blockLoopOp = rewriter.create<BlockLoopOp>(loc);
    auto [entryBlock, loopStartLabel, blockEndLabel] =
        blockLoopOp.initialize(rewriter);

    Block *conditionBlock = loopStartLabel;
    Block *inductionVariableUpdateBlock =
        rewriter.createBlock(&blockLoopOp.getRegion());

    assert(op.getRegion().getBlocks().size() == 1 &&
           "Only support scf.for op with one block");
    Block *bodyBlock = &op.getRegion().front();

    rewriter.inlineRegionBefore(op.getRegion(), inductionVariableUpdateBlock);

    auto bodyArgs = bodyBlock->getArguments();

    // create locals for iteration variables before the BlockLoopOp
    rewriter.setInsertionPoint(blockLoopOp);
    SmallVector<Value, 4> iterationLocals;
    for (Value initArg : initArgs) {
      auto local = rewriter.create<LocalOp>(loc, initArg).getResult();
      auto castedLocal =
          rewriter
              .create<UnrealizedConversionCastOp>(loc, initArg.getType(), local)
              .getResult(0);
      iterationLocals.push_back(castedLocal);
    }
    // create local for induction variable
    auto inductionLocal = rewriter.create<LocalOp>(loc, lowerBound).getResult();
    auto castedInductionLocal =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, rewriter.getI32Type(),
                                                inductionLocal)
            .getResult(0);

    // replace uses of iteration variables and induction variable with locals
    rewriter.replaceAllUsesWith(bodyArgs[0], castedInductionLocal);
    for (unsigned i = 0; i < initArgs.size(); ++i) {
      rewriter.replaceAllUsesWith(bodyArgs[i + 1], iterationLocals[i]);
    }

    // Condition Block: evaluate loop condition
    rewriter.setInsertionPointToStart(conditionBlock);
    auto comparisonOp =
        rewriter.create<ILeUOp>(loc, upperBound, inductionLocal);

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
    auto addOp = rewriter.create<AddOp>(loc, inductionLocal, step);
    rewriter.create<LocalSetOp>(loc, inductionLocal, addOp.getResult());
    rewriter.create<BlockLoopBranchOp>(loc, loopStartLabel);

    // Replace scf.for op with final iteration variable values
    rewriter.replaceOp(op, iterationLocals);

    // Remove block args
    TypeConverter::SignatureConversion signatureConversion(
        bodyBlock->getNumArguments());
    rewriter.applySignatureConversion(bodyBlock, signatureConversion);

    return success();
  }
};

struct IfElseOpLowering : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto condition = adaptor.getCondition();
    auto results = op.getResults();

    // Create the if-else operation
    SmallVector<Type, 4> convertedTypes;
    if (failed(getTypeConverter()->convertTypes(results.getTypes(),
                                                convertedTypes))) {
      emitError(loc, "Failed to convert types");
      return failure();
    }
    auto ifElseOp = rewriter.create<IfElseOp>(loc, convertedTypes, condition);

    // Move the then body
    rewriter.inlineRegionBefore(op.getThenRegion(), ifElseOp.getThenRegion(),
                                ifElseOp.getThenRegion().end());

    // Move the else body if it exists
    if (!op.getElseRegion().empty()) {
      rewriter.inlineRegionBefore(op.getElseRegion(), ifElseOp.getElseRegion(),
                                  ifElseOp.getElseRegion().end());
    } else {
      // create an empty block
      Block *elseBlock = rewriter.createBlock(&ifElseOp.getElseRegion());
      rewriter.setInsertionPointToEnd(elseBlock);
      rewriter.create<IfElseTerminatorOp>(loc, ValueRange());
    }

    // apply signature conversion to the then and else regions
    TypeConverter::SignatureConversion thenSignatureConversion(
        ifElseOp.getThenRegion().front().getNumArguments());
    TypeConverter::SignatureConversion elseSignatureConversion(
        ifElseOp.getElseRegion().front().getNumArguments());
    rewriter.applySignatureConversion(&ifElseOp.getThenRegion().front(),
                                      thenSignatureConversion);
    rewriter.applySignatureConversion(&ifElseOp.getElseRegion().front(),
                                      elseSignatureConversion);

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
  patterns.add<ForOpLowering, IfElseOpLowering>(typeConverter,
                                                patterns.getContext());
}

} // namespace mlir::ssawasm