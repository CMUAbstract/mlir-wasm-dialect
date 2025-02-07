#include "Wasm/ConversionPatterns/ScfToWasmPatterns.h"
#include "Wasm/WasmOps.h"
#include "llvm/Support/Debug.h"

namespace mlir::wasm {

// +----------------------------------------+
// | entrypointBlock                        |
// |   <initialize induction variable>      |
// |   cf.br ^conditionBlock                |
// +----------------------------------------+
//                 |
//                 v
// +----------------------------------------+
// | conditionBlock                         |
// |   <evaluate loop condition>            |
// |   cf.cond_br %cond, ^bodyStartBlock, ^terminationBlock |
// +----------------------------------------+
//         |                                         |
//         | True                                    | False
//         v                                         v
// +----------------------------------------+    +---------------------------+
// | bodyStartBlock                         |    | terminationBlock          |
// |   <loop body begins>                   |    |   <code after the loop>   |
// |                                        |    +---------------------------+
// |   ... (possible intermediate blocks)   |
// |                                        |
// | bodyEndBlock                           |
// |   <loop body ends>                     |
// +----------------------------------------+
//                 |
//                 v
// +----------------------------------------+
// | inductionVariableUpdateBlock           |
// |   <update induction variable>          |
// |   cf.br ^conditionBlock                |
// +----------------------------------------+
//                 |
//                 v
//          (Back to conditionBlock)

struct ForOpLowering : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = forOp.getLoc();

    // Use i32 type for induction variable, lower bound, upper bound, and step
    auto indexType = rewriter.getI32Type();
    auto localIndexType = typeConverter->convertType(rewriter.getI32Type());

    auto inductionVariable = forOp.getInductionVar();
    Value lowerBound = adaptor.getLowerBound();
    Value upperBound = adaptor.getUpperBound();
    Value step = adaptor.getStep();
    if (!lowerBound || !upperBound || !inductionVariable) {
      return rewriter.notifyMatchFailure(forOp, "missing loop bounds");
    }

    // Get the iteration arguments (initial values of iteration variables)
    ValueRange initArgs = adaptor.getInitArgs();

    auto loopOp = rewriter.create<BlockLoopOpDeprecated>(loc);
    auto *entryBlock = rewriter.createBlock(&loopOp.getRegion());
    auto *conditionBlock = rewriter.createBlock(&loopOp.getRegion());
    auto *inductionVariableUpdateBlock =
        rewriter.createBlock(&loopOp.getRegion());
    auto *terminationBlock = rewriter.createBlock(&loopOp.getRegion());

    // Get the body block and its arguments
    // Move for loop body blocks between conditionBlock and
    // inductionVariableUpdateBlock
    rewriter.inlineRegionBefore(forOp.getRegion(),
                                inductionVariableUpdateBlock);

    // After inlining, get the updated body blocks and their arguments
    auto it = loopOp.getRegion().begin();
    it++; // Skip the entry block
    it++; // skip condition block
    auto *bodyStartBlock = &*it;
    auto *bodyEndBlock = inductionVariableUpdateBlock->getPrevNode();
    auto bodyArgs = bodyStartBlock->getArguments();

    // set variables for the iteration
    rewriter.setInsertionPoint(loopOp);
    SmallVector<Value, 4> iterationLocals;
    for (Value initArg : initArgs) {
      auto innerType = cast<LocalType>(initArg.getType()).getInner();
      auto local = rewriter.create<TempLocalOp>(loc, innerType).getResult();
      iterationLocals.push_back(local);
      rewriter.create<TempLocalGetOp>(loc, initArg);
      rewriter.create<TempLocalSetOp>(loc, local);
    }

    // Modify entry block
    rewriter.setInsertionPointToEnd(entryBlock);

    auto inductionLocal =
        rewriter.create<TempLocalOp>(loc, indexType).getResult();
    auto castedLowerBound = typeConverter->materializeSourceConversion(
        rewriter, loc, localIndexType, lowerBound);
    rewriter.create<TempLocalGetOp>(loc, castedLowerBound);
    rewriter.create<TempLocalSetOp>(loc, inductionLocal);

    // Replace uses of the block arguments with the locals
    auto castedInductionLocal = typeConverter->materializeSourceConversion(
        rewriter, loc, indexType, inductionLocal);
    rewriter.replaceAllUsesWith(bodyArgs[0], castedInductionLocal);
    for (unsigned i = 0; i < initArgs.size(); ++i) {
      auto castedLocal = typeConverter->materializeSourceConversion(
          rewriter, loc,
          cast<LocalType>(iterationLocals[i].getType()).getInner(),
          initArgs[i]);
      rewriter.replaceAllUsesWith(bodyArgs[i + 1], castedLocal);
    }

    rewriter.create<BranchOpDeprecated>(loc, conditionBlock);

    // Body end block: update iteration variables and branch to
    // inductionVariableUpdateBlock
    rewriter.setInsertionPointToEnd(bodyEndBlock);
    Operation *terminator = bodyEndBlock->getTerminator();

    if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
      auto yieldedValues = yieldOp.getResults();

      for (unsigned i = 0; i < iterationLocals.size(); ++i) {
        Value yieldedValue = yieldedValues[i];
        Value castedYieldedValue = typeConverter->materializeSourceConversion(
            rewriter, loc, iterationLocals[i].getType(), yieldedValue);
        rewriter.create<TempLocalGetOp>(loc, castedYieldedValue);
        rewriter.create<TempLocalSetOp>(loc, iterationLocals[i]);
      }
      rewriter.eraseOp(yieldOp);
    }

    rewriter.create<wasm::BranchOpDeprecated>(loc,
                                              inductionVariableUpdateBlock);

    // Condition block
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto castedUpperBound = typeConverter->materializeTargetConversion(
        rewriter, loc, localIndexType, upperBound);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedUpperBound);
    rewriter.create<wasm::TempLocalGetOp>(loc, inductionLocal);

    // check if upper bound <= induction variable
    // if true, branch to termination block
    rewriter.create<wasm::ILeUOp>(loc, indexType);
    rewriter.create<wasm::CondBranchOpDeprecated>(loc, terminationBlock,
                                                  bodyStartBlock);

    // Induction variable update block
    rewriter.setInsertionPointToEnd(inductionVariableUpdateBlock);
    auto castedStep = typeConverter->materializeTargetConversion(
        rewriter, loc, localIndexType, step);
    rewriter.create<wasm::TempLocalGetOp>(loc, inductionLocal);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedStep);
    rewriter.create<wasm::AddOp>(loc, indexType);
    rewriter.create<wasm::TempLocalSetOp>(loc, inductionLocal);

    rewriter.create<wasm::BranchOpDeprecated>(loc, conditionBlock);

    // Termination block: read final values of iteration variables
    rewriter.setInsertionPointToEnd(terminationBlock);
    rewriter.create<wasm::BlockLoopEndOpDeprecated>(loc);

    // Replace scf.for op with final iteration variable values
    rewriter.replaceOp(forOp, iterationLocals);

    // Remove block arguments from the body block
    TypeConverter::SignatureConversion signatureConversion(
        bodyStartBlock->getNumArguments());
    rewriter.applySignatureConversion(bodyStartBlock, signatureConversion);

    return success();
  }
};

struct ForOpLoweringWithBlockParams : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO

    return success();
  }
};

void populateScfToWasmPatterns(TypeConverter &typeConverter,
                               RewritePatternSet &patterns,
                               bool enableBlockParams) {

  if (enableBlockParams) {
    patterns.add<ForOpLoweringWithBlockParams>(typeConverter,
                                               patterns.getContext());
  } else {

    patterns.add<ForOpLowering>(typeConverter, patterns.getContext());
  }
}

} // namespace mlir::wasm