#include "Wasm/ConversionPatterns/ScfToWasmPatterns.h"
#include "Wasm/WasmOps.h"

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

struct ForLowering : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // we use i32 type for induction variable, lower bound, upper bound, and
    // step
    auto indexType = rewriter.getI32Type();
    auto localIndexType = typeConverter->convertType(rewriter.getI32Type());

    auto inductionVariable = forOp.getInductionVar();
    Value lowerBound = adaptor.getLowerBound();
    Value upperBound = adaptor.getUpperBound();
    Value step = adaptor.getStep();
    if (!lowerBound || !upperBound || !inductionVariable) {
      return rewriter.notifyMatchFailure(forOp, "missing loop bounds");
    }

    Location loc = forOp.getLoc();

    auto loopOp = rewriter.create<LoopOp>(loc);
    auto *entryBlock = rewriter.createBlock(&loopOp.getRegion());
    auto *conditionBlock = rewriter.createBlock(&loopOp.getRegion());
    auto *inductionVariableUpdateBlock =
        rewriter.createBlock(&loopOp.getRegion());
    auto *terminationBlock = rewriter.createBlock(&loopOp.getRegion());

    // move for loop body blocks between conditionBlock and
    // inductionVariableUpdateBlock
    auto *bodyStartBlock = &forOp.getRegion().front();
    auto *bodyEndBlock = &forOp.getRegion().back();
    rewriter.inlineRegionBefore(forOp.getRegion(),
                                inductionVariableUpdateBlock);

    // body end block
    rewriter.setInsertionPointToEnd(bodyEndBlock);
    Operation *terminator = bodyEndBlock->getTerminator();
    rewriter.eraseOp(terminator);
    rewriter.create<wasm::BranchOp>(loc, inductionVariableUpdateBlock);

    // entry block
    rewriter.setInsertionPointToEnd(entryBlock);
    auto inductionLocal =
        rewriter.create<TempLocalOp>(loc, indexType).getResult();
    auto castedInductionLocal = typeConverter->materializeSourceConversion(
        rewriter, loc, indexType, inductionLocal);

    // initialize induction variable with lower bound
    auto castedLowerBound = typeConverter->materializeSourceConversion(
        rewriter, loc, localIndexType, lowerBound);
    rewriter.create<TempLocalGetOp>(loc, castedLowerBound);
    rewriter.create<TempLocalSetOp>(loc, inductionLocal);

    rewriter.replaceAllUsesWith(inductionVariable, castedInductionLocal);
    rewriter.create<BranchOp>(loc, conditionBlock);

    // condition block
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto castedUpperBound = typeConverter->materializeTargetConversion(
        rewriter, loc, localIndexType, upperBound);
    rewriter.create<wasm::TempLocalGetOp>(loc, inductionLocal);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedUpperBound);

    rewriter.create<wasm::ILtUOp>(loc, indexType);
    rewriter.create<wasm::CondBranchOp>(loc, terminationBlock, bodyStartBlock);

    // induction variable update block
    rewriter.setInsertionPointToEnd(inductionVariableUpdateBlock);
    auto castedStep = typeConverter->materializeTargetConversion(
        rewriter, loc, localIndexType, step);
    rewriter.create<wasm::TempLocalGetOp>(loc, inductionLocal);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedStep);
    rewriter.create<wasm::AddOp>(loc, indexType);
    rewriter.create<wasm::TempLocalSetOp>(loc, inductionLocal);

    rewriter.create<wasm::BranchOp>(loc, conditionBlock);

    // termination block
    rewriter.setInsertionPointToEnd(terminationBlock);
    rewriter.create<wasm::LoopEndOp>(loc);

    // remove unused block argument from bodyStartBlock
    TypeConverter::SignatureConversion empty(bodyStartBlock->getNumArguments());
    rewriter.applySignatureConversion(bodyStartBlock, empty);

    rewriter.eraseOp(forOp);

    return success();
  }
};

void populateScfToWasmPatterns(TypeConverter &typeConverter,
                               RewritePatternSet &patterns) {
  patterns.add<ForLowering>(typeConverter, patterns.getContext());
}

} // namespace mlir::wasm