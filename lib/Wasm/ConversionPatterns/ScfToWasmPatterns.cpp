#include "Wasm/ConversionPatterns/ScfToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

struct ForLowering : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = forOp.getLoc();

    auto *firstBlock = &forOp.getRegion().front();
    auto *lastBlock = &forOp.getRegion().back();

    auto loopOp = rewriter.create<wasm::LoopOp>(loc);
    rewriter.inlineRegionBefore(forOp.getRegion(), loopOp.getBody(),
                                loopOp.getBody().end());

    // firstBlockHead is the entrypoint of the loop. This does not have any
    // predecessors
    auto *firstBlockHead = firstBlock;
    // looping logic is defined in the condBlock
    auto *condBlock = rewriter.splitBlock(firstBlock, firstBlock->begin());
    auto *firstBlockEnd = rewriter.splitBlock(condBlock, condBlock->begin());
    // induction variable is updated in the bodyEndBlock
    Block *bodyEndBlock;
    if (firstBlock == lastBlock) {
      bodyEndBlock = firstBlockEnd;
    } else {
      bodyEndBlock = lastBlock;
    }
    Operation *terminator = bodyEndBlock->getTerminator();

    rewriter.setInsertionPointToEnd(bodyEndBlock);
    auto *terminationBlock = rewriter.createBlock(&loopOp.getRegion());

    // TODO: we don't really need this
    // we added this to avoid the error that the entry block should have no
    // predecessors
    rewriter.setInsertionPointToEnd(firstBlockHead);
    rewriter.create<wasm::BranchOp>(loc, condBlock);

    // set branching logic here
    // TODO: handle for loops with loop-carried values
    auto inductionVariable = firstBlockHead->getArgument(0);

    rewriter.setInsertionPointToEnd(condBlock);
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();

    if (!lowerBound || !upperBound) {
      return rewriter.notifyMatchFailure(forOp, "missing loop bounds");
    }

    auto castedInductionVariable = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(inductionVariable.getType()),
        inductionVariable);
    auto castedUpperBound = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(upperBound.getType()),
        upperBound);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedInductionVariable);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedUpperBound);

    rewriter.create<wasm::ILtUOp>(loc, rewriter.getI32Type()); // FIXME
    rewriter.create<wasm::CondBranchOp>(loc, terminationBlock);

    // update induction variable at the end of the loop body
    rewriter.setInsertionPointToEnd(bodyEndBlock);
    auto step = forOp.getStep();

    auto castedStep = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(step.getType()), step);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedInductionVariable);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedStep);
    rewriter.create<wasm::AddOp>(loc, rewriter.getI32Type());
    rewriter.create<wasm::TempLocalSetOp>(loc, castedInductionVariable);

    rewriter.replaceOpWithNewOp<wasm::BranchOp>(terminator, condBlock);

    // add terminator at the end of the termination block
    rewriter.setInsertionPointToEnd(terminationBlock);
    rewriter.create<wasm::LoopEndOp>(loc);

    rewriter.eraseOp(forOp);

    return success();
  }
};

void populateScfToWasmPatterns(TypeConverter &typeConverter,
                               RewritePatternSet &patterns) {
  patterns.add<ForLowering>(typeConverter, patterns.getContext());
}

} // namespace mlir::wasm