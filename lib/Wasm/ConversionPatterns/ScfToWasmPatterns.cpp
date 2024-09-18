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
    Location loc = forOp.getLoc();

    auto *firstBlock = &forOp.getRegion().front();
    auto *lastBlock = &forOp.getRegion().back();

    auto loopOp = rewriter.create<wasm::LoopOp>(loc);
    rewriter.inlineRegionBefore(forOp.getRegion(), loopOp.getBody(),
                                loopOp.getBody().end());

    auto *entrypointBlock = firstBlock;

    auto *conditionBlock = rewriter.splitBlock(firstBlock, firstBlock->begin());
    auto *firstBlockEnd =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *bodyStartBlock = firstBlockEnd;
    Block *bodyEndBlock;
    if (firstBlock == lastBlock) {
      bodyEndBlock = firstBlockEnd;
    } else {
      bodyEndBlock = lastBlock;
    }

    // NOTE: we add this block for the sake of clarity. Ideally this should be
    // merged with the bodyEndBlock by an optimizer.
    rewriter.setInsertionPointToEnd(bodyEndBlock);
    auto *inductionVariableUpdateBlock =
        rewriter.createBlock(&loopOp.getRegion());

    rewriter.setInsertionPointToEnd(bodyEndBlock);
    Operation *terminator = bodyEndBlock->getTerminator();
    rewriter.eraseOp(terminator);
    rewriter.create<wasm::BranchOp>(loc, inductionVariableUpdateBlock);

    rewriter.setInsertionPointToEnd(inductionVariableUpdateBlock);
    // terminationBlock is the exit point of the loop with a single terminator
    // wasm.LoopEndOp
    auto *terminationBlock = rewriter.createBlock(&loopOp.getRegion());

    // we add this to avoid the error that the entry block should have no
    // predecessors
    rewriter.setInsertionPointToEnd(entrypointBlock);

    // set branching logic here
    // TODO: handle for loops with loop-carried values
    auto inductionVariable = entrypointBlock->getArgument(0);
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    if (!lowerBound || !upperBound) {
      return rewriter.notifyMatchFailure(forOp, "missing loop bounds");
    }

    auto inductionLocalOp =
        rewriter.create<TempLocalOp>(loc, inductionVariable.getType());
    rewriter.replaceAllUsesWith(inductionVariable, inductionLocalOp);
    auto inductionLocal = inductionLocalOp.getResult();
    entrypointBlock->eraseArgument(0);

    // initialize induction local
    auto castedLowerBound = typeConverter->materializeTargetConversion(
        rewriter, loc,
        LocalType::get(rewriter.getContext(), lowerBound.getType()),
        lowerBound);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedLowerBound);
    rewriter.create<wasm::TempLocalSetOp>(loc, inductionLocal);

    rewriter.create<wasm::BranchOp>(loc, conditionBlock);

    rewriter.setInsertionPointToEnd(conditionBlock);

    auto castedUpperBound = typeConverter->materializeTargetConversion(
        rewriter, loc,
        LocalType::get(rewriter.getContext(), upperBound.getType()),
        upperBound);
    rewriter.create<wasm::TempLocalGetOp>(loc, inductionLocal);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedUpperBound);

    rewriter.create<wasm::ILtUOp>(loc, rewriter.getI32Type()); // FIXME
    rewriter.create<wasm::CondBranchOp>(loc, terminationBlock, bodyStartBlock);

    // update induction variable at the end of the loop body
    rewriter.setInsertionPointToEnd(inductionVariableUpdateBlock);
    auto step = forOp.getStep();

    auto castedStep = typeConverter->materializeTargetConversion(
        rewriter, loc, LocalType::get(rewriter.getContext(), step.getType()),
        step);
    rewriter.create<wasm::TempLocalGetOp>(loc, inductionLocal);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedStep);
    rewriter.create<wasm::AddOp>(loc, rewriter.getI32Type());
    rewriter.create<wasm::TempLocalSetOp>(loc, inductionLocal);

    rewriter.create<wasm::BranchOp>(loc, conditionBlock);

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