#include "SsaWasm/ConversionPatterns/ScfToSsaWasm.h"
#include "SsaWasm/SsaWasmOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

namespace mlir::ssawasm {

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
// |   cond_br %cond, ^bodyStartBlock       |
// +----------------------------------------+
//         |
//         | True
//         v
// +----------------------------------------+
// | bodyStartBlock                         |
// |   <loop body begins>                   |
// |                                        |
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

    Block *bodyStartBlock = &op.getRegion().front();
    Block *bodyEndBlock = &op.getRegion().back();

    rewriter.inlineRegionBefore(op.getRegion(), inductionVariableUpdateBlock);

    // NOTE: bodyStartBlock and bodyEndBlock may be the same
    bodyStartBlock->dump();
    auto bodyArgs = bodyStartBlock->getArguments();

    // create locals for iteration variables before the BlockLoopOp:w
    rewriter.setInsertionPoint(blockLoopOp);
    SmallVector<Value, 4> iterationLocals;
    for (Value initArg : initArgs) {
      auto local = rewriter.create<LocalOp>(loc, initArg).getResult();
      auto castedLocal = rewriter
                             .create<UnrealizedConversionCastOp>(
                                 loc, rewriter.getF32Type(), local)
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
    castedInductionLocal.dump();

    for (unsigned i = 0; i < iterationLocals.size(); ++i) {
      iterationLocals[i].dump();
    }

    // replace uses of iteration variables and induction variable with locals
    rewriter.replaceAllUsesWith(bodyArgs[0], castedInductionLocal);
    for (unsigned i = 0; i < initArgs.size(); ++i) {
      rewriter.replaceAllUsesWith(bodyArgs[i + 1], iterationLocals[i]);
    }

    // Condition Block: evaluate loop condition
    rewriter.setInsertionPointToStart(conditionBlock);
    auto comparisonOp =
        rewriter.create<IleUOp>(loc, upperBound, inductionLocal);

    rewriter.create<BlockLoopCondBranchOp>(loc, comparisonOp.getResult(),
                                           blockEndLabel, bodyStartBlock);

    // Body End Block: update induction variable and branch to
    // inductionVariableUpdateBlock
    rewriter.setInsertionPointToEnd(bodyEndBlock);
    // get the old terminator
    Operation *terminator = bodyEndBlock->getTerminator();
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
        bodyStartBlock->getNumArguments());
    rewriter.applySignatureConversion(bodyStartBlock, signatureConversion);

    return success();
  }
};

void populateScfToSsaWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.add<ForOpLowering>(typeConverter, patterns.getContext());
}

} // namespace mlir::ssawasm