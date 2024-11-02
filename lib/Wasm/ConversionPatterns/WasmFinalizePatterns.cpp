#include "Wasm/ConversionPatterns/WasmFinalizePatterns.h"

namespace mlir::wasm {

struct FinalizeTempLocalOp
    : public OpConversionPatternWithAnalysis<TempLocalOp> {
  using OpConversionPatternWithAnalysis<
      TempLocalOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct FinalizeTempLocalGetOp
    : public OpConversionPatternWithAnalysis<TempLocalGetOp> {
  using OpConversionPatternWithAnalysis<
      TempLocalGetOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempLocalGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LocalNumbering &localNumbering = getLocalNumbering();

    rewriter.replaceOpWithNewOp<LocalGetOp>(
        op,
        rewriter.getIndexAttr(localNumbering.getLocalIndex(op->getOperand(0))));
    return success();
  }
};

struct FinalizeTempLocalSetOp
    : public OpConversionPatternWithAnalysis<TempLocalSetOp> {
  using OpConversionPatternWithAnalysis<
      TempLocalSetOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempLocalSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LocalNumbering &localNumbering = getLocalNumbering();

    rewriter.replaceOpWithNewOp<LocalSetOp>(
        op,
        rewriter.getIndexAttr(localNumbering.getLocalIndex(op->getOperand(0))));
    return success();
  }
};

void populateWasmFinalizePatterns(MLIRContext *context,
                                  LocalNumbering &localNumbering,
                                  RewritePatternSet &patterns) {
  patterns
      .add<FinalizeTempLocalOp, FinalizeTempLocalGetOp, FinalizeTempLocalSetOp>(
          context, localNumbering);
}
} // namespace mlir::wasm