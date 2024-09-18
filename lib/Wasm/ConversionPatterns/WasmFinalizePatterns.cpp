#include "Wasm/ConversionPatterns/WasmFinalizePatterns.h"

namespace mlir::wasm {

void populateWasmFinalizePatterns(MLIRContext *context,
                                  VariableAnalysis &analysis,
                                  RewritePatternSet &patterns) {
  patterns
      .add<FinalizeTempLocalOp, FinalizeTempLocalGetOp, FinalizeTempLocalSetOp>(
          context, analysis);
}

LogicalResult FinalizeTempLocalOp::matchAndRewrite(
    TempLocalOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult FinalizeTempLocalGetOp::matchAndRewrite(
    TempLocalGetOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  VariableAnalysis &analysis = getAnalysis();

  rewriter.replaceOpWithNewOp<LocalGetOp>(
      op, rewriter.getIndexAttr(analysis.getLocalIndex(op->getOperand(0))));
  return success();
}

LogicalResult FinalizeTempLocalSetOp::matchAndRewrite(
    TempLocalSetOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  VariableAnalysis &analysis = getAnalysis();

  rewriter.replaceOpWithNewOp<LocalSetOp>(
      op, rewriter.getIndexAttr(analysis.getLocalIndex(op->getOperand(0))));
  return success();
}
} // namespace mlir::wasm