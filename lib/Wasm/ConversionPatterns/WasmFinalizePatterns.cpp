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

  auto localGetOp = rewriter.create<LocalGetOp>(
      op->getLoc(),
      rewriter.getIndexAttr(analysis.getLocalIndex(op->getOperand(0))));
  rewriter.replaceOp(op, localGetOp);
  return success();
}

LogicalResult FinalizeTempLocalSetOp::matchAndRewrite(
    TempLocalSetOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  VariableAnalysis &analysis = getAnalysis();

  auto localSetOp = rewriter.create<LocalSetOp>(
      op->getLoc(),
      rewriter.getIndexAttr(analysis.getLocalIndex(op->getOperand(0))));
  rewriter.replaceOp(op, localSetOp);
  return success();
}
} // namespace mlir::wasm