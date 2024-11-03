#include "Wasm/ConversionPatterns/WasmFinalizePatterns.h"
#include <string>

namespace mlir::wasm {

struct FinalizeTempGlobalOp
    : public OpConversionPatternWithAnalysis<TempGlobalOp> {
  using OpConversionPatternWithAnalysis<
      TempGlobalOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string name = getAnalysis().getGlobalName(op.getResult());
    rewriter.replaceOpWithNewOp<GlobalOp>(
        op, StringAttr::get(rewriter.getContext(), name),
        TypeAttr::get(op.getType()));
    return success();
  }
};

struct FinalizeTempGlobalGetOp
    : public OpConversionPatternWithAnalysis<TempGlobalGetOp> {
  using OpConversionPatternWithAnalysis<
      TempGlobalGetOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempGlobalGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    WasmFinalizeAnalysis &analysis = getAnalysis();

    rewriter.replaceOpWithNewOp<GlobalGetOp>(
        op, FlatSymbolRefAttr::get(rewriter.getContext(),
                                   analysis.getGlobalName(op->getOperand(0))));
    return success();
  }
};

struct FinalizeTempGlobalSetOp
    : public OpConversionPatternWithAnalysis<TempGlobalSetOp> {
  using OpConversionPatternWithAnalysis<
      TempGlobalSetOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempGlobalSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    WasmFinalizeAnalysis &analysis = getAnalysis();

    rewriter.replaceOpWithNewOp<GlobalSetOp>(
        op, FlatSymbolRefAttr::get(rewriter.getContext(),
                                   analysis.getGlobalName(op->getOperand(0))));
    return success();
  }
};

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
    WasmFinalizeAnalysis &analysis = getAnalysis();

    Operation *funcOp = op->getParentOfType<WasmFuncOp>().getOperation();
    int localIndex = analysis.getLocalIndex(funcOp, op->getOperand(0));

    rewriter.replaceOpWithNewOp<LocalGetOp>(op,
                                            rewriter.getIndexAttr(localIndex));
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
    WasmFinalizeAnalysis &analysis = getAnalysis();

    Operation *funcOp = op->getParentOfType<WasmFuncOp>().getOperation();
    int localIndex = analysis.getLocalIndex(funcOp, op->getOperand(0));

    rewriter.replaceOpWithNewOp<LocalSetOp>(op,
                                            rewriter.getIndexAttr(localIndex));
    return success();
  }
};

void populateWasmFinalizePatterns(MLIRContext *context,
                                  WasmFinalizeAnalysis &analysis,
                                  RewritePatternSet &patterns) {
  patterns.add<FinalizeTempLocalOp, FinalizeTempLocalGetOp,
               FinalizeTempLocalSetOp, FinalizeTempGlobalOp,
               FinalizeTempGlobalGetOp, FinalizeTempGlobalSetOp>(context,
                                                                 analysis);
}

} // namespace mlir::wasm