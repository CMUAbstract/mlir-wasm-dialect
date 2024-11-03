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
    rewriter.create<GlobalOp>(
        op.getLoc(), StringAttr::get(rewriter.getContext(), name),
        BoolAttr::get(rewriter.getContext(), op.getIsMutable()),
        TypeAttr::get(op.getType()));
    rewriter.eraseOp(op);
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

struct InsertLocalOp : public OpConversionPatternWithAnalysis<WasmFuncOp> {
  using OpConversionPatternWithAnalysis<
      WasmFuncOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(WasmFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    WasmFinalizeAnalysis &analysis = getAnalysis();

    if (op.getBody().empty()) {
      op.emitError("function body must have at least one "
                   "block when inserting local variables");
      return failure();
    }

    Block &entryBlock = op.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);

    vector<Attribute> typesAttr = analysis.getLocalTypesAttr(op.getOperation());
    ArrayRef<Attribute> types(typesAttr);
    rewriter.create<wasm::LocalOp>(op.getLoc(), rewriter.getArrayAttr(types));

    rewriter.startOpModification(op);
    rewriter.finalizeOpModification(op);

    return success();
  }
};

void populateWasmFinalizePatterns(MLIRContext *context,
                                  WasmFinalizeAnalysis &analysis,
                                  RewritePatternSet &patterns) {
  patterns.add<FinalizeTempLocalOp, FinalizeTempLocalGetOp,
               FinalizeTempLocalSetOp, FinalizeTempGlobalOp,
               FinalizeTempGlobalGetOp, FinalizeTempGlobalSetOp, InsertLocalOp>(
      context, analysis);
}

} // namespace mlir::wasm