#ifndef WASM_WASMFINALIZEPATTERNS_H
#define WASM_WASMFINALIZEPATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Wasm/VariableAnalysis.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

void populateWasmFinalizePatterns(MLIRContext *context,
                                  VariableAnalysis &analysis,
                                  RewritePatternSet &patterns);

template <typename SourceOp>
class OpConversionPatternWithAnalysis : public OpConversionPattern<SourceOp> {
public:
  OpConversionPatternWithAnalysis(MLIRContext *context,
                                  VariableAnalysis &analysis,
                                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit), analysis(analysis) {}

  VariableAnalysis &getAnalysis() const { return analysis; }

private:
  VariableAnalysis &analysis;
};

struct FinalizeTempLocalOp
    : public OpConversionPatternWithAnalysis<TempLocalOp> {
  using OpConversionPatternWithAnalysis<
      TempLocalOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FinalizeTempLocalGetOp
    : public OpConversionPatternWithAnalysis<TempLocalGetOp> {
  using OpConversionPatternWithAnalysis<
      TempLocalGetOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempLocalGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FinalizeTempLocalSetOp
    : public OpConversionPatternWithAnalysis<TempLocalSetOp> {
  using OpConversionPatternWithAnalysis<
      TempLocalSetOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(TempLocalSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace mlir::wasm

#endif // WASM_WASMFINALIZEPATTERNS_H