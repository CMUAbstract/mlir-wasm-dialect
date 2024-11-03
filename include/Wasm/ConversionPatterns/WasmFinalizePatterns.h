#ifndef WASM_WASMFINALIZEPATTERNS_H
#define WASM_WASMFINALIZEPATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Wasm/WasmFinalizeAnalysis.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

void populateWasmFinalizePatterns(MLIRContext *context,
                                  WasmFinalizeAnalysis &analysis,
                                  RewritePatternSet &patterns);

template <typename SourceOp>
class OpConversionPatternWithAnalysis : public OpConversionPattern<SourceOp> {
public:
  OpConversionPatternWithAnalysis(MLIRContext *context,
                                  WasmFinalizeAnalysis &analysis,
                                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit), analysis(analysis) {}

  WasmFinalizeAnalysis &getAnalysis() const { return analysis; }

private:
  WasmFinalizeAnalysis &analysis;
};

} // namespace mlir::wasm

#endif // WASM_WASMFINALIZEMODULESPATTERNS_H