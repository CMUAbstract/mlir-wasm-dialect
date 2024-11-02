#ifndef WASM_WASMFINALIZEPATTERNS_H
#define WASM_WASMFINALIZEPATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Wasm/LocalNumbering.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

void populateWasmFinalizePatterns(MLIRContext *context,
                                  LocalNumbering &localNumbering,
                                  RewritePatternSet &patterns);

template <typename SourceOp>
class OpConversionPatternWithAnalysis : public OpConversionPattern<SourceOp> {
public:
  OpConversionPatternWithAnalysis(MLIRContext *context,
                                  LocalNumbering &localNumbering,
                                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit),
        localNumbering(localNumbering) {}

  LocalNumbering &getLocalNumbering() const { return localNumbering; }

private:
  LocalNumbering &localNumbering;
};

} // namespace mlir::wasm

#endif // WASM_WASMFINALIZEPATTERNS_H