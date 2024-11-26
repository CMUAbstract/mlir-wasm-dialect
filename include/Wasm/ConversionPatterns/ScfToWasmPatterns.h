#ifndef WASM_SCFTOWASMPATTERNS_H
#define WASM_SCFTOWASMPATTERNS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wasm {
void populateScfToWasmPatterns(TypeConverter &typeConverter,
                               RewritePatternSet &patterns,
                               bool enableBlockParams);

} // namespace mlir::wasm

#endif // WASM_SCFTOWASMPATTERNS_H