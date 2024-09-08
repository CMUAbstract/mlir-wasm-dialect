#ifndef WASM_FUNCTOWASMPATTERNS_H
#define WASM_FUNCTOWASMPATTERNS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wasm {
void populateFuncToWasmPatterns(TypeConverter &typeConverter,
                                RewritePatternSet &patterns);

} // namespace mlir::wasm

#endif // WASM_FUNCTOWASMPATTERNS_H