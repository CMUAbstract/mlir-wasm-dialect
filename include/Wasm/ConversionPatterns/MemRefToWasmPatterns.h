#ifndef WASM_MEMREFTOWASMPATTERNS_H
#define WASM_MEMREFTOWASMPATTERNS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wasm {
void populateMemRefToWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);

} // namespace mlir::wasm

#endif // WASM_MEMREFTOWASMPATTERNS_H