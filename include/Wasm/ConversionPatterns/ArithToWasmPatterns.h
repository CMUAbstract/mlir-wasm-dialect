#ifndef WASM_ARITHTOWASMPATTERNS_H
#define WASM_ARITHTOWASMPATTERNS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wasm {
void populateArithToWasmPatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns);

} // namespace mlir::wasm

#endif // WASM_ARITHTOWASMPATTERNS_H