#ifndef SSAWASM_ARITHTOSSAWASMPATTERNS_H
#define SSAWASM_ARITHTOSSAWASMPATTERNS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ssawasm {
void populateArithToSsaWasmPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns);

} // namespace mlir::ssawasm

#endif // SSAWASM_ARITHTOSSAWASMPATTERNS_H
