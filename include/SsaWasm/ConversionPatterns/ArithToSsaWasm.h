#ifndef SSAWASM_ARITHTOSSAWASMPATTERNS_H
#define SSAWASM_ARITHTOSSAWASMPATTERNS_H

#include "SsaWasm/SsaWasmOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ssawasm {
void populateArithToSsaWasmPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns);

} // namespace mlir::ssawasm

#endif // SSAWASM_ARITHTOSSAWASMPATTERNS_H
