#ifndef SSAWASM_MEMREFTOSSAWASMPATTERNS_H
#define SSAWASM_MEMREFTOSSAWASMPATTERNS_H

#include "SsaWasm/SsaWasmOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ssawasm {
void populateMemRefToSsaWasmPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns);

} // namespace mlir::ssawasm

#endif // SSAWASM_MEMREFTOSSAWASMPATTERNS_H
