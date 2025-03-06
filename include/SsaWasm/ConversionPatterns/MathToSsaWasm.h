#ifndef SSAWASM_MATHTOSSAWASMPATTERNS_H
#define SSAWASM_MATHTOSSAWASMPATTERNS_H

#include "SsaWasm/SsaWasmOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ssawasm {
void populateMathToSsaWasmPatterns(TypeConverter &typeConverter,
                                   RewritePatternSet &patterns);

} // namespace mlir::ssawasm

#endif // SSAWASM_MATHTOSSAWASMPATTERNS_H
