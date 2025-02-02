#ifndef SSAWASM_FUNCTOSSAWASMPATTERNS_H
#define SSAWASM_FUNCTOSSAWASMPATTERNS_H

#include "SsaWasm/SsaWasmOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ssawasm {
void populateFuncToSsaWasmPatterns(TypeConverter &typeConverter,
                                   RewritePatternSet &patterns);

} // namespace mlir::ssawasm

#endif // SSAWASM_FUNCTOSSAWASMPATTERNS_H
