#ifndef SSAWASM_SCFTOSSAWASMPATTERNS_H
#define SSAWASM_SCFTOSSAWASMPATTERNS_H

#include "SsaWasm/SsaWasmOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ssawasm {
void populateScfToSsaWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);

} // namespace mlir::ssawasm

#endif // SSAWASM_SCFTOSSAWASMPATTERNS_H
