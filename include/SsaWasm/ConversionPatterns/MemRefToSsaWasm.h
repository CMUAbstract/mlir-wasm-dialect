#ifndef SSAWASM_MEMREFTOSSAWASMPATTERNS_H
#define SSAWASM_MEMREFTOSSAWASMPATTERNS_H

#include "SsaWasm/SsaWasmOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace std;

namespace mlir::ssawasm {
class BaseAddressAnalysis {
public:
  BaseAddressAnalysis(ModuleOp &moduleOp);
  unsigned getBaseAddress(const string &globalOpName) const;

private:
  map<string, unsigned> baseAddressMap;
};

void populateMemRefToSsaWasmPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     BaseAddressAnalysis &baseAddressAnalysis);

} // namespace mlir::ssawasm

#endif // SSAWASM_MEMREFTOSSAWASMPATTERNS_H
