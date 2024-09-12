#include "Wasm/ConversionPatterns/MemRefToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

struct GlobalOpLowering : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern<memref::GlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    return success();
  }
};

void populateMemRefToWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.add<GlobalOpLowering>(typeConverter, patterns.getContext());
}

} // namespace mlir::wasm