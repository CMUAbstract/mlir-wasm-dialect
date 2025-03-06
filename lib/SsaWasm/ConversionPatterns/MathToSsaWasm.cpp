#include "SsaWasm/ConversionPatterns/MathToSsaWasm.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::ssawasm {

namespace {

struct SqrtOpLowering : public OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern<math::SqrtOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(math::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<SqrtOp>(op, adaptor.getOperand().getType(),
                                        adaptor.getOperand());
    return success();
  }
};

} // namespace

void populateMathToSsaWasmPatterns(TypeConverter &typeConverter,
                                   RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<SqrtOpLowering>(typeConverter, context);
}
} // namespace mlir::ssawasm
