#include "SsaWasm/ConversionPatterns/ArithToSsaWasm.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::ssawasm {

namespace {
struct AddIOpLowering : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<AddOp>(op, resultType, adaptor.getLhs(),
                                       adaptor.getRhs());
    return success();
  }
};

struct AddFOpLowering : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<AddOp>(op, resultType, adaptor.getLhs(),
                                       adaptor.getRhs());
    return success();
  }
};

struct ConstantOpLowering : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, adaptor.getValue());
    return success();
  }
};
} // namespace

void populateArithToSsaWasmPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<AddIOpLowering, AddFOpLowering, ConstantOpLowering>(
      typeConverter, context);
}
} // namespace mlir::ssawasm
