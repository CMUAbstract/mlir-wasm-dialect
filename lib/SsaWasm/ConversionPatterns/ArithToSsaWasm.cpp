#include "SsaWasm/ConversionPatterns/ArithToSsaWasm.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::ssawasm {

namespace {
template <typename SrcOp, typename TgtOp>
struct NumericBinaryOpLowering : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the result type
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    // Replace old op with the new AddOp from your target dialect
    rewriter.replaceOpWithNewOp<TgtOp>(op, resultType, adaptor.getLhs(),
                                       adaptor.getRhs());
    return success();
  }
};

using AddIOpLowering = NumericBinaryOpLowering<arith::AddIOp, AddOp>;
using AddFOpLowering = NumericBinaryOpLowering<arith::AddFOp, AddOp>;
using SubIOpLowering = NumericBinaryOpLowering<arith::SubIOp, SubOp>;
using SubFOpLowering = NumericBinaryOpLowering<arith::SubFOp, SubOp>;
using MulIOpLowering = NumericBinaryOpLowering<arith::MulIOp, MulOp>;
using MulFOpLowering = NumericBinaryOpLowering<arith::MulFOp, MulOp>;

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
  patterns.add<AddIOpLowering, AddFOpLowering, SubIOpLowering, SubFOpLowering,
               MulIOpLowering, MulFOpLowering, ConstantOpLowering>(
      typeConverter, context);
}
} // namespace mlir::ssawasm
