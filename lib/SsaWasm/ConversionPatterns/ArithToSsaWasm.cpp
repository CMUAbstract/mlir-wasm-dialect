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
using MinFOpLowering = NumericBinaryOpLowering<arith::MinimumFOp, MinOp>;
using MaxFOpLowering = NumericBinaryOpLowering<arith::MaximumFOp, MaxOp>;
using RemUIOpLowering = NumericBinaryOpLowering<arith::RemUIOp, RemUOp>;

struct CmpIOpLowering : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getPredicate() == arith::CmpIPredicate::eq) {
      rewriter.replaceOpWithNewOp<EqOp>(op, adaptor.getLhs(), adaptor.getRhs());
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported comparison predicate");
    }
    return success();
  }
};

struct ConstantOpLowering : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Attribute value = adaptor.getValue();

    // If the constant is of index type, convert it to i32
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      if (intAttr.getType().isIndex()) {
        auto i32Type = IntegerType::get(op.getContext(), 32);
        // FIXME: Warn if the value is too large to fit in 32 bits
        APInt truncated = intAttr.getValue().trunc(32);
        value = IntegerAttr::get(i32Type, truncated);
      }
    }

    auto typedAttr = cast<TypedAttr>(value);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, typedAttr);
    return success();
  }
};

struct IndexCastOpLowering : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the index cast with its operand directly
    rewriter.replaceOp(op, adaptor.getIn());
    return success();
  }
};
} // namespace

void populateArithToSsaWasmPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<AddIOpLowering, AddFOpLowering, SubIOpLowering, SubFOpLowering,
               MulIOpLowering, MulFOpLowering, MinFOpLowering, MaxFOpLowering,
               CmpIOpLowering, RemUIOpLowering, ConstantOpLowering,
               IndexCastOpLowering>(typeConverter, context);
}
} // namespace mlir::ssawasm
