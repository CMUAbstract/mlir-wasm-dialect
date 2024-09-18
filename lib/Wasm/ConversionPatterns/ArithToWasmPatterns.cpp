
#include "Wasm/ConversionPatterns/ArithToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

template <typename S, typename T>
LogicalResult matchAndRewriteBinaryOp(S op, ConversionPatternRewriter &rewriter,
                                      const TypeConverter *typeConverter) {
  Location loc = op->getLoc();
  Value result = op.getResult();
  Type type = op.getType();

  auto tempLocalOp = rewriter.create<wasm::TempLocalOp>(loc, type);

  auto lhs = op.getOperand(0);
  auto rhs = op.getOperand(1);

  if ((lhs.getType() != type) || (rhs.getType() != type)) {
    return rewriter.notifyMatchFailure(op, "type mismatch");
  }

  auto castedLhs = typeConverter->materializeTargetConversion(
      rewriter, loc, LocalType::get(rewriter.getContext(), type), lhs);
  auto castedRhs = typeConverter->materializeTargetConversion(
      rewriter, loc, LocalType::get(rewriter.getContext(), type), rhs);

  rewriter.create<wasm::TempLocalGetOp>(loc, castedLhs);
  rewriter.create<wasm::TempLocalGetOp>(loc, castedRhs);

  rewriter.create<T>(loc, type);

  rewriter.create<wasm::TempLocalSetOp>(loc, tempLocalOp.getResult());
  rewriter.replaceOp(op, tempLocalOp);

  return success();
}

struct MulFOpLowering : public OpConversionPattern<arith::MulFOp> {
  using OpConversionPattern<arith::MulFOp>::OpConversionPattern; // Constructor

  LogicalResult
  matchAndRewrite(arith::MulFOp mulFOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return matchAndRewriteBinaryOp<arith::MulFOp, wasm::MulOp>(mulFOp, rewriter,
                                                               typeConverter);
  }
};

struct AddFOpLowering : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern; // Constructor
  LogicalResult
  matchAndRewrite(arith::AddFOp addFOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return matchAndRewriteBinaryOp<arith::AddFOp, wasm::AddOp>(addFOp, rewriter,
                                                               typeConverter);
  }
};

struct AddIOpLowering : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddIOp addIOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return matchAndRewriteBinaryOp<arith::AddIOp, wasm::AddOp>(addIOp, rewriter,
                                                               typeConverter);
  }
};

struct ConstantOpLowering : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp constantOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = constantOp->getLoc();
    Type type = constantOp.getResult().getType();
    Attribute value = constantOp->getAttr("value");

    auto tempLocalOp = rewriter.create<wasm::TempLocalOp>(loc, type);
    rewriter.create<wasm::ConstantOp>(loc, value);
    rewriter.create<wasm::TempLocalSetOp>(loc, tempLocalOp.getResult());
    rewriter.replaceOp(constantOp, tempLocalOp);

    return success();
  }
};

void populateArithToWasmPatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns
      .add<AddIOpLowering, AddFOpLowering, MulFOpLowering, ConstantOpLowering>(
          typeConverter, context);
}

} // namespace mlir::wasm
