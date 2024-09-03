
#include "Wasm/ConversionPatterns/ArithToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

struct AddIOpLowering : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddIOp addIOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = addIOp->getLoc();
    Value result = addIOp.getResult();
    Type type = result.getType();

    auto localType = mlir::wasm::LocalType::get(addIOp->getContext(), type);

    auto tempLocalOp = rewriter.create<wasm::TempLocalOp>(loc, type);

    if ((addIOp.getLhs().getType() != type) ||
        (addIOp.getRhs().getType() != type)) {
      return rewriter.notifyMatchFailure(addIOp, "type mismatch");
    }
    // TODO: ideally TypeConverter should automatically handle type conversion
    // but it doesn't seem to work because TempLocalGetOp are not the operations
    // that are being converted
    auto localType = typeConverter->convertType(type);
    auto castedLhs = typeConverter->materializeTargetConversion(
        rewriter, loc, localType, addIOp.getLhs());
    auto castedRhs = typeConverter->materializeTargetConversion(
        rewriter, loc, localType, addIOp.getRhs());

    rewriter.create<wasm::TempLocalGetOp>(loc, castedLhs);
    rewriter.create<wasm::TempLocalGetOp>(loc, castedRhs);

    rewriter.create<wasm::AddOp>(loc, type);
    rewriter.create<wasm::TempLocalSetOp>(loc, tempLocalOp.getResult());
    rewriter.replaceOp(addIOp, tempLocalOp);

    return success();
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
                                 MLIRContext *context,
                                 RewritePatternSet &patterns) {
  patterns.add<AddIOpLowering, ConstantOpLowering>(typeConverter, context);
}

} // namespace mlir::wasm