
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

    auto lhsCastOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, localType, addIOp.getLhs());
    auto rhsCastOp = rewriter.create<UnrealizedConversionCastOp>(
        addIOp->getLoc(), localType, addIOp.getRhs());

    if ((addIOp.getLhs().getType() != type) ||
        (addIOp.getRhs().getType() != type)) {
      return rewriter.notifyMatchFailure(addIOp, "type mismatch");
    }

    rewriter.create<wasm::TempLocalGetOp>(loc, lhsCastOp.getResult(0));
    rewriter.create<wasm::TempLocalGetOp>(loc, rhsCastOp.getResult(0));

    rewriter.create<wasm::AddOp>(loc, type);
    rewriter.create<wasm::TempLocalSetOp>(loc, tempLocalOp.getResult());

    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, type, tempLocalOp.getResult());

    rewriter.replaceOp(addIOp, castOp);

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

    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, type, tempLocalOp.getResult());

    rewriter.replaceOp(constantOp, castOp);

    return success();
  }
};

void populateArithToWasmPatterns(MLIRContext *context,
                                 RewritePatternSet &patterns) {
  patterns.add<AddIOpLowering, ConstantOpLowering>(context);
}

} // namespace mlir::wasm