
#include "Wasm/ConversionPatterns/ArithToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

struct AddIOpLowering : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value result = op.getResult();

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    rewriter.setInsertionPoint(op);
    auto tempLocalOp =
        rewriter.create<wasm::TempLocalOp>(op->getLoc(), result.getType());

    auto localType =
        mlir::wasm::LocalType::get(op->getContext(), result.getType());
    auto lhsCastOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), localType, lhs);
    rewriter.create<wasm::TempLocalGetOp>(op->getLoc(), lhsCastOp.getResult(0));
    auto rhsCastOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), localType, rhs);
    rewriter.create<wasm::TempLocalGetOp>(op->getLoc(), rhsCastOp.getResult(0));
    // TODO: Verify somewhere that two locals are of same type
    rewriter.create<wasm::AddOp>(op->getLoc(), lhs.getType());
    rewriter.create<wasm::TempLocalSetOp>(op->getLoc(),
                                          tempLocalOp.getResult());

    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), result.getType(), tempLocalOp.getResult());
    rewriter.clearInsertionPoint();

    rewriter.replaceOp(op, castOp);

    return success();
  }
};
struct ConstantOpLowering : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value result = op.getResult();

    Attribute attr = op->getAttr("value");

    rewriter.setInsertionPoint(op);
    auto tempLocalOp =
        rewriter.create<wasm::TempLocalOp>(op->getLoc(), result.getType());
    rewriter.create<wasm::ConstantOp>(op->getLoc(), attr);
    rewriter.create<wasm::TempLocalSetOp>(op->getLoc(),
                                          tempLocalOp.getResult());
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), result.getType(), tempLocalOp.getResult());
    rewriter.clearInsertionPoint();

    rewriter.replaceOp(op, castOp);

    return success();
  }
};

void populateArithToWasmPatterns(MLIRContext *context,
                                 RewritePatternSet &patterns) {
  patterns.add<AddIOpLowering, ConstantOpLowering>(context);
}

} // namespace mlir::wasm