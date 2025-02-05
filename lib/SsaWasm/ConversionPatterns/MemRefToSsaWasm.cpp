#include "SsaWasm/ConversionPatterns/MemRefToSsaWasm.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::ssawasm {

namespace {
struct GlobalOpLowering : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern<memref::GlobalOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    return success();
  }
};

struct GetGlobalOpLowering : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern<memref::GetGlobalOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    return success();
  }
};

struct LoadOpLowering : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    return success();
  }
};

struct AllocOpLowering : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    return success();
  }
};

struct DeallocOpLowering : public OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    return success();
  }
};

struct ExpandShapeLowering : public OpConversionPattern<memref::ExpandShapeOp> {
  using OpConversionPattern<memref::ExpandShapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getResult().getType(), adaptor.getSrc());
    return success();
  }
};

struct CollapseShapeLowering
    : public OpConversionPattern<memref::CollapseShapeOp> {
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getResult().getType(), adaptor.getSrc());
    return success();
  }
};

} // namespace

void populateMemRefToSsaWasmPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GlobalOpLowering, GetGlobalOpLowering, LoadOpLowering,
               StoreOpLowering, AllocOpLowering, DeallocOpLowering,
               ExpandShapeLowering, CollapseShapeLowering>(typeConverter,
                                                           context);
}
} // namespace mlir::ssawasm
