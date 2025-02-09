#include "SsaWasm/ConversionPatterns/MemRefToSsaWasm.h"
#include "Wasm/utility.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::ssawasm {

namespace {
using namespace std;

struct GlobalOpLowering : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern<memref::GlobalOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto initialValue = adaptor.getInitialValue();

    if (initialValue.has_value()) {
      rewriter.replaceOpWithNewOp<ssawasm::DataOp>(
          op,
          adaptor.getSymName(), // StringAttr
          adaptor.getType(),    // TypeAttr
          initialValue.value(),
          adaptor.getConstant(), // UnitAttr
          0);                    // TODO: compute base_addr
      return success();
    }
    return rewriter.notifyMatchFailure(op, "not supported");
  }
};

struct GetGlobalOpLowering : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern<memref::GetGlobalOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    WasmMemRefType memRefType =
        WasmMemRefType::get(rewriter.getContext(), op.getResult().getType());

    rewriter.replaceOpWithNewOp<GetDataOp>(op, memRefType, adaptor.getName());
    return success();
  }
};

pair<LogicalResult, Value>
generatePointerComputation(Operation *op, Value base, MemRefType memRefType,
                           ValueRange indices,
                           const TypeConverter *const typeConverter,
                           ConversionPatternRewriter &rewriter) {

  Location loc = op->getLoc();
  SmallVector<int64_t, 4> strides;
  int64_t offset;
  // Compute strides and offset using the utility function
  if (failed(mlir::getStridesAndOffset(memRefType, strides, offset))) {
    return std::make_pair(
        rewriter.notifyMatchFailure(
            op, "Cannot compute strides and offset for the given MemRefType."),
        Value());
  }

  auto pointerAsInteger = rewriter.create<AsPointerOp>(loc, base).getResult();

  Value result = pointerAsInteger;
  // linearIndex += indices[i] * strides[i] * 4;
  for (int i = 0; i < memRefType.getRank(); i++) {
    if (ShapedType::isDynamic(strides[i])) {
      return std::make_pair(
          rewriter.notifyMatchFailure(
              op, "Cannot handle dynamic strides in the MemRefType."),
          Value());
    } else {
      Value stride =
          rewriter
              .create<ConstantOp>(loc, rewriter.getI32IntegerAttr(strides[i]))
              .getResult();
      Value multiplied =
          rewriter.create<MulOp>(loc, stride, indices[i]).getResult();
      result = rewriter.create<AddOp>(loc, result, multiplied).getResult();
    }
  }

  return std::make_pair(success(), result);
}

struct LoadOpLowering : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto [result, pointer] = generatePointerComputation(
        op, adaptor.getMemref(), op.getMemRefType(), adaptor.getIndices(),
        typeConverter, rewriter);

    if (failed(result)) {
      return result;
    }

    auto memRefType = op.getMemRefType();
    auto elementType = memRefType.getElementType();
    auto resultType = getTypeConverter()->convertType(elementType);

    rewriter.replaceOpWithNewOp<LoadOp>(op, resultType, pointer);
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [result, pointer] = generatePointerComputation(
        op, adaptor.getMemref(), op.getMemRefType(), adaptor.getIndices(),
        typeConverter, rewriter);

    if (failed(result)) {
      return result;
    }

    rewriter.replaceOpWithNewOp<StoreOp>(op, pointer, adaptor.getValue());
    return success();
  }
};

struct AllocOpLowering : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memRefType = op.getType();
    int64_t alignment = 1;
    auto alignmentAttr = adaptor.getAlignment();
    if (alignmentAttr.has_value()) {
      alignment = alignmentAttr.value();
    }
    int64_t size = wasm::memRefSize(memRefType, alignment);
    auto constantOp =
        rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(size));
    rewriter.replaceOpWithNewOp<CallOp>(
        op, "malloc",
        TypeRange{WasmMemRefType::get(rewriter.getContext(), memRefType)},
        ValueRange{constantOp.getResult()});
    return success();
  }
};

struct DeallocOpLowering : public OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<CallOp>(
        op, "free", WasmIntegerType::get(rewriter.getContext(), 32),
        adaptor.getMemref());
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
