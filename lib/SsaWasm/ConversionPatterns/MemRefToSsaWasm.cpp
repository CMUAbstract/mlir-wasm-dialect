#include "SsaWasm/ConversionPatterns/MemRefToSsaWasm.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <map>

namespace mlir::ssawasm {
using namespace std;

int64_t computeMemRefSize(MemRefType memRefType, int64_t alignment) {
  // Step 1: Get the shape (dimensions)
  auto shape = memRefType.getShape();

  // Step 2: Compute the total number of elements
  int64_t totalElements = 1;
  for (int64_t dimSize : shape) {
    totalElements *= dimSize;
  }

  // Step 3: Determine the size of each element
  int64_t elementSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;

  // Step 4: Calculate the total memory size
  int64_t totalMemorySize = totalElements * elementSize;

  // Step 5: Adjust for alignment
  // Align to the nearest multiple of 'alignment'
  int64_t alignedMemorySize =
      ((totalMemorySize + alignment - 1) / alignment) * alignment;

  return alignedMemorySize;
}

BaseAddressAnalysis::BaseAddressAnalysis(ModuleOp &moduleOp) {
  unsigned baseAddress =
      1024; // start from 1024. This seems to be the wasm convention

  moduleOp.walk([this, &baseAddress](memref::GlobalOp op) {
    auto symName = op.getSymName().str();
    baseAddressMap.emplace(symName, baseAddress);

    int64_t alignment = op.getAlignment().value_or(1);

    baseAddress += computeMemRefSize(op.getType(), alignment);
  });
}

unsigned BaseAddressAnalysis::getBaseAddress(const string &globalOpName) const {
  return baseAddressMap.at(globalOpName);
}

namespace {

struct GlobalOpLowering : public OpConversionPattern<memref::GlobalOp> {
  GlobalOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                   BaseAddressAnalysis &baseAddressAnalysis)
      : OpConversionPattern<memref::GlobalOp>(typeConverter, context),
        baseAddressAnalysis(baseAddressAnalysis) {}
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto initialValue = adaptor.getInitialValue();
    auto baseAddress =
        baseAddressAnalysis.getBaseAddress(op.getSymName().str());

    if (initialValue.has_value()) {
      rewriter.replaceOpWithNewOp<ssawasm::DataOp>(
          op,
          adaptor.getSymName(), // StringAttr
          adaptor.getType(),    // TypeAttr
          initialValue.value(),
          adaptor.getConstant(), // UnitAttr
          baseAddress);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "not supported");
  }

private:
  BaseAddressAnalysis &baseAddressAnalysis;
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
      Type memRefElementType = memRefType.getElementType();
      int64_t elementSize = memRefElementType.getIntOrFloatBitWidth() / 8;
      Value size =
          rewriter
              .create<ConstantOp>(loc, rewriter.getI32IntegerAttr(elementSize))
              .getResult();
      Value stride =
          rewriter
              .create<ConstantOp>(loc, rewriter.getI32IntegerAttr(strides[i]))
              .getResult();
      Value sizeXstride = rewriter.create<MulOp>(loc, size, stride).getResult();

      Value sizeXstrideXindex =
          rewriter.create<MulOp>(loc, sizeXstride, indices[i]).getResult();
      result =
          rewriter.create<AddOp>(loc, result, sizeXstrideXindex).getResult();
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
    int64_t size = computeMemRefSize(memRefType, alignment);
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
    rewriter.replaceOpWithNewOp<CallOp>(op, "free", TypeRange{},
                                        adaptor.getMemref());
    return success();
  }
};

struct ExpandShapeLowering : public OpConversionPattern<memref::ExpandShapeOp> {
  using OpConversionPattern<memref::ExpandShapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = op.getResult().getType();
    auto resultType = getTypeConverter()->convertType(memRefType);
    rewriter.replaceOpWithNewOp<AsMemRefOp>(op, resultType, adaptor.getSrc());
    return success();
  }
};

struct CollapseShapeLowering
    : public OpConversionPattern<memref::CollapseShapeOp> {
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = op.getResult().getType();
    auto resultType = getTypeConverter()->convertType(memRefType);
    rewriter.replaceOpWithNewOp<AsMemRefOp>(op, resultType, adaptor.getSrc());
    return success();
  }
};

} // namespace

void populateMemRefToSsaWasmPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     BaseAddressAnalysis &baseAddressAnalysis) {

  MLIRContext *context = patterns.getContext();

  patterns.add<GlobalOpLowering>(typeConverter, context, baseAddressAnalysis);
  patterns.add<GetGlobalOpLowering, LoadOpLowering, StoreOpLowering,
               AllocOpLowering, DeallocOpLowering, ExpandShapeLowering,
               CollapseShapeLowering>(typeConverter, context);
}
} // namespace mlir::ssawasm
