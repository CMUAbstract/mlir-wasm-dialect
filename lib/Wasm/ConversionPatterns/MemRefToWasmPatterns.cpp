#include "Wasm/ConversionPatterns/MemRefToWasmPatterns.h"
#include "Wasm/WasmOps.h"
#include "Wasm/utility.h"

#include <iomanip>
#include <sstream>
#include <string>

static const uint8_t s_is_char_escaped[] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static const char s_hexdigits[] = "0123456789abcdef";

std::string generateStr(const void *bytes, size_t length) {
  const uint8_t *u8_data = static_cast<const uint8_t *>(bytes);

  std::stringstream ss;

  for (size_t i = 0; i < length; ++i) {
    uint8_t c = u8_data[i];
    if (s_is_char_escaped[c]) {
      ss << "\\";
      ss << s_hexdigits[c >> 4];
      ss << s_hexdigits[c & 0xf];
    } else {
      ss << c;
    }
  }

  return ss.str();
}

namespace mlir::wasm {

template <typename SourceOp>
class OpConversionPatternWithAnalysis : public OpConversionPattern<SourceOp> {
public:
  OpConversionPatternWithAnalysis(TypeConverter &typeConverter,
                                  MLIRContext *context,
                                  BaseAddrAnalysis &analysis,
                                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit),
        analysis(analysis) {}

  BaseAddrAnalysis &getAnalysis() const { return analysis; }

private:
  BaseAddrAnalysis &analysis;
};

struct GlobalOpLowering
    : public OpConversionPatternWithAnalysis<memref::GlobalOp> {
  using OpConversionPatternWithAnalysis<
      memref::GlobalOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(memref::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto analysis = getAnalysis();
    if (globalOp.isExternal()) {
      return rewriter.notifyMatchFailure(globalOp,
                                         "external global op not supported");
    }
    if (globalOp.isUninitialized()) {
      return rewriter.notifyMatchFailure(
          globalOp, "uninitialized global op not supported");
    }

    auto baseAddr = analysis.getBaseAddr(globalOp.getName().str());
    if (auto denseElementsAttr =
            dyn_cast<DenseElementsAttr>(globalOp.getConstantInitValue())) {
      auto rawData = denseElementsAttr.getRawData();
      std::string bytes = generateStr(rawData.data(), rawData.size());
      rewriter.replaceOpWithNewOp<wasm::DataOp>(
          globalOp, rewriter.getStringAttr(globalOp.getSymName()),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), baseAddr),
          rewriter.getStringAttr(bytes.c_str()),
          TypeAttr::get(globalOp.getType()));
      return success();
    }

    return failure();
  }
};

struct GlobalGetOpLowering
    : public OpConversionPatternWithAnalysis<memref::GetGlobalOp> {
  using OpConversionPatternWithAnalysis<
      memref::GetGlobalOp>::OpConversionPatternWithAnalysis;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getGlobalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto analysis = getAnalysis();
    Location loc = getGlobalOp.getLoc();

    // we don't know if this is a global op or not
    IntegerAttr baseAddr;
    memref::GlobalOp globalOp =
        SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
            getGlobalOp,
            StringAttr::get(rewriter.getContext(), getGlobalOp.getName()));
    if (globalOp) {
      baseAddr = rewriter.getIntegerAttr(
          rewriter.getIntegerType(32),
          getAnalysis().getBaseAddr(globalOp.getName().str()));
    }
    if (!globalOp) {
      auto dataOp = SymbolTable::lookupNearestSymbolFrom<wasm::DataOp>(
          getGlobalOp,
          StringAttr::get(rewriter.getContext(), getGlobalOp.getName()));
      baseAddr = rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                         dataOp.getOffset());
    }

    rewriter.create<ConstantOp>(loc, baseAddr);
    auto tempLocalOp = rewriter.create<TempLocalOp>(loc, rewriter.getI32Type());
    rewriter.create<TempLocalSetOp>(loc, tempLocalOp.getResult());

    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, getGlobalOp.getResult().getType(), tempLocalOp.getResult());

    rewriter.replaceOp(getGlobalOp, castOp);
    return success();
  }
};

LogicalResult computeAddress(Operation *op, Value memref, MemRefType memRefType,
                             ValueRange indices,
                             const TypeConverter *const typeConverter,
                             ConversionPatternRewriter &rewriter) {

  Location loc = op->getLoc();
  SmallVector<int64_t, 4> strides;
  int64_t offset;
  // Compute strides and offset using the utility function
  if (failed(mlir::getStridesAndOffset(memRefType, strides, offset))) {
    return rewriter.notifyMatchFailure(
        op, "Cannot compute strides and offset for the given MemRefType.");
  }

  // asssume that the memref if actually a pointer to the base address
  // convert the memref back to local<i32>
  auto pointer = typeConverter->materializeTargetConversion(
      rewriter, loc,
      LocalType::get(rewriter.getContext(), rewriter.getI32Type()), memref);
  // read
  rewriter.create<TempLocalGetOp>(loc, pointer);

  // FIXME: handle non-i32 types
  rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  // linearIndex += indices[i] * strides[i] * 4;
  for (int i = 0; i < memRefType.getRank(); i++) {
    // we need to push this index to the stack!
    Value castedIndex = typeConverter->materializeTargetConversion(
        rewriter, loc,
        LocalType::get(rewriter.getContext(), rewriter.getI32Type()),
        indices[i]);
    rewriter.create<TempLocalGetOp>(loc, castedIndex);

    rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(strides[i]));
    rewriter.create<MulOp>(loc, rewriter.getI32Type());
    rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(4));
    rewriter.create<MulOp>(loc, rewriter.getI32Type());
  }
  rewriter.create<AddOp>(loc, rewriter.getI32Type());

  return success();
}

struct LoadOpLowering : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult result =
        computeAddress(loadOp, loadOp.getMemRef(), loadOp.getMemRefType(),
                       adaptor.getIndices(), typeConverter, rewriter);
    if (failed(result)) {
      return result;
    }

    // the result of the load is pushed to the stack
    // we should pop it, assign it to a local variable,
    // and replace all uses of the load with this local variable
    auto loc = loadOp.getLoc();
    auto elementType = loadOp.getMemRefType().getElementType();

    rewriter.create<LoadOp>(loc, TypeAttr::get(elementType));
    auto localOp = rewriter.create<TempLocalOp>(loc, elementType);
    rewriter.create<TempLocalSetOp>(loc, localOp.getResult());
    rewriter.replaceOp(loadOp, localOp.getResult());

    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();
    auto value = storeOp.getValueToStore();

    auto castedValue = typeConverter->materializeTargetConversion(
        rewriter, loc, LocalType::get(rewriter.getContext(), value.getType()),
        value);

    // push this value to stack
    rewriter.create<TempLocalGetOp>(loc, castedValue);

    LogicalResult result =
        computeAddress(storeOp, storeOp.getMemRef(), storeOp.getMemRefType(),
                       adaptor.getIndices(), typeConverter, rewriter);
    if (failed(result)) {
      return result;
    }

    // call wasm.store
    rewriter.replaceOpWithNewOp<StoreOp>(
        storeOp, TypeAttr::get(storeOp.getMemRefType().getElementType()));

    return success();
  }
};

struct AllocOpLowering : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = allocOp.getLoc();
    MemRefType memRefType = allocOp.getResult().getType();
    int64_t alignment = 1;
    auto alignmentAttr = allocOp.getAlignment();
    if (alignmentAttr.has_value()) {
      alignment = alignmentAttr.value();
    }
    int64_t size = memRefSize(memRefType, alignment);
    rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(size));

    rewriter.create<CallOp>(loc,
                            StringAttr::get(rewriter.getContext(), "malloc"));

    auto localOp = rewriter.create<TempLocalOp>(loc, rewriter.getI32Type());
    // assume that the result of malloc is pushed to the stack
    rewriter.create<TempLocalSetOp>(loc, localOp.getResult());

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        allocOp, memRefType, localOp.getResult());
    return success();
  }
};

struct ExpandShapeLowering : public OpConversionPattern<memref::ExpandShapeOp> {
  using OpConversionPattern<memref::ExpandShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp expandShapeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        expandShapeOp, expandShapeOp.getResult().getType(),
        expandShapeOp.getOperand(0));

    return success();
  }
};

// TODO: pre-processing
// - compute the offset of each data segment
// - add (import "malloc")

void populateMemRefToWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  BaseAddrAnalysis &analysis) {
  patterns.add<GlobalOpLowering, GlobalGetOpLowering>(
      typeConverter, patterns.getContext(), analysis);
  patterns.add<AllocOpLowering, StoreOpLowering, LoadOpLowering,
               ExpandShapeLowering>(typeConverter, patterns.getContext());
}

} // namespace mlir::wasm