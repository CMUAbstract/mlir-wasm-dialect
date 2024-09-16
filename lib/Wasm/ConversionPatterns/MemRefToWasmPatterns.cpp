#include "Wasm/ConversionPatterns/MemRefToWasmPatterns.h"
#include "Wasm/WasmOps.h"

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

struct GlobalOpLowering : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern<memref::GlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (globalOp.isExternal()) {
      return rewriter.notifyMatchFailure(globalOp,
                                         "external global op not supported");
    }
    if (globalOp.isUninitialized()) {
      return rewriter.notifyMatchFailure(
          globalOp, "uninitialized global op not supported");
    }
    if (auto denseElementsAttr =
            dyn_cast<DenseElementsAttr>(globalOp.getConstantInitValue())) {
      auto rawData = denseElementsAttr.getRawData();
      std::string bytes = generateStr(rawData.data(), rawData.size());
      auto dataOp = rewriter.replaceOpWithNewOp<wasm::DataOp>(
          globalOp, rewriter.getStringAttr(globalOp.getSymName()),
          // TODO: baseAddr should not be 0
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0),
          rewriter.getStringAttr(bytes.c_str()),
          TypeAttr::get(globalOp.getType()));

      dataOp->setAttr("baseAddr",
                      rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
      return success();
    }

    return failure();
  }
};

struct GlobalGetOpLowering : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern<memref::GetGlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getGlobalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = getGlobalOp.getLoc();

    // we don't know if this is a global op or not
    IntegerAttr baseAddr;
    memref::GlobalOp globalOp =
        SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
            getGlobalOp,
            StringAttr::get(rewriter.getContext(), getGlobalOp.getName()));
    if (globalOp) {
      baseAddr = cast<IntegerAttr>(globalOp->getAttr("baseAddr"));
    }
    if (!globalOp) {
      auto dataOp = SymbolTable::lookupNearestSymbolFrom<wasm::DataOp>(
          getGlobalOp,
          StringAttr::get(rewriter.getContext(), getGlobalOp.getName()));
      baseAddr = cast<IntegerAttr>(dataOp->getAttr("baseAddr"));
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
                             OperandRange indices,
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

  rewriter.create<AddOp>(loc, rewriter.getI32Type());

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
    typeConverter->materializeTargetConversion(
        rewriter, loc, rewriter.getI32Type(), indices[i]);
    rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(strides[i]));
    rewriter.create<MulOp>(loc, rewriter.getI32Type());
    rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(4));
    rewriter.create<MulOp>(loc, rewriter.getI32Type());
  }
  return success();
}

struct LoadOpLowering : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult result =
        computeAddress(loadOp, loadOp.getMemRef(), loadOp.getMemRefType(),
                       loadOp.getIndices(), typeConverter, rewriter);
    if (failed(result)) {
      return result;
    }
    rewriter.replaceOpWithNewOp<LoadOp>(
        loadOp, TypeAttr::get(loadOp.getMemRefType().getElementType()));
    // TODO
    // 1. read from the temporary wasm operation about the memory offset and
    // shape
    // 2. compute the memory location to read from
    // 3. perform the load operation (i32.load, i64.load, f32.load, f64.load)
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
                       storeOp.getIndices(), typeConverter, rewriter);
    if (failed(result)) {
      return result;
    }

    // call wasm.store
    rewriter.replaceOpWithNewOp<StoreOp>(
        storeOp, TypeAttr::get(storeOp.getMemRefType().getElementType()));

    return success();
  }
};

int64_t memRefSize(MemRefType memRefType, int64_t alignment) {
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

    rewriter.create<MallocOp>(loc);

    auto localOp = rewriter.create<TempLocalOp>(loc, rewriter.getI32Type());
    // assume that the result of malloc is pushed to the stack
    rewriter.create<TempLocalSetOp>(loc, localOp.getResult());

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        allocOp, memRefType, localOp.getResult());
    return success();
  }
};

// TODO: pre-processing
// - compute the offset of each data segment
// - add (import "malloc")

void populateMemRefToWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.add<GlobalOpLowering, GlobalGetOpLowering, AllocOpLowering,
               StoreOpLowering>(typeConverter, patterns.getContext());
}

} // namespace mlir::wasm