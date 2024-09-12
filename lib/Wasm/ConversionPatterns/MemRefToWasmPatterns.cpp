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
      rewriter.replaceOpWithNewOp<wasm::DataOp>(
          globalOp, rewriter.getStringAttr(globalOp.getSymName()),
          // TODO: offset should not be 0
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0),
          rewriter.getStringAttr(bytes.c_str()));
      return success();
    }

    return failure();
  }
};

struct GlobalGetOpLowering : public OpConversionPattern<memref::GlobalGetOp> {
  using OpConversionPattern<memref::GlobalGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalGetOp globalGetOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    // we should add a temporary wasm operation that holds the memory
    // offset and the shape
    return success();
  }
};

struct LoadOpLowering : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
    // TODO
    // 1. read from the temporary wasm operation the memory offset and
    // shape
    // 2. compute the memory location to read from
    // 3. perform the store operation (i32.store, i64.store, f32.store,
    // f64.store)
    return success();
  }
};

struct AllocOpLowering : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO
    // 1. compute the size of the memory to allocate
    // 2. create wasm.malloc operation (this should store the shape of the
    // allocated memory)
    return success();
  }
};

// TODO: pre-processing
// - compute the offset of each data segment
// - add (import "malloc")

// TODO: post-processing
// - remove temporary wasm operations
// - remove the `shape` attribute from wasm.malloc

void populateMemRefToWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.add<GlobalOpLowering>(typeConverter, patterns.getContext());
}

} // namespace mlir::wasm