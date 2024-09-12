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

void populateMemRefToWasmPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.add<GlobalOpLowering>(typeConverter, patterns.getContext());
}

} // namespace mlir::wasm