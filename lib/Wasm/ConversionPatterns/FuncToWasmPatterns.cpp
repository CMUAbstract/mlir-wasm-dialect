#include "Wasm/ConversionPatterns/FuncToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

void populateFuncToWasmPatterns(TypeConverter &converter,
                                RewritePatternSet &patterns) {
  patterns.add<ConvertFunc, ConvertReturn>(context);
  // TODO
}

LogicalResult
ConvertFunc::matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  // TODO
  return success();
}

LogicalResult
ConvertReturn::matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  // TODO
  return success();
}

} // namespace mlir::wasm