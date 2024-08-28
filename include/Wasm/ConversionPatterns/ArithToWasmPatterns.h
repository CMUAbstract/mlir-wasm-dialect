#ifndef WASM_ARITHTOWASMPATTERNS_H
#define WASM_ARITHTOWASMPATTERNS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wasm {
void populateArithToWasmPatterns(MLIRContext *context,
                                 RewritePatternSet &patterns);

struct ConvertAdd : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
struct ConvertConstant : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace mlir::wasm

#endif // WASM_ARITHTOWASMPATTERNS_H