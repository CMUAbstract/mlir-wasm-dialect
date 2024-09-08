#include "llvm/Support/Debug.h"

#include "Wasm/ConversionPatterns/FuncToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

void populateFuncToWasmPatterns(TypeConverter &typeConverter,
                                RewritePatternSet &patterns) {
  patterns.add<ConvertFunc, ConvertReturn>(typeConverter,
                                           patterns.getContext());
}

LogicalResult
ConvertFunc::matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {

  TypeConverter::SignatureConversion signatureConverter(
      funcOp.getFunctionType().getNumInputs());
  for (const auto &inputType :
       enumerate(funcOp.getFunctionType().getInputs())) {
    signatureConverter.addInputs(
        inputType.index(),
        LocalType::get(funcOp.getContext(), inputType.value()));
  }

  auto newFuncType =
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               funcOp.getFunctionType().getResults());
  auto newFuncOp = rewriter.create<WasmFuncOp>(funcOp.getLoc(),
                                               funcOp.getName(), newFuncType);

  if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter(),
                                         &signatureConverter))) {
    return failure();
  }

  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());

  for (const auto &namedAttr : funcOp->getAttrs()) {
    if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
        namedAttr.getName() != SymbolTable::getSymbolAttrName())
      newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  rewriter.eraseOp(funcOp);

  return success();
}

LogicalResult
ConvertReturn::matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  for (auto operand : returnOp->getOperands()) {
    // TODO: ideally TypeConverter should automatically handle type conversion
    // but it doesn't seem to work because TempLocalGetOp are not the operations
    // that are being converted
    auto casted = typeConverter->materializeTargetConversion(
        rewriter, returnOp.getLoc(),
        typeConverter->convertType(operand.getType()), operand);
    rewriter.create<TempLocalGetOp>(returnOp.getLoc(), casted);
  }
  rewriter.replaceOpWithNewOp<WasmReturnOp>(returnOp);

  return success();
}

} // namespace mlir::wasm