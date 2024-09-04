#include "llvm/Support/Debug.h"

#include "Wasm/ConversionPatterns/FuncToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

void populateFuncToWasmPatterns(TypeConverter &converter,
                                RewritePatternSet &patterns) {
  patterns.add<ConvertFunc, ConvertReturn>(patterns.getContext());
}

LogicalResult
ConvertFunc::matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  auto newFuncOp = rewriter.create<WasmFuncOp>(
      funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());
  // TODO: change FunctionType

  for (const auto &namedAttr : funcOp->getAttrs()) {
    if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
        namedAttr.getName() != SymbolTable::getSymbolAttrName())
      newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  rewriter.eraseOp(funcOp);

  return success();
}

LogicalResult
ConvertReturn::matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  // TODO: add TempLocalGetOp for each returnOp->getOperands()
  rewriter.replaceOpWithNewOp<WasmReturnOp>(returnOp);

  return success();
}

} // namespace mlir::wasm