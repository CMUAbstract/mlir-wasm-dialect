#include "llvm/Support/Debug.h"

#include "Wasm/ConversionPatterns/FuncToWasmPatterns.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {

struct FuncOpLowering : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    TypeConverter::SignatureConversion signatureConverter(
        funcOp.getFunctionType().getNumInputs());
    for (const auto &inputType :
         enumerate(funcOp.getFunctionType().getInputs())) {
      signatureConverter.addInputs(
          inputType.index(), typeConverter->convertType(inputType.value()));
    }

    // we should return i32 for memref types
    llvm::SmallVector<Type, 4> newResultTypes;
    for (auto resultType : funcOp.getFunctionType().getResults()) {
      if (isa<MemRefType>(resultType)) {
        newResultTypes.push_back(rewriter.getI32Type());
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    auto newFuncType = rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), newResultTypes);
    auto newFuncOp = rewriter.create<WasmFuncOp>(funcOp.getLoc(),
                                                 funcOp.getName(), newFuncType);

    if (failed(rewriter.convertRegionTypes(
            &funcOp.getBody(), *getTypeConverter(), &signatureConverter))) {
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
};

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    for (auto operand : adaptor.getOperands()) {
      rewriter.create<TempLocalGetOp>(returnOp.getLoc(), operand);
    }
    rewriter.replaceOpWithNewOp<WasmReturnOp>(returnOp);

    return success();
  }
};

void populateFuncToWasmPatterns(TypeConverter &typeConverter,
                                RewritePatternSet &patterns) {
  patterns.add<FuncOpLowering, ReturnOpLowering>(typeConverter,
                                                 patterns.getContext());
}
} // namespace mlir::wasm