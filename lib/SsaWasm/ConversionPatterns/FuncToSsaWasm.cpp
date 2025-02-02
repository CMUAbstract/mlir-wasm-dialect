#include "SsaWasm/ConversionPatterns/FuncToSsaWasm.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::ssawasm {

namespace {
struct FuncOpLowering : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    TypeConverter::SignatureConversion signatureConverter(
        op.getFunctionType().getNumInputs());
    for (const auto &inputType : enumerate(op.getFunctionType().getInputs())) {
      signatureConverter.addInputs(
          inputType.index(),
          getTypeConverter()->convertType(inputType.value()));
    }
    // handle memref?
    // TODO
    llvm::SmallVector<Type, 4> newResultTypes;
    for (auto resultType : op.getFunctionType().getResults()) {
      newResultTypes.push_back(resultType);
    }

    auto newFuncType = rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), newResultTypes);
    auto newFuncOp =
        rewriter.create<FuncOp>(op.getLoc(), op.getName(), newFuncType);

    if (failed(rewriter.convertRegionTypes(&op.getBody(), *getTypeConverter(),
                                           &signatureConverter))) {
      return failure();
    }

    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    for (const auto &namedAttr : op->getAttrs()) {
      if (namedAttr.getName() != op.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());

    return success();
  }
};

} // namespace

void populateFuncToSsaWasmPatterns(TypeConverter &typeConverter,
                                   RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<FuncOpLowering, ReturnOpLowering>(typeConverter, context);
}
} // namespace mlir::ssawasm
