
namespace {
struct AddIOpLowering : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.replaceWithNewOp<ssawasm::AddOp>(op, adaptor.getResult(),
                                        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

} // namespace
namespace mlir::ssawasm {

void populateArithToSsaWasmPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<AddIOpLowering>(typeConverter, context);
}
} // namespace mlir::ssawasm
