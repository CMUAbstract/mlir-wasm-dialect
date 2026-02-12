//===- WAMIConvertMath.cpp - Math to WasmSSA conversion --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion patterns from Math dialect to the upstream
// WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/ConversionPatterns/WAMIConvertMath.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"

namespace mlir::wami {

namespace {

struct SqrtOpLowering : public OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern<math::SqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "unsupported sqrt result type");

    rewriter.replaceOpWithNewOp<wasmssa::SqrtOp>(op, resultType,
                                                 adaptor.getOperand());
    return success();
  }
};

} // namespace

void populateWAMIConvertMathPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<SqrtOpLowering>(typeConverter, context);
}

} // namespace mlir::wami
