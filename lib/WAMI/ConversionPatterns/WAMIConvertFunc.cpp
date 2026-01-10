//===- WAMIConvertFunc.cpp - Func to WasmSSA conversion ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion patterns from Func dialect to the upstream
// WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/ConversionPatterns/WAMIConvertFunc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"

namespace mlir::wami {

namespace {

//===----------------------------------------------------------------------===//
// FuncOp Lowering
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Set up signature conversion: wrap argument types in LocalRefType
    TypeConverter::SignatureConversion signatureConverter(
        op.getFunctionType().getNumInputs());

    for (const auto &[idx, inputType] :
         llvm::enumerate(op.getFunctionType().getInputs())) {
      Type convertedType = getTypeConverter()->convertType(inputType);
      // Wrap in LocalRefType for WasmSSA function arguments
      auto localRefType = wasmssa::LocalRefType::get(convertedType);
      signatureConverter.addInputs(idx, localRefType);
    }

    // Convert result types (not wrapped in LocalRefType)
    SmallVector<Type, 4> newResultTypes;
    for (Type resultType : op.getFunctionType().getResults()) {
      newResultTypes.push_back(getTypeConverter()->convertType(resultType));
    }

    // Create function type with value types.
    // wasmssa.func stores plain types in the function signature, but the entry
    // block arguments are LocalRefType - this asymmetry is by design and
    // handled by the wasmssa.func verifier.
    SmallVector<Type, 4> newInputTypes;
    for (Type inputType : op.getFunctionType().getInputs()) {
      newInputTypes.push_back(getTypeConverter()->convertType(inputType));
    }
    auto newFuncType = rewriter.getFunctionType(newInputTypes, newResultTypes);

    // Handle external function declarations
    if (op.isDeclaration()) {
      wasmssa::FuncImportOp::create(rewriter, op.getLoc(), op.getName(), "env",
                                    op.getName(), newFuncType);
      rewriter.eraseOp(op);
      return success();
    }

    // Create the new WasmSSA function
    auto newFuncOp = wasmssa::FuncOp::create(rewriter, op.getLoc(),
                                             op.getName(), newFuncType);

    // Convert region types using the signature conversion
    // This properly handles the argument type transformation
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *getTypeConverter(),
                                           &signatureConverter))) {
      return failure();
    }

    // Move the converted region to the new function
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Copy over any attributes (except function type and symbol name)
    for (const NamedAttribute &namedAttr : op->getAttrs()) {
      if (namedAttr.getName() != op.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReturnOp Lowering
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<wasmssa::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CallOp Lowering
//===----------------------------------------------------------------------===//

struct CallOpLowering : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert result types
    SmallVector<Type, 4> newResultTypes;
    for (Type resultType : op.getResultTypes()) {
      newResultTypes.push_back(getTypeConverter()->convertType(resultType));
    }

    auto callOp =
        wasmssa::FuncCallOp::create(rewriter, op.getLoc(), newResultTypes,
                                    op.getCallee(), adaptor.getOperands());
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateWAMIConvertFuncPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<FuncOpLowering, ReturnOpLowering, CallOpLowering>(typeConverter,
                                                                 context);
}

} // namespace mlir::wami
