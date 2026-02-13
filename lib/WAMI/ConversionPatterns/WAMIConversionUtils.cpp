//===- WAMIConversionUtils.cpp - Shared conversion helpers -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements shared helper utilities for WAMI conversion patterns.
//
//===----------------------------------------------------------------------===//

#include "WAMI/ConversionPatterns/WAMIConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::wami {

Value ensureI32Condition(Value cond, Location loc,
                         ConversionPatternRewriter &rewriter) {
  Value normalized;
  (void)normalizeToType(cond, rewriter.getI32Type(), loc, rewriter, normalized);
  return normalized;
}

LogicalResult normalizeToType(Value input, Type targetType, Location loc,
                              ConversionPatternRewriter &rewriter,
                              Value &result) {
  Type inputType = input.getType();
  if (inputType == targetType) {
    result = input;
    return success();
  }

  auto srcInt = dyn_cast<IntegerType>(inputType);
  auto dstInt = dyn_cast<IntegerType>(targetType);
  if (srcInt && dstInt) {
    unsigned srcWidth = srcInt.getWidth();
    unsigned dstWidth = dstInt.getWidth();
    if (srcWidth < dstWidth) {
      result = arith::ExtUIOp::create(rewriter, loc, targetType, input);
      return success();
    }
    result = arith::TruncIOp::create(rewriter, loc, targetType, input);
    return success();
  }

  if (isa<IndexType>(inputType) && dstInt) {
    result = arith::IndexCastOp::create(rewriter, loc, targetType, input);
    return success();
  }

  if (srcInt && isa<IndexType>(targetType)) {
    result = arith::IndexCastOp::create(rewriter, loc, targetType, input);
    return success();
  }

  auto srcFloat = dyn_cast<FloatType>(inputType);
  auto dstFloat = dyn_cast<FloatType>(targetType);
  if (srcFloat && dstFloat) {
    if (srcFloat.isF32() && dstFloat.isF64()) {
      result = arith::ExtFOp::create(rewriter, loc, targetType, input);
      return success();
    }
    if (srcFloat.isF64() && dstFloat.isF32()) {
      result = arith::TruncFOp::create(rewriter, loc, targetType, input);
      return success();
    }
  }

  result = UnrealizedConversionCastOp::create(rewriter, loc, targetType, input)
               .getResult(0);
  return success();
}

LogicalResult normalizeOperandsToTypes(ValueRange inputs, TypeRange targetTypes,
                                       Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       SmallVectorImpl<Value> &results) {
  if (inputs.size() != targetTypes.size())
    return failure();

  results.clear();
  results.reserve(inputs.size());
  for (unsigned i = 0, e = inputs.size(); i < e; ++i) {
    Value normalized;
    if (failed(normalizeToType(inputs[i], targetTypes[i], loc, rewriter,
                               normalized)))
      return failure();
    results.push_back(normalized);
  }
  return success();
}

} // namespace mlir::wami
