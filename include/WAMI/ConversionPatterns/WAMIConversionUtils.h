//===- WAMIConversionUtils.h - Shared conversion helpers -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares shared helper utilities for WAMI conversion patterns.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_CONVERSIONPATTERNS_WAMICONVERSIONUTILS_H
#define WAMI_CONVERSIONPATTERNS_WAMICONVERSIONUTILS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wami {

/// Normalizes \p cond to i32 for WebAssembly condition consumers.
Value ensureI32Condition(Value cond, Location loc,
                         ConversionPatternRewriter &rewriter);

/// Converts a single value to \p targetType at conversion boundaries.
LogicalResult normalizeToType(Value input, Type targetType, Location loc,
                              ConversionPatternRewriter &rewriter,
                              Value &result);

/// Converts each input operand to the corresponding target type.
LogicalResult normalizeOperandsToTypes(ValueRange inputs, TypeRange targetTypes,
                                       Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       SmallVectorImpl<Value> &results);

} // namespace mlir::wami

#endif // WAMI_CONVERSIONPATTERNS_WAMICONVERSIONUTILS_H
