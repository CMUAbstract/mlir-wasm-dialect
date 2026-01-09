//===- WAMIConvertFunc.h - Func to WasmSSA conversion -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares conversion patterns from Func dialect to the upstream
// WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_CONVERSIONPATTERNS_WAMICONVERTFUNC_H
#define WAMI_CONVERSIONPATTERNS_WAMICONVERTFUNC_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wami {

/// Populates patterns for converting Func dialect operations to WasmSSA.
void populateWAMIConvertFuncPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns);

} // namespace mlir::wami

#endif // WAMI_CONVERSIONPATTERNS_WAMICONVERTFUNC_H
