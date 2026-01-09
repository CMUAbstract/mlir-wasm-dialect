//===- WAMIConvertArith.h - Arith to WasmSSA patterns ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares conversion patterns from Arith dialect to the upstream
// WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_CONVERSIONPATTERNS_WAMICONVERTARITH_H
#define WAMI_CONVERSIONPATTERNS_WAMICONVERTARITH_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wami {

/// Populates conversion patterns from Arith dialect to WasmSSA dialect.
void populateWAMIConvertArithPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns);

} // namespace mlir::wami

#endif // WAMI_CONVERSIONPATTERNS_WAMICONVERTARITH_H
