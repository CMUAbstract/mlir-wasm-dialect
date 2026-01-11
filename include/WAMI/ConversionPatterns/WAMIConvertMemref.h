//===- WAMIConvertMemref.h - MemRef to WasmSSA/WAMI patterns ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares conversion patterns from MemRef dialect to the upstream
// WasmSSA dialect and WAMI dialect for memory operations.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_CONVERSIONPATTERNS_WAMICONVERTMEMREF_H
#define WAMI_CONVERSIONPATTERNS_WAMICONVERTMEMREF_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
#include <map>
#include <string>

namespace mlir {
class ModuleOp;
}

namespace mlir::wami {

/// Computes the total size of a memref in bytes, aligned to the given alignment
int64_t computeMemRefSize(MemRefType memRefType, int64_t alignment = 1);

/// Analysis pass that assigns base addresses to global memrefs.
/// Addresses start at 1024 (WebAssembly convention to skip first 1KB).
class WAMIBaseAddressAnalysis {
public:
  explicit WAMIBaseAddressAnalysis(ModuleOp &moduleOp);

  /// Returns the base address assigned to a global memref by symbol name.
  uint32_t getBaseAddress(const std::string &globalOpName) const;

private:
  std::map<std::string, uint32_t> baseAddressMap;
};

/// Populates conversion patterns from MemRef dialect to WasmSSA/WAMI dialects.
void populateWAMIConvertMemrefPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    WAMIBaseAddressAnalysis &baseAddressAnalysis);

} // namespace mlir::wami

#endif // WAMI_CONVERSIONPATTERNS_WAMICONVERTMEMREF_H
