#ifndef WASM_UTILITY_H
#define WASM_UTILITY_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::wasm {

int64_t memRefSize(MemRefType memRefType, int64_t alignment);
} // namespace mlir::wasm

#endif // WASM_UTILITY_H
