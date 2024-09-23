// this computes the base address of each memref.global
#ifndef WASM_BASEADDRANALYSIS_H
#define WASM_BASEADDRANALYSIS_H

#include <map>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

using namespace std;

namespace mlir::wasm {
class BaseAddrAnalysis {
public:
  BaseAddrAnalysis(ModuleOp &module);
  unsigned getBaseAddr(string globalOpName);

private:
  std::map<string, unsigned> baseAddrMap;
};
} // namespace mlir::wasm

#endif // WASM_BASEADDRANALYSIS_H