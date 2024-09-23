#include "Wasm/BaseAddrAnalysis.h"
#include "Wasm/utility.h"

namespace mlir::wasm {

BaseAddrAnalysis::BaseAddrAnalysis(ModuleOp &moduleOp) {
  unsigned baseAddr = 0;

  moduleOp.walk([this, &baseAddr](memref::GlobalOp globalOp) {
    baseAddrMap[globalOp.getName().str()] = baseAddr;

    int64_t alignment = globalOp.getAlignment().value_or(1);

    baseAddr += memRefSize(globalOp.getType(), alignment);
  });
}

unsigned BaseAddrAnalysis::getBaseAddr(string globalOpName) {
  return baseAddrMap[globalOpName];
}

} // namespace mlir::wasm