
#include "Wasm/WasmFinalizeAnalysis.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {
WasmFinalizeAnalysis::WasmFinalizeAnalysis(ModuleOp &moduleOp) {
  moduleOp.walk([&](TempGlobalOp globalOp) {
    reg2Global.push_back(globalOp.getResult());
  });
  moduleOp.walk([&](WasmFuncOp funcOp) {
    Operation *funcOpPtr = funcOp.getOperation();
    for (auto arg : funcOp.getArguments()) {
      reg2LocOf[funcOpPtr].push_back(dyn_cast<Value>(arg));
    }
    numArgumentsOf[funcOpPtr] = funcOp.getNumArguments();

    funcOp.walk([&](Operation *op) {
      if (auto constantOp = dyn_cast<TempLocalOp>(op)) {
        mlir::Value result = constantOp.getResult();
        reg2LocOf[funcOpPtr].push_back(result);
      }
    });

    auto reg2Loc = reg2LocOf[funcOp];
    auto numArguments = numArgumentsOf[funcOp];

    vector<Attribute> types;
    types.reserve(reg2Loc.size() - numArguments);
    auto it = reg2Loc.begin();
    std::advance(it, numArguments);
    for (; it != reg2Loc.end(); it++) {
      auto typeAttr =
          TypeAttr::get(dyn_cast<LocalType>(it->getType()).getInner());
      types.push_back(typeAttr);
    }
    localTypesAttrOf[funcOpPtr] = types;
  });
}

std::string WasmFinalizeAnalysis::getGlobalName(const Value &tempGlobal) {
  int index = -1;
  // TODO: Error handling
  auto result = std::find(reg2Global.begin(), reg2Global.end(), tempGlobal);
  if (result != reg2Global.end()) {
    index = result - reg2Global.begin();
  }

  std::string name = "global_" + std::to_string(index);
  return name;
}

int WasmFinalizeAnalysis::getLocalIndex(Operation *funcOp, const Value &reg) {
  auto reg2Loc = reg2LocOf[funcOp];
  auto result = std::find(reg2Loc.begin(), reg2Loc.end(), reg);
  if (result != reg2Loc.end()) {
    return result - reg2Loc.begin();
  }
  // TODO: Error handling
  return -1;
}
vector<Attribute> WasmFinalizeAnalysis::getLocalTypesAttr(Operation *funcOp) {
  return localTypesAttrOf[funcOp];
}

int WasmFinalizeAnalysis::numLocals(Operation *funcOp) {
  return localTypesAttrOf[funcOp].size();
}

} // namespace mlir::wasm