#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Wasm/LocalNumbering.h"
#include "Wasm/WasmOps.h"

namespace mlir::wasm {
LocalNumbering::LocalNumbering(Operation *op) {
  numArguments = 0;
  numLocals = 0;
  if (auto func = dyn_cast<wasm::WasmFuncOp>(op)) {
    for (auto arg : func.getArguments()) {
      reg2Loc.push_back(dyn_cast<Value>(arg));
    }
    numArguments = func.getNumArguments();

    func.walk([&](Operation *op) {
      if (auto constantOp = dyn_cast<TempLocalOp>(op)) {
        mlir::Value result = constantOp.getResult();
        reg2Loc.push_back(result);
        numLocals++;
      }
    });
  }
}

int LocalNumbering::getLocalIndex(const Value &tempLocal) {
  auto result = std::find(reg2Loc.begin(), reg2Loc.end(), tempLocal);
  if (result != reg2Loc.end()) {
    return result - reg2Loc.begin();
  }
  // TODO: Error handling
  return -1;
}

ArrayRef<Attribute> LocalNumbering::getLocalTypesRef() {
  vector<Attribute> types;
  types.reserve(reg2Loc.size() - numArguments);
  auto it = reg2Loc.begin();
  for (auto _ = 0; _ < numArguments; _++) {
    it++;
  }
  for (; it != reg2Loc.end(); it++) {
    auto typeAttr =
        TypeAttr::get(dyn_cast<LocalType>(it->getType()).getInner());
    types.push_back(typeAttr);
  }
  ArrayRef<Attribute> typesRef(types);
  return typesRef;
}
} // namespace mlir::wasm