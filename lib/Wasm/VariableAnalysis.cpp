#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Wasm/VariableAnalysis.h"
#include "Wasm/WasmOps.h"

using namespace mlir;
using namespace mlir::wasm;

namespace mlir::wasm {
VariableAnalysis::VariableAnalysis(Operation *op) {
  numArguments = 0;
  numVariables = 0;
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    // TODO: add function arguments to reg2Loc
    // function arguments are also local variables in wasm
    // Before that, we need to convert the arguments to LocalType
    // for (auto arg : func.getArguments()) {
    //   reg2Loc.push_back(dyn_cast<Value>(arg));
    // }
    numArguments = func.getNumArguments();

    func.walk([&](Operation *op) {
      if (auto constantOp = dyn_cast<TempLocalOp>(op)) {
        mlir::Value result = constantOp.getResult();
        reg2Loc.push_back(result);
        numVariables++;
      }
    });
  }
}

int VariableAnalysis::getNumVariables() { return numVariables; }

int VariableAnalysis::getLocalIndex(const Value &tempLocal) {
  auto result = std::find(reg2Loc.begin(), reg2Loc.end(), tempLocal);
  if (result != reg2Loc.end()) {
    // TODO: + numArguments should be removed after adding function arguments to
    // reg2Loc
    return result - reg2Loc.begin() + numArguments;
  }
  // TODO: Error handling
  return -1;
}

vector<Attribute> VariableAnalysis::getTypeAttrs() {
  vector<Attribute> types;
  types.reserve(reg2Loc.size());
  std::transform(reg2Loc.begin(), reg2Loc.end(), std::back_inserter(types),
                 [](const auto &tempLocal) {
                   return mlir::TypeAttr::get(
                       dyn_cast<LocalType>(tempLocal.getType()).getInner());
                 });
  return types;
}
} // namespace mlir::wasm