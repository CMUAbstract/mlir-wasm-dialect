#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Wasm/VariableAnalysis.h"

using namespace mlir;

namespace mlir::wasm {
VariableAnalysis::VariableAnalysis(Operation *op) {
  numArguments = 0;
  numVariables = 0;
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    numArguments = func.getNumArguments();
    // TODO: initialize reg2Loc with arguments

    func.walk([&](Operation *op) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        mlir::Value result = constantOp.getResult();
        reg2Loc.push_back(result);
        numVariables++;
      }
      if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        mlir::Value result = addOp.getResult();
        reg2Loc.push_back(result);
        numVariables++;
      }
      // TODO: handle other operations that define new variables
    });
  }
}

int VariableAnalysis::getNumVariables() { return numVariables; }

int VariableAnalysis::getLocalIndex(const Value &reg) {
  auto result = std::find(reg2Loc.begin(), reg2Loc.end(), reg);
  if (result != reg2Loc.end()) {
    return result - reg2Loc.begin() + numArguments;
  }
  return -1;
}

vector<Attribute> VariableAnalysis::getTypeAttrs() {
  vector<Attribute> types;
  types.reserve(reg2Loc.size());
  std::transform(
      reg2Loc.begin(), reg2Loc.end(), std::back_inserter(types),
      [](const auto &reg) { return mlir::TypeAttr::get(reg.getType()); });
  return types;
}
} // namespace mlir::wasm