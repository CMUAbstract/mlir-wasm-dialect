#ifndef WASM_VARIABLEANALYSIS_H
#define WASM_VARIABLEANALYSIS_H

#include <vector>

#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace std;

namespace mlir::wasm {
class VariableAnalysis {
public:
  VariableAnalysis(Operation *op);
  int getNumVariables();
  int getLocalIndex(const Value &reg);

  vector<Attribute> getTypeAttrs();

private:
  int numArguments;
  int numVariables;
  vector<Value> reg2Loc;
};
} // namespace mlir::wasm

#endif // WASM_VARIABLEANALYSIS_H