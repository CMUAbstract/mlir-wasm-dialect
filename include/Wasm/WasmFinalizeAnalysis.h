#ifndef WASM_FINALIZE_ANALYSIS_H
#define WASM_FINALIZE_ANALYSIS_H

#include <map>
#include <string>
#include <vector>

#include "Wasm/WasmOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace std;

namespace mlir::wasm {
class WasmFinalizeAnalysis {
public:
  WasmFinalizeAnalysis(ModuleOp &op);
  string getGlobalName(const Value &reg);
  int getLocalIndex(Operation *func, const Value &reg);
  ArrayRef<Attribute> getLocalTypesRef(Operation *func);

private:
  map<Operation *, int> numArgumentsOf;
  map<Operation *, vector<Value>> reg2LocOf;
  vector<Value> reg2Global;
};
} // namespace mlir::wasm

#endif // WASM_FINALIZE_ANALYSIS_H