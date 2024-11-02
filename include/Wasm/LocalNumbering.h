#ifndef WASM_LOCALNUMBERING_H
#define WASM_LOCALNUMBERING_H

#include <vector>

#include "mlir/IR/Operation.h"

using namespace std;

namespace mlir::wasm {
class LocalNumbering {
public:
  LocalNumbering(Operation *op);
  int getLocalIndex(const Value &reg);

  ArrayRef<Attribute> getLocalTypesRef();

private:
  int numArguments;
  int numLocals;
  vector<Value> reg2Loc;
};
} // namespace mlir::wasm

#endif // WASM_LOCALNUMBERING_H