//===- wasm-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "Wasm/WasmDialect.h"
#include "Wasm/WasmOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <vector>

namespace mlir::wasm {

llvm::LogicalResult getWatType(Type mlirType, std::string &watType) {
  if (mlirType.isInteger(32)) {
    watType = "i32";
  } else if (mlirType.isInteger(64)) {
    watType = "i64";
  } else if (mlirType.isF32()) {
    watType = "f32";
  } else if (mlirType.isF64()) {
    watType = "f64";
  } else {
    // Unsupported type
    return failure();
  }
  return success();
}

llvm::LogicalResult getWatType(Attribute attr, std::string &watType) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (intAttr.getType().isInteger(32)) {

      watType = "i32";
    } else if (intAttr.getType().isInteger(64)) {
      watType = "i64";
    } else {
      return failure();
    }
  } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    if (floatAttr.getType().isF32()) {
      watType = "f32";
    } else if (floatAttr.getType().isF64()) {
      watType = "f64";
    } else {
      return failure();
    }
  } else {
    return failure();
  }
  return success();
}

llvm::LogicalResult getNumericAttrValue(Attribute attr, std::string &value) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    value = std::to_string(intAttr.getInt());
  } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    value = llvm::formatv("{0:E}", floatAttr.getValueAsDouble());
  } else {
    return failure();
  }
  return success();
}

llvm::LogicalResult translateOperand(Value operand, raw_ostream &output) {
  if (isa<BlockArgument>(operand)) {
    // It's a function argument
    unsigned argIndex = cast<BlockArgument>(operand).getArgNumber();
    output << "(get_local $p" << argIndex << ")";
  } else {
    // It's a result from another operation, assign local variables as needed
    // For simplicity, we'll need to manage a symbol table for local variables
    // This is a placeholder
    output << "(local.get $var" << operand.getDefiningOp()->getResult(0) << ")";
  }
  return success();
}

llvm::LogicalResult translateConstantOp(ConstantOp constantOp,
                                        raw_ostream &output) {
  std::string watType;
  Attribute valueAttr = constantOp.getValue();
  if (failed(getWatType(valueAttr, watType))) {
    constantOp.emitError("unsupported constant type");
    return failure();
  }
  output << "(" << watType << ".const ";

  std::string value;
  if (failed(getNumericAttrValue(valueAttr, value))) {
    constantOp.emitError("unsupported constant value");
    return failure();
  }
  output << value;
  output << ")";
  return success();
}

llvm::LogicalResult translateLocalOp(LocalOp localOp, raw_ostream &output) {
  output << "(local ";
  bool isFirstOp = true;
  for (Attribute attr : localOp.getTypes()) {
    TypeAttr typeAttr = cast<TypeAttr>(attr);
    std::string watType;
    if (failed(getWatType(typeAttr.getValue(), watType))) {
      localOp.emitError("unsupported local type");
      return failure();
    }
    if (!isFirstOp) {
      output << " ";
    } else {
      isFirstOp = false;
    }
    output << watType;
  }
  output << ")";
  return success();
}

llvm::LogicalResult translateLocalGetOp(LocalGetOp localGetOp,
                                        raw_ostream &output) {
  output << "(local.get " << localGetOp.getIdx() << ")";
  return success();
}
llvm::LogicalResult translateLocalSetOp(LocalSetOp localSetOp,
                                        raw_ostream &output) {
  output << "(local.set " << localSetOp.getIdx() << ")";
  return success();
}

llvm::LogicalResult translateCallOp(CallOp callOp, raw_ostream &output) {
  output << "(call $" << callOp.getCallee() << ")";
  return success();
}

// we should use op interface to refactor this to support all arithmetic ops
llvm::LogicalResult translateAddOp(AddOp addOp, raw_ostream &output) {
  output << "(";
  std::string watType;
  if (failed(getWatType(addOp.getType(), watType))) {
    addOp.emitError("unsupported add type");
    return failure();
  }
  output << watType << ".add)";
  return success();
}
llvm::LogicalResult translateMulOp(MulOp mulOp, raw_ostream &output) {
  output << "(";
  std::string watType;
  if (failed(getWatType(mulOp.getType(), watType))) {
    mulOp.emitError("unsupported add type");
    return failure();
  }
  output << watType << ".mul)";
  return success();
}
llvm::LogicalResult translateILtUOp(ILtUOp iLtUOp, raw_ostream &output) {
  output << "(";
  std::string watType;
  if (failed(getWatType(iLtUOp.getType(), watType))) {
    iLtUOp.emitError("unsupported add type");
    return failure();
  }
  output << watType << ".lt_u)";
  return success();
}
llvm::LogicalResult translateLoadOp(LoadOp loadOp, raw_ostream &output) {
  output << "(";
  std::string watType;
  if (failed(getWatType(loadOp.getType(), watType))) {
    loadOp.emitError("unsupported load type");
    return failure();
  }
  output << watType << ".load)";
  return success();
}
llvm::LogicalResult translateStoreOp(StoreOp storeOp, raw_ostream &output) {
  output << "(";
  std::string watType;
  if (failed(getWatType(storeOp.getType(), watType))) {
    storeOp.emitError("unsupported store type");
    return failure();
  }
  output << watType << ".load)";
  return success();
}

llvm::LogicalResult translateReturnOp(WasmReturnOp returnOp,
                                      raw_ostream &output) {
  output << "return";
  return success();
}

static unsigned counter = 0;
std::string getUniqueBlockName() {
  return "$block" + std::to_string(counter++);
}
std::string getUniqueLoopName() { return "$loop" + std::to_string(counter++); }
std::string getUniqueCondLoopName() {
  return "$condloop" + std::to_string(counter++);
}

// TODO: move this to header file?
llvm::LogicalResult translateOperation(Operation *op, raw_ostream &output);

llvm::LogicalResult translateLoopOp(LoopOp loopOp, raw_ostream &output) {
  auto blockIt = loopOp.getRegion().begin();
  Block *preheader = &*blockIt;
  blockIt++;
  Block *conditionBlock = &*blockIt;
  blockIt++;
  Block *bodyBlock = &*blockIt;
  blockIt++;
  Block *inductionVariableUpdateBlock = &*blockIt;
  blockIt++;
  Block *terminationBlock = &*blockIt;

  std::string blockName = getUniqueBlockName();
  std::string loopName = getUniqueLoopName();
  std::string condLoopName = getUniqueCondLoopName();

  output << "(block " << blockName << "\n";
  output << "(loop " << loopName << "\n";
  // preheader block
  // assert that preheader does not have an operation except for the branch to
  // the condition block
  if (preheader->getOperations().size() != 1) {
    loopOp.emitError("preheader block should have exactly one operation");
    return failure();
  }
  auto &preheaderBranch = *preheader->getOperations().begin();
  if (!isa<BranchOp>(preheaderBranch)) {
    loopOp.emitError(
        "preheader block should have exactly one operation which is a branch");
    return failure();
  }
  auto preheaderBranchOp = cast<BranchOp>(preheaderBranch);
  if (preheaderBranchOp.getSuccessor() != conditionBlock) {
    loopOp.emitError("preheader block should have exactly one operation which "
                     "is a branch to the condition block");
    return failure();
  }

  // condition block
  // translate all operations in the condition block
  // except for the last one, which should be a conditional branch
  output << "(loop " << condLoopName << "\n";
  auto &conditionOps = conditionBlock->getOperations();
  if (conditionOps.empty()) {
    loopOp.emitError("Condition block is empty");
    return failure();
  }
  auto conditionOpIt = conditionOps.begin();
  auto conditionOpEnd = conditionOps.end();
  --conditionOpEnd; // Point to the last operation
  for (; conditionOpIt != conditionOpEnd; ++conditionOpIt) {
    if (failed(translateOperation(&*conditionOpIt, output))) {
      conditionOpIt->emitError(
          "Failed to translate operation in condition block");
      return failure();
    }
    output << "\n";
  }
  Operation *lastConditionOp = &*conditionOpEnd;
  if (!isa<CondBranchOp>(lastConditionOp)) {
    loopOp.emitError(
        "Last operation in condition block should be a CondBranchOp");
    return failure();
  }
  output << "br_if " << blockName << "\n";

  // bodyBlock
  // last operation in bodyBlock should be a branch to the
  // inductionVariableUpdateBlock
  auto &bodyBlockOps = bodyBlock->getOperations();
  auto bodyBlockOpIt = bodyBlockOps.begin();
  auto bodyBlockOpEnd = bodyBlockOps.end();
  --bodyBlockOpEnd; // Point to the last operation
  for (; bodyBlockOpIt != bodyBlockOpEnd; ++bodyBlockOpIt) {
    if (failed(translateOperation(&*bodyBlockOpIt, output))) {
      bodyBlockOpIt->emitError("Failed to translate operation in loop body");
      return failure();
    }
    output << "\n";
  }
  Operation *lastBodyBlockOp = &*bodyBlockOpEnd;
  if (!isa<BranchOp>(lastBodyBlockOp)) {
    loopOp.emitError("Last operation in body block should be a BranchOp");
    return failure();
  }
  output << "br " << condLoopName << "\n";
  output << ")\n"; // end of condition loop

  // inductionVariableUpdateBlock
  // translate all operations in the inductionVariableUpdateBlock
  // except for the last one, which should be a branch to the condition block
  auto &updateOps = inductionVariableUpdateBlock->getOperations();
  if (updateOps.empty()) {
    loopOp.emitError("Induction variable update block is empty");
    return failure();
  }

  auto updateOpIt = updateOps.begin();
  auto updateOpEnd = updateOps.end();
  --updateOpEnd; // Point to the last operation
  for (; updateOpIt != updateOpEnd; ++updateOpIt) {
    if (failed(translateOperation(&*updateOpIt, output))) {
      updateOpIt->emitError(
          "Failed to translate operation in induction variable update block");
      return failure();
    }
    output << "\n";
  }

  // Handle the last operation: Branch to condition block
  Operation *lastUpdateOp = &*updateOpEnd;
  if (!isa<BranchOp>(lastUpdateOp)) {
    lastUpdateOp->dump();
    loopOp.emitError("Last operation in induction variable update block should "
                     "be a BranchOp");
    return failure();
  }
  output << "br " << loopName << "\n";

  // termination block
  // assert that termination block has exactly one operation which is a loop end
  if (terminationBlock->getOperations().size() != 1) {
    loopOp.emitError("Termination block should have exactly one operation");
    return failure();
  }
  auto &terminationOp = *terminationBlock->getOperations().begin();
  if (!isa<LoopEndOp>(
          terminationOp)) { // Assuming LoopEndOp represents loop termination
    loopOp.emitError("Termination block's operation should be a LoopEndOp");
    return failure();
  }
  output << "))\n";

  return success();
}

llvm::LogicalResult translateOperation(Operation *op, raw_ostream &output) {
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    return translateConstantOp(constantOp, output);
  } else if (auto localOp = dyn_cast<LocalOp>(op)) {
    return translateLocalOp(localOp, output);
  } else if (auto localGetOp = dyn_cast<LocalGetOp>(op)) {
    return translateLocalGetOp(localGetOp, output);
  } else if (auto localSetOp = dyn_cast<LocalSetOp>(op)) {
    return translateLocalSetOp(localSetOp, output);
  } else if (auto callOp = dyn_cast<CallOp>(op)) {
    return translateCallOp(callOp, output);
  } else if (auto addOp = dyn_cast<AddOp>(op)) {
    return translateAddOp(addOp, output);
  } else if (auto mulOp = dyn_cast<MulOp>(op)) {
    return translateMulOp(mulOp, output);
  } else if (auto iLtUOp = dyn_cast<ILtUOp>(op)) {
    return translateILtUOp(iLtUOp, output);
  } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
    return translateLoadOp(loadOp, output);
  } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
    return translateStoreOp(storeOp, output);
  } else if (auto returnOp = dyn_cast<WasmReturnOp>(op)) {
    return translateReturnOp(returnOp, output);
  } else if (auto loopOp = dyn_cast<LoopOp>(op)) {
    return translateLoopOp(loopOp, output);
  } else {
    op->emitError("unsupported operation");
    return failure();
  }
  return success();
}

llvm::LogicalResult translateFunctionBody(WasmFuncOp funcOp,
                                          raw_ostream &output) {
  output << "\n    "; // Indentation for readability
  bool isFirstOp = true;
  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (!isFirstOp) {
        output << "\n    ";
      } else {
        isFirstOp = false;
      }
      if (failed(translateOperation(&op, output))) {
        return failure();
      }
    }
  }
  return success();
}

llvm::LogicalResult translateData(DataOp dataOp, raw_ostream &output) {
  output << "(data $.L" << dataOp.getSymName() << " (i32.const "
         << dataOp.getOffset() << ") \"";
  output << dataOp.getBytes();
  output << "\")\n";
  return success();
}

struct WasmFunctionSignature {
  std::vector<std::string> paramTypes;
  std::vector<std::string> resultTypes;

  WasmFunctionSignature(WasmFuncOp &funcOp) {
    paramTypes.reserve(funcOp.getNumArguments());
    for (Type argType : funcOp.getFunctionType().getInputs()) {
      // function arguments are always local variables
      Type innerType = cast<LocalType>(argType).getInner();
      std::string watType;
      if (failed(getWatType(innerType, watType))) {
        funcOp.emitError("unsupported argument type");
        // TODO: handle error
      }
      paramTypes.push_back(watType);
    }
    resultTypes.reserve(funcOp.getFunctionType().getResults().size());
    for (auto resultType : funcOp.getFunctionType().getResults()) {
      std::string watType;
      if (failed(getWatType(resultType, watType))) {
        funcOp.emitError("unsupported result type");
        // TODO: handle error
      }
      resultTypes.push_back(watType);
    }
  }

  bool operator<(const WasmFunctionSignature &other) const {
    if (paramTypes != other.paramTypes)
      return paramTypes < other.paramTypes;
    return resultTypes < other.resultTypes;
  }
  bool operator==(const WasmFunctionSignature &other) const {
    return paramTypes == other.paramTypes && resultTypes == other.resultTypes;
  }
};

using func_signature_map_t = std::map<WasmFunctionSignature, unsigned>;

llvm::LogicalResult translateFunction(func_signature_map_t &funcSignatureMap,
                                      WasmFuncOp funcOp, raw_ostream &output) {
  // Start function declaration
  output << "  (func $" << funcOp.getName() << " ";

  // function type
  WasmFunctionSignature funcSignature(funcOp);
  output << "(type " << funcSignatureMap[funcSignature] << ")";

  // function params
  if (!funcSignature.paramTypes.empty()) {
    output << " (param ";
    bool isFirst = true;
    for (auto paramType : funcSignature.paramTypes) {
      output << paramType;
      if (!isFirst) {
        output << " ";
      } else {
        isFirst = false;
      }
    }
    output << ")";
  }

  // function results
  if (!funcSignature.resultTypes.empty()) {
    output << " (result ";
    bool isFirst = true;
    for (auto resultType : funcSignature.resultTypes) {
      output << resultType;
      if (!isFirst) {
        output << " ";
      } else {
        isFirst = false;
      }
    }
    output << ")";
  }

  // Translate function body
  if (failed(translateFunctionBody(funcOp, output))) {
    funcOp.emitError("translating function body failed");
    return failure();
  }

  // Close function
  output << ")\n";
  return success();
}

func_signature_map_t initializeFunctionSignatureMap(ModuleOp &module) {
  func_signature_map_t funcSignatureMap;
  unsigned typeIndex = 0;
  for (auto funcOp : module.getOps<WasmFuncOp>()) {
    WasmFunctionSignature funcSignature(funcOp);
    if (auto search = funcSignatureMap.find(funcSignature);
        search == funcSignatureMap.end()) {
      funcSignatureMap[funcSignature] = typeIndex;
      typeIndex++;
    }
  }
  return funcSignatureMap;
}

LogicalResult
translateFunctionSignatures(func_signature_map_t &funcSignatureMap,
                            raw_ostream &output) {
  for (auto entry : funcSignatureMap) {
    output << "(type ";
    output << "(;" << entry.second << ";) ";
    output << "(func ";
    if (!entry.first.paramTypes.empty()) {
      for (auto paramType : entry.first.paramTypes) {
        output << "(param " << paramType << ") ";
      }
    }
    if (!entry.first.resultTypes.empty()) {
      for (auto resultType : entry.first.resultTypes) {
        output << "(result " << resultType << ") ";
      }
    }
    output << "))\n";
  }
  return success();
}

LogicalResult translateModuleToWat(ModuleOp module, raw_ostream &output) {
  func_signature_map_t funcSignatureMap =
      initializeFunctionSignatureMap(module);

  output << "(module\n";

  if (failed(translateFunctionSignatures(funcSignatureMap, output))) {
    return failure();
  }

  for (auto funcOp : module.getOps<WasmFuncOp>()) {
    if (failed(translateFunction(funcSignatureMap, funcOp, output))) {
      funcOp.emitError("failed to translate WasmFuncOp");
      return failure();
    }
  }

  for (auto dataOp : module.getOps<DataOp>()) {
    if (failed(translateData(dataOp, output))) {
      dataOp.emitError("failed to translate DataOp");
      return failure();
    }
  }

  output << ")\n";
  return success();
}
} // namespace mlir::wasm

using namespace mlir;

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration registration(
      "mlir-to-wat", "translate from mlir wasm dialect to wat",
      [](ModuleOp module, raw_ostream &output) {
        return mlir::wasm::translateModuleToWat(module, output);
      },
      [](DialectRegistry &registry) { registry.insert<wasm::WasmDialect>(); });

  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Tool"));

} // namespace mlir