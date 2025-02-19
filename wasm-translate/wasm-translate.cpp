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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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
  } else if (auto continuationType = dyn_cast<ContinuationType>(mlirType)) {
    watType = "$" + continuationType.getName().str();
    return success();
  } else {
    // Unsupported type
    return failure();
  }
  return success();
}

llvm::LogicalResult getWatType(Attribute attr, std::string &watType) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return getWatType(intAttr.getType(), watType);
  } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return getWatType(floatAttr.getType(), watType);
  } else {
    // Unsupported attribute type
    return failure();
  }
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

template <typename T>
LogicalResult translateSimpleOp(T op, raw_ostream &output, std::string opName) {
  output << "(";
  std::string watType;
  if (failed(getWatType(op.getType(), watType))) {
    op.emitError("unsupported type");
    return failure();
  }
  output << watType << "." << opName << ")";
  return success();
}
// we should use op interface to refactor this to support all arithmetic ops
llvm::LogicalResult translateAddOp(AddOp addOp, raw_ostream &output) {
  return translateSimpleOp(addOp, output, "add");
}
llvm::LogicalResult translateMulOp(MulOp mulOp, raw_ostream &output) {
  return translateSimpleOp(mulOp, output, "mul");
}
llvm::LogicalResult translateILeUOp(ILeUOp iLeUOp, raw_ostream &output) {
  return translateSimpleOp(iLeUOp, output, "le_u");
}
llvm::LogicalResult translateEqOp(EqOp eqOp, raw_ostream &output) {
  return translateSimpleOp(eqOp, output, "eq");
}
llvm::LogicalResult translateLtSOp(LtSOp ltSOp, raw_ostream &output) {
  return translateSimpleOp(ltSOp, output, "lt_s");
}
llvm::LogicalResult translateDivSOp(IDivSOp divSOp, raw_ostream &output) {
  return translateSimpleOp(divSOp, output, "div_s");
}
llvm::LogicalResult translateLoadOp(LoadOp loadOp, raw_ostream &output) {
  return translateSimpleOp(loadOp, output, "load");
}
llvm::LogicalResult translateStoreOp(StoreOp storeOp, raw_ostream &output) {
  return translateSimpleOp(storeOp, output, "store");
}
llvm::LogicalResult translateFMinOp(FMinOp fMinOp, raw_ostream &output) {
  return translateSimpleOp(fMinOp, output, "min");
}
llvm::LogicalResult translateFMaxOp(FMaxOp fMaxOp, raw_ostream &output) {
  return translateSimpleOp(fMaxOp, output, "max");
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

// TODO: move this to header file?
llvm::LogicalResult translateOperation(Operation *op, raw_ostream &output);

// DEPRECATED. TODO: remove
llvm::LogicalResult translateBlockLoopOpDeprecated(BlockLoopOpDeprecated loopOp,
                                                   raw_ostream &output) {
  // FIXME: This translation logic should be all handled by lowering passes
  // and this function should perform only trivial translation

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

  output << "(block " << blockName << "\n";
  // preheader block

  auto &preheaderOps = preheader->getOperations();
  auto preheaderOpIt = preheaderOps.begin();
  auto preheaderOpEnd = preheaderOps.end();
  --preheaderOpEnd; // Point to the last operation
  for (; preheaderOpIt != preheaderOpEnd; ++preheaderOpIt) {
    if (failed(translateOperation(&*preheaderOpIt, output))) {
      preheaderOpIt->emitError(
          "Failed to translate operation in condition block");
      return failure();
    }
    output << "\n";
  }

  if (!isa<BranchOpDeprecated>(preheaderOpEnd)) {
    loopOp.emitError(
        "the last operation in the preheader block should be a branch");
    return failure();
  }
  auto preheaderBranchOp = cast<BranchOpDeprecated>(preheaderOpEnd);
  if (preheaderBranchOp.getSuccessor() != conditionBlock) {
    loopOp.emitError("preheader block should have exactly one operation which "
                     "is a branch to the condition block");
    return failure();
  }

  // condition block
  // translate all operations in the condition block
  // except for the last one, which should be a conditional branch
  output << "(loop " << loopName << "\n";
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
  if (!isa<CondBranchOpDeprecated>(lastConditionOp)) {
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
  if (!isa<BranchOpDeprecated>(lastBodyBlockOp)) {
    loopOp.emitError(
        "Last operation in body block should be a BranchOpDeprecated");
    return failure();
  }
  // we don't need an explicit branch to the inductionVariableUpdateBlock

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
  if (!isa<BranchOpDeprecated>(lastUpdateOp)) {
    loopOp.emitError("Last operation in induction variable update block should "
                     "be a BranchOpDeprecated");
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
  if (!isa<BlockLoopEndOpDeprecated>(
          terminationOp)) { // Assuming BlockLoopEndOp
                            // represents loop termination
    loopOp.emitError(
        "Termination block's operation should be a BlockLoopEndOp");
    return failure();
  }
  output << "))\n";

  return success();
}

// DEPRECATED. TODO: remove
LogicalResult translateLoopOpDeprecated(LoopOpDeprecated loopOp,
                                        raw_ostream &output) {
  output << "(loop $" << loopOp.getName() << "\n";
  for (auto &block : loopOp.getBody()) {
    if (&block == loopOp.getEntryBlock()) {
      continue;
    }
    for (Operation &op : block) {
      if (isa<LoopEndOp>(op)) {
        continue;
      }
      if (failed(translateOperation(&op, output))) {
        return failure();
      }
      output << "\n";
    }
  }
  output << ")";
  return success();
}

// DEPRECATED. TODO: remove
LogicalResult translateBranchOpDeprecated(BranchOpDeprecated branchOp,
                                          raw_ostream &output) {
  // For now, we assume that the branch destination is a LoopOp
  // TODO: Handle other types of destinations
  auto loopOp =
      cast<LoopOpDeprecated>(branchOp->getSuccessor(0)->getParentOp());

  // loop name
  output << "(br $" << loopOp.getName() << ")";
  return success();
}

LogicalResult translateGlobalGetOp(GlobalGetOp globalGetOp,
                                   raw_ostream &output) {
  output << "(global.get $" << globalGetOp.getName() << ")";
  return success();
}

LogicalResult translateGlobalSetOp(GlobalSetOp globalSetOp,
                                   raw_ostream &output) {
  output << "(global.set $" << globalSetOp.getName() << ")";
  return success();
}

LogicalResult translateResumeSwitchOp(ResumeSwitchOp resumeSwitchOp,
                                      raw_ostream &output) {
  output << "(resume $" << resumeSwitchOp.getCt() << " (on $"
         << resumeSwitchOp.getTag() << " switch))";
  return success();
}

LogicalResult translateSwitchOp(SwitchOp switchOp, raw_ostream &output) {
  output << "(switch $" << switchOp.getCt() << " $" << switchOp.getTag() << ")";
  return success();
}

LogicalResult translateNullContRefOp(NullContRefOp nullContRefOp,
                                     raw_ostream &output) {
  output << "(ref.null $" << nullContRefOp.getCt() << ")";
  return success();
}

LogicalResult translateTableGetOp(TableGetOp tableGetOp, raw_ostream &output) {
  output << "(table.get $" << tableGetOp.getTableName() << ")";
  return success();
}

LogicalResult translateTableSetOp(TableSetOp tableSetOp, raw_ostream &output) {
  output << "(table.set $" << tableSetOp.getTableName() << ")";
  return success();
}

LogicalResult translateFuncRefOp(FuncRefOp funcRefOp, raw_ostream &output) {
  output << "(ref.func $" << funcRefOp.getFunc() << ")";
  return success();
}

LogicalResult translateContNewOp(ContNewOp contNewOp, raw_ostream &output) {
  output << "(cont.new $" << contNewOp.getCt() << ")";
  return success();
}

LogicalResult translateElemDeclareFuncOp(ElemDeclareFuncOp elemDeclareFuncOp,
                                         raw_ostream &output) {
  output << "(elem declare func $" << elemDeclareFuncOp.getFuncName() << ")";
  return success();
}

LogicalResult translateBlockOp(BlockOp blockOp, raw_ostream &output) {
  output << "(block $" << blockOp.getName() << "\n";
  for (auto &block : blockOp.getBody()) {
    for (Operation &op : block) {
      if (isa<BlockEndOp>(op)) {
        continue;
      }
      if (failed(translateOperation(&op, output))) {
        return failure();
      }
      output << "\n";
    }
  }
  output << ")\n";
  return success();
}

LogicalResult translateLoopOp(LoopOp loopOp, raw_ostream &output) {
  output << "(loop $" << loopOp.getName() << "\n";
  for (auto &block : loopOp.getBody()) {
    for (Operation &op : block) {
      if (isa<LoopEndOp>(op)) {
        continue;
      }
      if (failed(translateOperation(&op, output))) {
        return failure();
      }
      output << "\n";
    }
  }
  output << ")\n";
  return success();
}

LogicalResult translateBranchOp(BranchOp branchOp, raw_ostream &output) {
  output << "(br $" << branchOp.getName() << ")";
  return success();
}

LogicalResult translateCondBranchOp(CondBranchOp condBranchOp,
                                    raw_ostream &output) {
  output << "(br_if $" << condBranchOp.getName() << ")";
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
  } else if (auto iLeUOp = dyn_cast<ILeUOp>(op)) {
    return translateILeUOp(iLeUOp, output);
  } else if (auto eqOp = dyn_cast<EqOp>(op)) {
    return translateEqOp(eqOp, output);
  } else if (auto ltSOp = dyn_cast<LtSOp>(op)) {
    return translateLtSOp(ltSOp, output);
  } else if (auto divSOp = dyn_cast<IDivSOp>(op)) {
    return translateDivSOp(divSOp, output);
  } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
    return translateLoadOp(loadOp, output);
  } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
    return translateStoreOp(storeOp, output);
  } else if (auto fMinOp = dyn_cast<FMinOp>(op)) {
    return translateFMinOp(fMinOp, output);
  } else if (auto fMaxOp = dyn_cast<FMaxOp>(op)) {
    return translateFMaxOp(fMaxOp, output);
  } else if (auto returnOp = dyn_cast<WasmReturnOp>(op)) {
    return translateReturnOp(returnOp, output);
    // DEPRECATED. TODO: remove
  } else if (auto loopOp = dyn_cast<LoopOpDeprecated>(op)) {
    return translateLoopOpDeprecated(loopOp, output);
    // DEPRECATED. TODO: remove
  } else if (auto branchOp = dyn_cast<BranchOpDeprecated>(op)) {
    return translateBranchOpDeprecated(branchOp, output);
    // DEPRECATED. TODO: remove
  } else if (auto blockLoopOp = dyn_cast<BlockLoopOpDeprecated>(op)) {
    return translateBlockLoopOpDeprecated(blockLoopOp, output);
  } else if (auto globalGetOp = dyn_cast<GlobalGetOp>(op)) {
    return translateGlobalGetOp(globalGetOp, output);
  } else if (auto globalSetOp = dyn_cast<GlobalSetOp>(op)) {
    return translateGlobalSetOp(globalSetOp, output);
  } else if (auto resumeSwitchOp = dyn_cast<ResumeSwitchOp>(op)) {
    return translateResumeSwitchOp(resumeSwitchOp, output);
  } else if (auto switchOp = dyn_cast<SwitchOp>(op)) {
    return translateSwitchOp(switchOp, output);
  } else if (auto nullContRefOp = dyn_cast<NullContRefOp>(op)) {
    return translateNullContRefOp(nullContRefOp, output);
  } else if (auto tableGetOp = dyn_cast<TableGetOp>(op)) {
    return translateTableGetOp(tableGetOp, output);
  } else if (auto tableSetOp = dyn_cast<TableSetOp>(op)) {
    return translateTableSetOp(tableSetOp, output);
  } else if (auto tableSetOp = dyn_cast<TableSetOp>(op)) {
    return translateTableSetOp(tableSetOp, output);
  } else if (auto funcRefOp = dyn_cast<FuncRefOp>(op)) {
    return translateFuncRefOp(funcRefOp, output);
  } else if (auto contNewOp = dyn_cast<ContNewOp>(op)) {
    return translateContNewOp(contNewOp, output);
  } else if (auto blockOp = dyn_cast<BlockOp>(op)) {
    return translateBlockOp(blockOp, output);
  } else if (auto loopOp = dyn_cast<LoopOp>(op)) {
    return translateLoopOp(loopOp, output);
  } else if (auto branchOp = dyn_cast<BranchOp>(op)) {
    return translateBranchOp(branchOp, output);
  } else if (auto condBranchOp = dyn_cast<CondBranchOp>(op)) {
    return translateCondBranchOp(condBranchOp, output);
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

struct FuncSignature {
  std::vector<std::string> paramTypes;
  std::vector<std::string> resultTypes;

  FuncSignature() = default;
  FuncSignature(FuncTypeDeclOp &funcTypeOp) {
    for (Type argType : funcTypeOp.getFunctionType().getInputs()) {
      // function arguments are always local variables
      std::string watType;
      if (failed(getWatType(argType, watType))) {
        funcTypeOp.emitError("unsupported argument type");
        // TODO: handle error
      }
      paramTypes.push_back(watType);
    }
    resultTypes.reserve(funcTypeOp.getFunctionType().getResults().size());
    for (auto resultType : funcTypeOp.getFunctionType().getResults()) {
      std::string watType;
      if (failed(getWatType(resultType, watType))) {
        funcTypeOp.emitError("unsupported result type");
        // TODO: handle error
      }
      resultTypes.push_back(watType);
    }
  }
  FuncSignature(ImportFuncOp &importFuncOp) {
    for (Type argType : importFuncOp.getFunctionType().getInputs()) {
      // function arguments are always local variables
      std::string watType;
      if (failed(getWatType(argType, watType))) {
        importFuncOp.emitError("unsupported argument type");
        // TODO: handle error
      }
      paramTypes.push_back(watType);
    }
    resultTypes.reserve(importFuncOp.getFunctionType().getResults().size());
    for (auto resultType : importFuncOp.getFunctionType().getResults()) {
      std::string watType;
      if (failed(getWatType(resultType, watType))) {
        importFuncOp.emitError("unsupported result type");
        // TODO: handle error
      }
      resultTypes.push_back(watType);
    }
  }
  FuncSignature(WasmFuncOp &funcOp) {
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

  bool operator<(const FuncSignature &other) const {
    if (paramTypes != other.paramTypes)
      return paramTypes < other.paramTypes;
    return resultTypes < other.resultTypes;
  }
  bool operator==(const FuncSignature &other) const {
    return paramTypes == other.paramTypes && resultTypes == other.resultTypes;
  }
};

class FuncSignatureList {
public:
  std::string getFunctionTypeNameOrIndex(FuncSignature &funcSignature) {
    auto search =
        std::find_if(funcSignatureList.begin(), funcSignatureList.end(),
                     [&](const std::pair<FuncSignature, std::string> &entry) {
                       return entry.first == funcSignature;
                     });
    if (search == funcSignatureList.end()) {
      // TODO: handle error
      return "UNKNOWN";
    }
    if (search->second != "") {
      return "$" + search->second;
    }
    return std::to_string(std::distance(funcSignatureList.begin(), search));
  }

  std::string getFunctionTypeNameOrIndexForSignature(size_t index) {
    auto name = funcSignatureList[index].second;
    if (name != "") {
      return "$" + name;
    }
    return "(;" + std::to_string(index) + ";)";
  }

  size_t size() { return funcSignatureList.size(); }

  void push_back(FuncSignature &funcSignature, std::string name) {
    auto search =
        std::find_if(funcSignatureList.begin(), funcSignatureList.end(),
                     [&](const std::pair<FuncSignature, std::string> &entry) {
                       return entry.first == funcSignature;
                     });
    if (search == funcSignatureList.end()) {
      funcSignatureList.push_back(std::make_pair(funcSignature, name));
    } else {
      search->second = name;
    }
  }
  void push_back(FuncSignature &funcSignature) {
    auto search =
        std::find_if(funcSignatureList.begin(), funcSignatureList.end(),
                     [&](const std::pair<FuncSignature, std::string> &entry) {
                       return entry.first == funcSignature;
                     });
    if (search == funcSignatureList.end()) {
      funcSignatureList.push_back(std::make_pair(funcSignature, ""));
    }
  }

  FuncSignature get(size_t index) { return funcSignatureList[index].first; }

private:
  std::vector<std::pair<FuncSignature, std::string>> funcSignatureList;
};

llvm::LogicalResult translateFunction(FuncSignatureList &funcSignatureList,
                                      WasmFuncOp funcOp, raw_ostream &output) {
  // Start function declaration
  output << "  (func $" << funcOp.getName() << " ";

  // function type
  FuncSignature funcSignature(funcOp);
  output << "(type ";
  if (funcOp->hasAttr("type_id")) {
    output << "$" << cast<StringAttr>(funcOp->getAttr("type_id")).getValue()
           << ")";
  } else {
    output << funcSignatureList.getFunctionTypeNameOrIndex(funcSignature)
           << ")";
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

FuncSignatureList initializeFuncSignatureList(ModuleOp &moduleOp,
                                              bool addDebugFunctions) {
  FuncSignatureList funcSignatureList;
  // Type 0, 1, 2 and 3 are reserved for malloc/free related functions.
  FuncSignature mallocSignature; // type 0
  mallocSignature.paramTypes.push_back("i32");
  mallocSignature.resultTypes.push_back("i32");
  funcSignatureList.push_back(mallocSignature);

  FuncSignature freeSignature; // type 1
  freeSignature.paramTypes.push_back("i32");
  funcSignatureList.push_back(freeSignature);

  FuncSignature prependAllocSignature; // type 2
  prependAllocSignature.paramTypes.push_back("i32");
  prependAllocSignature.paramTypes.push_back("i32");
  prependAllocSignature.paramTypes.push_back("i32");
  prependAllocSignature.resultTypes.push_back("i32");
  funcSignatureList.push_back(prependAllocSignature);

  FuncSignature abortSignature; // type 3
  funcSignatureList.push_back(abortSignature);

  // for debugging purposes, we include log functions: log_i32 and log_f32
  // Note that log_i32 has the same signature as malloc
  if (addDebugFunctions) {
    FuncSignature logF32Signature; // type 4
    logF32Signature.paramTypes.push_back("f32");
    logF32Signature.resultTypes.push_back("f32");
    funcSignatureList.push_back(logF32Signature);
  }

  // add defined signatures
  for (auto funcTypeOp : moduleOp.getOps<FuncTypeDeclOp>()) {
    FuncSignature funcSignature(funcTypeOp);
    funcSignatureList.push_back(funcSignature, funcTypeOp.getName().str());
  }

  // add signatures of functions, if they are not already in the list
  for (auto importFuncOp : moduleOp.getOps<ImportFuncOp>()) {
    FuncSignature funcSignature(importFuncOp);
    funcSignatureList.push_back(funcSignature);
  }
  for (auto funcOp : moduleOp.getOps<WasmFuncOp>()) {
    if (funcOp->hasAttr("type_id")) {
      // If a function has a type_id attribute,
      // its type declaration is already handled separately
      continue;
    }
    FuncSignature funcSignature(funcOp);
    funcSignatureList.push_back(funcSignature);
  }
  return funcSignatureList;
}

LogicalResult translateFunctionSignatures(FuncSignatureList &funcSignatureList,
                                          raw_ostream &output) {
  for (size_t i = 0; i < funcSignatureList.size(); i++) {
    output << "(type "
           << funcSignatureList.getFunctionTypeNameOrIndexForSignature(i)
           << " ";
    output << "(func ";
    if (!funcSignatureList.get(i).paramTypes.empty()) {
      output << "(param ";
      for (auto paramType : funcSignatureList.get(i).paramTypes) {
        output << paramType;
        output << " ";
      }
      output << ") ";
    }
    if (!funcSignatureList.get(i).resultTypes.empty()) {
      output << "(result ";
      for (auto resultType : funcSignatureList.get(i).resultTypes) {
        output << resultType;
        output << " ";
      }
      output << ")";
    }
    output << "))\n";
  }
  return success();
}

LogicalResult translateRecContFuncDeclOps(ModuleOp &moduleOp,
                                          raw_ostream &output) {
  if (moduleOp.getOps<RecContFuncDeclOp>().empty()) {
    return success();
  }
  output << "(rec\n";
  output << "    (type $ft (func (param (ref null $ct))))\n";
  output << "    (type $ct (cont $ft))\n";
  output << ")\n";

  return success();
}

LogicalResult translateImportOps(ModuleOp &moduleOp, raw_ostream &output,
                                 FuncSignatureList &funcSignatureList,
                                 bool addDebugFunctions) {
  moduleOp.walk([&](ImportFuncOp importFuncOp) {
    // FIXME
    FuncSignature signature(importFuncOp);
    output << "(import \"" << "env" << "\"" << " " << "\""
           << importFuncOp.getName() << "\"" << " "
           << "(func $" << importFuncOp.getName() << " " << "(type "
           << funcSignatureList.getFunctionTypeNameOrIndex(signature) << ")"
           << "))\n";
  });

  if (addDebugFunctions) {
    output << R""""(
   (import "env" "log_i32" (func $log_i32 (type 0)))
   (import "env" "log_f32" (func $log_f32 (type 4)))
   )"""";
  }
  return success();
}

LogicalResult translateTagOps(ModuleOp &moduleOp, raw_ostream &output) {
  moduleOp.walk(
      [&](TagOp tagOp) { output << "(tag $" << tagOp.getName() << ")\n"; });
  return success();
}

// DEPRECATED. TODO: remove
LogicalResult translateContinuationTypeDeclOps(ModuleOp &moduleOp,
                                               raw_ostream &output) {
  moduleOp.walk([&](ContinuationTypeDeclOp op) {
    auto contType = cast<ContinuationType>(op.getCont());
    output << "(type $" << contType.getName().getValue() << " (cont $"
           << contType.getFuncId().getValue() << "))\n";
  });
  return success();
}

LogicalResult translateTableOps(ModuleOp &moduleOp, raw_ostream &output) {
  moduleOp.walk([&](ContinuationTableOp continuationTableOp) {
    // FIXME: handle nullability at operation level
    output << "(table $" << continuationTableOp.getName() << " "
           << continuationTableOp.getSize() << " " << "(ref null $"
           << continuationTableOp.getCt() << "))\n";
  });
  return success();
}

LogicalResult translateGlobalOps(ModuleOp &moduleOp, raw_ostream &output) {
  moduleOp.walk([&](GlobalOp globalOp) {
    output << "(global $" << globalOp.getName() << " ";
    output << "(";
    if (globalOp.getIsMutable()) {
      output << "mut ";
    }
    std::string watType;
    if (failed(getWatType(globalOp.getType(), watType))) {
      globalOp.emitError("unsupported global type");
    }
    output << watType << ")";
    output << " ";
    // FIXME: handle initialization at operation level
    output << "(" << watType << ".const " << 0 << "))\n";
  });
  return success();
}

LogicalResult addLibc(ModuleOp &moduleOp, raw_ostream &output,
                      bool hasMemoryOp) {
  if (hasMemoryOp) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
        llvm::MemoryBuffer::getFile("wasm-translate/libc.wat");
    if (std::error_code EC = FileOrErr.getError()) {
      llvm::errs() << "Error opening file: " << EC.message() << "\n";
      return failure();
    }
    std::unique_ptr<llvm::MemoryBuffer> &FileBuffer = FileOrErr.get();
    output << FileBuffer->getBuffer();
  }

  return success();
}

LogicalResult translateModuleToWat(ModuleOp module, raw_ostream &output,
                                   bool addDebugFunctions) {
  FuncSignatureList funcSignatureList =
      initializeFuncSignatureList(module, addDebugFunctions);

  bool hasMemoryOp = false;

  module.walk([&](Operation *op) {
    if (isa<LoadOp>(op) || isa<StoreOp>(op)) {
      hasMemoryOp = true;
    }
  });

  output << "(module\n";

  if (failed(translateFunctionSignatures(funcSignatureList, output))) {
    return failure();
  }

  if (failed(translateRecContFuncDeclOps(module, output))) {
    module.emitError(
        "failed to translate recursive continuation function declarations");
    return failure();
  }

  if (failed(translateImportOps(module, output, funcSignatureList,
                                addDebugFunctions))) {
    module.emitError("failed to translate imports");
    return failure();
  }

  // DEPRECATED. TODO: remove
  if (failed(translateContinuationTypeDeclOps(module, output))) {
    module.emitError("failed to translate continuation types");
    return failure();
  }

  if (failed(translateTagOps(module, output))) {
    module.emitError("failed to translate tags");
    return failure();
  }

  if (failed(translateTableOps(module, output))) {
    module.emitError("failed to translate tables");
    return failure();
  }

  if (failed(translateGlobalOps(module, output))) {
    module.emitError("failed to translate globals");
    return failure();
  }

  // define malloc and free
  if (failed(addLibc(module, output, hasMemoryOp))) {
    module.emitError("failed to add libc");
    return failure();
  }

  // translate each function
  for (auto funcOp : module.getOps<WasmFuncOp>()) {
    if (failed(translateFunction(funcSignatureList, funcOp, output))) {
      funcOp.emitError("failed to translate WasmFuncOp");
      return failure();
    }
  }

  // translate elem declare func
  for (auto elemDeclareFuncOp : module.getOps<ElemDeclareFuncOp>()) {
    if (failed(translateElemDeclareFuncOp(elemDeclareFuncOp, output))) {
      elemDeclareFuncOp.emitError("failed to translate ElemDeclareFuncOp");
      return failure();
    }
    output << "\n";
  }

  for (auto dataOp : module.getOps<DataOp>()) {
    if (failed(translateData(dataOp, output))) {
      dataOp.emitError("failed to translate DataOp");
      return failure();
    }
  }

  // FIXME: Do not hardcode the memory size
  output << R""""(
    (memory (;0;) 3)
  )"""";

  // export memory, main, malloc, and free
  output << R""""(
  (export "memory" (memory 0))
  (export "main" (func $main))
  )"""";

  if (hasMemoryOp) {
    output << R""""(
    (export "malloc" (func $malloc))
    (export "free" (func $free))
    )"""";
  }

  output << ")\n";
  return success();

  output << ")\n";
  return success();
}
} // namespace mlir::wasm

using namespace mlir;

static llvm::cl::opt<bool>
    addDebugFunctions("add-debug-functions",
                      llvm::cl::desc("Add debug functions log_i32 and log_f32"),
                      llvm::cl::init(false)); // Default value is false

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration registration(
      "mlir-to-wat", "translate from mlir wasm dialect to wat",
      [](ModuleOp module, raw_ostream &output) {
        return mlir::wasm::translateModuleToWat(module, output,
                                                addDebugFunctions);
      },
      [](DialectRegistry &registry) { registry.insert<wasm::WasmDialect>(); });

  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Tool"));

} // namespace mlir