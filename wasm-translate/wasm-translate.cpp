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
#include "llvm/Support/raw_ostream.h"

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
    value = std::to_string(floatAttr.getValueAsDouble());
  } else {
    return failure();
  }
  return success();
}

llvm::LogicalResult translateFunctionArguments(WasmFuncOp funcOp,
                                               raw_ostream &output) {
  auto functionType = funcOp.getFunctionType();
  unsigned numInputs = functionType.getNumInputs();
  for (unsigned i = 0; i < numInputs; ++i) {
    Type inputType = functionType.getInput(i);
    std::string watType;
    if (failed(getWatType(inputType, watType))) {
      return failure();
    }
    output << "(param $p" << i << " " << watType << ") ";
  }
  return success();
}

llvm::LogicalResult translateFunctionResults(WasmFuncOp funcOp,
                                             raw_ostream &output) {
  auto functionType = funcOp.getFunctionType();
  unsigned numResults = functionType.getNumResults();
  for (unsigned i = 0; i < numResults; ++i) {
    Type resultType = functionType.getResult(i);
    std::string watType;
    if (failed(getWatType(resultType, watType))) {
      return failure();
    }
    output << "(result " << watType << ") ";
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
  bool first = true;
  for (Attribute attr : localOp.getTypes()) {
    TypeAttr typeAttr = cast<TypeAttr>(attr);
    std::string watType;
    if (failed(getWatType(typeAttr.getValue(), watType))) {
      localOp.emitError("unsupported local type");
      return failure();
    }
    if (!first) {
      output << " ";
    } else {
      first = false;
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

llvm::LogicalResult translateReturnOp(WasmReturnOp returnOp,
                                      raw_ostream &output) {
  output << "return";
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
  } else if (auto returnOp = dyn_cast<WasmReturnOp>(op)) {
    return translateReturnOp(returnOp, output);
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

llvm::LogicalResult translateFunction(WasmFuncOp funcOp, raw_ostream &output) {
  // Start function declaration
  output << "  (func $" << funcOp.getName() << " ";

  // TODO: translate function type

  // Translate function arguments
  // if (failed(translateFunctionArguments(funcOp, output))) {
  //  funcOp.emitError("translating function arguments failed");
  //  return failure();
  //}

  // Translate function result types
  // if (failed(translateFunctionResults(funcOp, output))) {
  //  funcOp.emitError("translating function results failed");
  //  return failure();
  //}

  // Translate function body
  if (failed(translateFunctionBody(funcOp, output))) {
    funcOp.emitError("translating function body failed");
    return failure();
  }

  // Close function
  output << ")\n";
  return success();
}

LogicalResult translateModuleToWat(ModuleOp module, raw_ostream &output) {
  output << "(module\n";

  for (auto funcOp : module.getOps<WasmFuncOp>()) {
    if (failed(translateFunction(funcOp, output))) {
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