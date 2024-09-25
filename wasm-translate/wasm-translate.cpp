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

llvm::LogicalResult translateOperation(Operation *op, raw_ostream &output) {
  if (auto addOp = dyn_cast<AddOp>(op)) {
    output << "(i32.add ";
    // Translate operands
    for (Value operand : op->getOperands()) {
      if (failed(translateOperand(operand, output))) {
        return failure();
      }
      output << " ";
    }
    output << ")";
  }
  // Handle other operations similarly
  else if (isa<WasmReturnOp>(op)) {
    output << "(return)";
  } else {
    // Unsupported operation
    return failure();
  }
  output << "\n    ";
  return success();
}

llvm::LogicalResult translateFunctionBody(WasmFuncOp funcOp,
                                          raw_ostream &output) {
  output << "\n    "; // Indentation for readability
  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translateOperation(&op, output))) {
        return failure();
      }
    }
  }
  return success();
}

llvm::LogicalResult translateData(DataOp dataOp, raw_ostream &output) {
  output << "(data $" << dataOp.getSymName() << " (i32.const "
         << dataOp.getOffset() << ") \"";
  output << dataOp.getBytes();
  output << "\")\n";
  return success();
}

llvm::LogicalResult translateFunction(WasmFuncOp funcOp, raw_ostream &output) {
  // Start function declaration
  output << "  (func $" << funcOp.getName() << " ";

  // TODO: add function type

  // Translate function arguments
  if (failed(translateFunctionArguments(funcOp, output))) {
    funcOp.emitError("translating function arguments failed");
    return failure();
  }

  // Translate function result types
  if (failed(translateFunctionResults(funcOp, output))) {
    funcOp.emitError("translating function results failed");
    return failure();
  }

  // Translate function body
  if (failed(translateFunctionBody(funcOp, output))) {
    funcOp.emitError("translating function body failed");
    return failure();
  }

  // Close function
  output << "  )\n";
  return success();
}

LogicalResult translateModuleToWat(ModuleOp module, raw_ostream &output) {
  output << "(module\n";

  for (Operation &op : module.getOps()) {
    if (auto funcOp = dyn_cast<WasmFuncOp>(&op)) {
      if (failed(translateFunction(funcOp, output))) {
        return failure();
      }
    }
    if (auto dataOp = dyn_cast<DataOp>(&op)) {
      if (failed(translateData(dataOp, output))) {
        return failure();
      }
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