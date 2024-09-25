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
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

LogicalResult translateModuleToWat(ModuleOp module, raw_ostream &output) {
  // TODO
  return success();
}

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration registration(
      "mlir-to-wat", "translate from mlir wasm dialect to wat",
      [](ModuleOp module, raw_ostream &output) {
        return translateModuleToWat(module, output);
      },
      [](DialectRegistry &registry) { registry.insert<wasm::WasmDialect>(); });

  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Tool"));

} // namespace mlir