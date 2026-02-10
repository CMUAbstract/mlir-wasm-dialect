//===- wasm-emit.cpp - WasmStack to WebAssembly binary ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates WasmStack MLIR to WebAssembly
// binary format.
//
//===----------------------------------------------------------------------===//

#include "Target/WasmStack/WasmBinaryEmitter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackOps.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration registration(
      "mlir-to-wasm", "emit WasmStack dialect to WebAssembly binary",
      [](ModuleOp module, raw_ostream &output) {
        return wasmstack::emitWasmBinary(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<wasmstack::WasmStackDialect>();
      });

  return failed(mlirTranslateMain(argc, argv, "WasmStack Binary Emitter"));
}
