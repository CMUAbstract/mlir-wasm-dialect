//===- WasmStackTypes.cpp - WasmStack dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wasmstack/WasmStackTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "wasmstack/WasmStackDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::wasmstack;

#define GET_TYPEDEF_CLASSES
#include "wasmstack/WasmStackTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

void WasmStackDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "wasmstack/WasmStackTypes.cpp.inc"
      >();
}
