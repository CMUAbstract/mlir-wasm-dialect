//===- WasmTypes.cpp - Wasm dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Wasm/WasmTypes.h"

#include "Wasm/WasmDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::wasm;

#define GET_TYPEDEF_CLASSES
#include "Wasm/WasmTypes.cpp.inc"

void WasmDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Wasm/WasmTypes.cpp.inc"
      >();
}
