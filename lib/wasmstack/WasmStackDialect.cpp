//===- WasmStackDialect.cpp - WasmStack dialect implementation ----*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wasmstack/WasmStackDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "wasmstack/WasmStackAttrs.h"
#include "wasmstack/WasmStackOps.h"
#include "wasmstack/WasmStackTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::wasmstack;

#define GET_ATTRDEF_CLASSES
#include "wasmstack/WasmStackAttrs.cpp.inc"

#include "wasmstack/WasmStackDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// WasmStack dialect initialization
//===----------------------------------------------------------------------===//

void WasmStackDialect::initialize() {
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "wasmstack/WasmStackAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "wasmstack/WasmStackOps.cpp.inc"
      >();
}
