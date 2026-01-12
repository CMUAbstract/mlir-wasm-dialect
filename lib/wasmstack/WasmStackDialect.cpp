//===- WasmStackDialect.cpp - WasmStack dialect implementation ----*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackOps.h"
#include "wasmstack/WasmStackTypes.h"

using namespace mlir;
using namespace mlir::wasmstack;

#include "wasmstack/WasmStackDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// WasmStack dialect initialization
//===----------------------------------------------------------------------===//

void WasmStackDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "wasmstack/WasmStackOps.cpp.inc"
      >();
}
