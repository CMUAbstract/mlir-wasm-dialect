//===- SsaWasmTypes.cpp - SsaWasm dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SsaWasm/SsaWasmTypes.h"

#include "SsaWasm/SsaWasmDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::ssawasm;

#define GET_TYPEDEF_CLASSES
#include "SsaWasm/SsaWasmTypes.cpp.inc"

void SsaWasmDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SsaWasm/SsaWasmTypes.cpp.inc"
      >();
}
