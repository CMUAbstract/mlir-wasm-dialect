//===- DContTypes.cpp - DCont dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DCont/DContTypes.h"

#include "DCont/DContDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::dcont;

#define GET_TYPEDEF_CLASSES
#include "DCont/DContTypes.cpp.inc"

void DContDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "DCont/DContTypes.cpp.inc"
      >();
}
