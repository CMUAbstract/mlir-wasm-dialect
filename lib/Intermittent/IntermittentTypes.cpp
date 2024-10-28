//===- IntermittentTypes.cpp - Wasm dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Intermittent/IntermittentTypes.h"

#include "Intermittent/IntermittentDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::intermittent;

#define GET_TYPEDEF_CLASSES
#include "Intermittent/IntermittentTypes.cpp.inc"

void IntermittentDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Intermittent/IntermittentTypes.cpp.inc"
      >();
}
