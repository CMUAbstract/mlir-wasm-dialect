//===- WAMITypes.cpp - WAMI dialect types -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the types for the WAMI dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/WAMITypes.h"

#include "WAMI/WAMIDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::wami;

#define GET_TYPEDEF_CLASSES
#include "WAMI/WAMITypes.cpp.inc"

void WAMIDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "WAMI/WAMITypes.cpp.inc"
      >();
}
