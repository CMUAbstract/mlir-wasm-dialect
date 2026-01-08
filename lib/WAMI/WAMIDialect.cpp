//===- WAMIDialect.cpp - WAMI dialect implementation ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the WAMI dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/WAMIDialect.h"
#include "WAMI/WAMIOps.h"
#include "WAMI/WAMITypes.h"

using namespace mlir;
using namespace mlir::wami;

#include "WAMI/WAMIDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// WAMI dialect
//===----------------------------------------------------------------------===//

void WAMIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "WAMI/WAMIOps.cpp.inc"
      >();
  registerTypes();
}
