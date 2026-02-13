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
#include "WAMI/WAMIAttrs.h"
#include "WAMI/WAMIOps.h"
#include "WAMI/WAMITypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::wami;

#define GET_ATTRDEF_CLASSES
#include "WAMI/WAMIAttrs.cpp.inc"

#include "WAMI/WAMIDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// WAMI dialect
//===----------------------------------------------------------------------===//

void WAMIDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "WAMI/WAMIAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "WAMI/WAMIOps.cpp.inc"
      >();
  registerTypes();
}
