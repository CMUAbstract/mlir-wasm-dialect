//===- DContDialect.cpp - DCont dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DCont/DContDialect.h"
#include "DCont/DContOps.h"
#include "DCont/DContTypes.h"

using namespace mlir;
using namespace mlir::dcont;

#include "DCont/DContDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// DCont dialect.
//===----------------------------------------------------------------------===//

void DContDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DCont/DContOps.cpp.inc"
      >();
  registerTypes();
}
