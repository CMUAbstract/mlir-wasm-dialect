//===- IntermittentDialect.cpp - Intermittent dialect ---------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Intermittent/IntermittentDialect.h"
#include "Intermittent/IntermittentOps.h"
#include "Intermittent/IntermittentTypes.h"

using namespace mlir;
using namespace mlir::intermittent;

#include "Intermittent/IntermittentDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Intermittent dialect.
//===----------------------------------------------------------------------===//

void IntermittentDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Intermittent/IntermittentOps.cpp.inc"
      >();
  registerTypes();
}
