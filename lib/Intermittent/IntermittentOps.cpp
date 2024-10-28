//===- IntermittentOps.cpp - Intermittent dialect ops ---------------*- C++
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

#define GET_OP_CLASSES
#include "Intermittent/IntermittentOps.cpp.inc"

namespace mlir::intermittent {} // namespace mlir::intermittent
