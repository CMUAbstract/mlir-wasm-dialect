//===- IntermittentPasses.h - Intermittent passes  ------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef INTERMITTENT_INTERMITTENTPASSES_H
#define INTERMITTENT_INTERMITTENTPASSES_H

#include "Intermittent/IntermittentDialect.h"
#include "Intermittent/IntermittentOps.h"
#include "Wasm/WasmDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::intermittent {
#define GEN_PASS_DECL
#include "Intermittent/IntermittentPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Intermittent/IntermittentPasses.h.inc"
} // namespace mlir::intermittent

#endif
