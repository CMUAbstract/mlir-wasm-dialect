//===- DContPasses.h - DCont passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef DCONT_DCONT_PASSES_H
#define DCONT_DCONT_PASSES_H

#include "DCont/DContDialect.h"
#include "DCont/DContOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::dcont {
#define GEN_PASS_DECL
#include "DCont/DContPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "DCont/DContPasses.h.inc"
} // namespace mlir::dcont

#endif
