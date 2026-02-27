//===- CoroPasses.h - Coro passes -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes for the coroutine intrinsic ABI.
//
//===----------------------------------------------------------------------===//

#ifndef CORO_COROPASSES_H
#define CORO_COROPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::coro {

#define GEN_PASS_DECL
#include "Coro/CoroPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Coro/CoroPasses.h.inc"

/// Registers all Coro passes.
inline void registerPasses() { registerCoroPasses(); }

} // namespace mlir::coro

#endif // CORO_COROPASSES_H
