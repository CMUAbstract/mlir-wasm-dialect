//===- WAMIPasses.h - WAMI dialect passes -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes for the WAMI dialect and conversion passes
// to/from the upstream WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_WAMIPASSES_H
#define WAMI_WAMIPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::wami {

#define GEN_PASS_DECL
#include "WAMI/WAMIPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "WAMI/WAMIPasses.h.inc"

/// Registers all WAMI passes.
inline void registerPasses() { registerWAMIPasses(); }

} // namespace mlir::wami

#endif // WAMI_WAMIPASSES_H
