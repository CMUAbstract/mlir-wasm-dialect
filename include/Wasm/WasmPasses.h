//===- WasmPasses.h - Wasm passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef WASM_WASMPASSES_H
#define WASM_WASMPASSES_H

#include "Wasm/WasmDialect.h"
#include "Wasm/WasmOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace wasm {
#define GEN_PASS_DECL
#include "Wasm/WasmPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Wasm/WasmPasses.h.inc"
} // namespace wasm
} // namespace mlir

#endif
