//===- SsaWasmPasses.h - SsaWasm passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SSAWASM_SSAWASMPASSES_H
#define SSAWASM_SSAWASMPASSES_H

#include "SsaWasm/SsaWasmDialect.h"
#include "SsaWasm/SsaWasmOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::ssawasm {
#define GEN_PASS_DECL
#include "SsaWasm/SsaWasmPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "SsaWasm/SsaWasmPasses.h.inc"
} // namespace mlir::ssawasm

#endif
