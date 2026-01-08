//===- WAMIDialect.h - WAMI dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the WAMI dialect, which extends the upstream WasmSSA
// dialect with WebAssembly proposal features such as stack switching and
// continuations. The WAMI dialect operates at the same SSA abstraction level
// as WasmSSA, and both dialects lower together to stack-based WebAssembly.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_WAMIDIALECT_H
#define WAMI_WAMIDIALECT_H

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Dialect.h"

#include "WAMI/WAMIDialect.h.inc"

#endif // WAMI_WAMIDIALECT_H
