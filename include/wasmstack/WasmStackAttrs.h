//===- WasmStackAttrs.h - WasmStack dialect attributes ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_WASMSTACKATTRS_H
#define WASMSTACK_WASMSTACKATTRS_H

#include "mlir/IR/Attributes.h"
#include "wasmstack/WasmStackDialect.h"

#define GET_ATTRDEF_CLASSES
#include "wasmstack/WasmStackAttrs.h.inc"

#endif // WASMSTACK_WASMSTACKATTRS_H
