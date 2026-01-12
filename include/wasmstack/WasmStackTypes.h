//===- WasmStackTypes.h - WasmStack dialect types ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_WASMSTACKTYPES_H
#define WASMSTACK_WASMSTACKTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "wasmstack/WasmStackTypes.h.inc"

#endif // WASMSTACK_WASMSTACKTYPES_H
