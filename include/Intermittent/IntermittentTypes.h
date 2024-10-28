//===- IntermittentTypes.h - Wasm dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INTERMITTENT_INTERMITTENTTYPES_H
#define INTERMITTENT_INTERMITTENTTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Wasm/WasmTypes.h.inc"

#endif // INTERMITTENT_INTERMITTENTTYPES_H