//===- WAMITypes.h - WAMI dialect types -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the WAMI dialect. These types extend the
// WebAssembly type system with support for continuations and other features
// from WebAssembly proposals.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_WAMITYPES_H
#define WAMI_WAMITYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "WAMI/WAMITypes.h.inc"

#endif // WAMI_WAMITYPES_H
