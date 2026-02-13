//===- WAMIOps.h - WAMI dialect operations ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the WAMI dialect. These operations
// extend WebAssembly with support for continuations and other features from
// WebAssembly proposals such as stack switching.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_WAMIOPS_H
#define WAMI_WAMIOPS_H

#include "WAMI/WAMIAttrs.h"
#include "WAMI/WAMIDialect.h"
#include "WAMI/WAMITypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "WAMI/WAMIOps.h.inc"

#endif // WAMI_WAMIOPS_H
