//===- WasmStackOps.h - WasmStack dialect operations ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_WASMSTACKOPS_H
#define WASMSTACK_WASMSTACKOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "wasmstack/WasmStackAttrs.h"
#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackTypes.h"

// Include generated interface declarations
#include "wasmstack/WasmStackInterfaces.h.inc"

#define GET_OP_CLASSES
#include "wasmstack/WasmStackOps.h.inc"

#endif // WASMSTACK_WASMSTACKOPS_H
