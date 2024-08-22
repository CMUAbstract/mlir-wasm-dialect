//===- WasmOps.cpp - Wasm dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Wasm/WasmOps.h"
#include "Wasm/WasmDialect.h"
#include "Wasm/WasmTypes.h"

#define GET_OP_CLASSES
#include "Wasm/WasmOps.cpp.inc"

llvm::LogicalResult mlir::wasm::ConstantOp::verify() {
  // TODO: Value must be either of i32, i64, f32, or f64 attribute.
  if (!llvm::isa<IntegerAttr, FloatAttr>(getValue())) {
    return emitOpError(
        "value must be either of i32, i64, f32, or f64 attribute");
  }
  return success();
}