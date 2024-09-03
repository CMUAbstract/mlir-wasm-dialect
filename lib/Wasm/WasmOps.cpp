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

void mlir::wasm::TempLocalOp::build(OpBuilder &builder, OperationState &state,
                                    mlir::Type inner) {
  auto context = inner.getContext();
  auto localType = mlir::wasm::LocalType::get(context, inner);
  state.addTypes(localType);
  state.addAttribute("type", mlir::TypeAttr::get(inner));
}

void mlir::wasm::LoopOp::build(OpBuilder &builder, OperationState &state) {
  state.addRegion();
}