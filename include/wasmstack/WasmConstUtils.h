//===- WasmConstUtils.h - WasmStack constant emission ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for emitting WasmStack constant operations.
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_WASMCONSTUTILS_H
#define WASMSTACK_WASMCONSTUTILS_H

#include "wasmstack/WasmStackOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::wasmstack {

/// Emit a WasmStack constant op from a typed attribute.
/// Supports i32/i64/f32/f64 constants.
inline LogicalResult emitWasmStackConst(OpBuilder &builder, Location loc,
                                        TypedAttr valueAttr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    unsigned bitWidth = intAttr.getType().getIntOrFloatBitWidth();
    if (bitWidth == 32) {
      wasmstack::I32ConstOp::create(
          builder, loc,
          builder.getI32IntegerAttr(static_cast<int32_t>(intAttr.getInt())));
      return success();
    }
    if (bitWidth == 64) {
      wasmstack::I64ConstOp::create(
          builder, loc, builder.getI64IntegerAttr(intAttr.getInt()));
      return success();
    }
    return failure();
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    if (floatAttr.getType().isF32()) {
      wasmstack::F32ConstOp::create(builder, loc,
                                    builder.getF32FloatAttr(static_cast<float>(
                                        floatAttr.getValueAsDouble())));
      return success();
    }
    if (floatAttr.getType().isF64()) {
      wasmstack::F64ConstOp::create(
          builder, loc, builder.getF64FloatAttr(floatAttr.getValueAsDouble()));
      return success();
    }
    return failure();
  }

  return failure();
}

} // namespace mlir::wasmstack

#endif // WASMSTACK_WASMCONSTUTILS_H
