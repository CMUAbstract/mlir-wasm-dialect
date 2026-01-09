//===- WAMITypeConverter.h - WAMI type converter ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type converter for converting MLIR types to
// WebAssembly-compatible types used by the WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_WAMITYPECONVERTER_H
#define WAMI_WAMITYPECONVERTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wami {

/// Type converter for converting MLIR types to WasmSSA-compatible types.
/// WebAssembly only supports i32, i64, f32, f64 as numeric types.
class WAMITypeConverter : public TypeConverter {
public:
  WAMITypeConverter(MLIRContext *ctx) {
    // Identity conversion for types that don't need conversion
    addConversion([](Type type) { return type; });

    // Integer types: WebAssembly only supports i32 and i64
    addConversion([](IntegerType type) -> Type {
      unsigned width = type.getWidth();
      if (width <= 32)
        return IntegerType::get(type.getContext(), 32);
      return IntegerType::get(type.getContext(), 64);
    });

    // Index type is converted to i32 (32-bit addressing in Wasm32)
    addConversion(
        [ctx](IndexType type) -> Type { return IntegerType::get(ctx, 32); });

    // Source materialization: convert back to original type when needed
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    });

    // Target materialization: convert to target type when needed
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    });
  }
};

} // namespace mlir::wami

#endif // WAMI_WAMITYPECONVERTER_H
