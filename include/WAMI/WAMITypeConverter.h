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

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
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

    // Source materialization: convert from converted types back to source types
    // This handles LocalRefType -> value type conversion via local_get
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;

      Value input = inputs[0];
      Type inputType = input.getType();

      // If input is a LocalRefType and we need a value type, insert local_get
      if (auto localRefType = dyn_cast<wasmssa::LocalRefType>(inputType)) {
        Type innerType = localRefType.getElementType();

        // Case 1: innerType exactly matches requested type - use local_get
        if (innerType == type) {
          return wasmssa::LocalGetOp::create(builder, loc, input);
        }

        // Case 2: innerType is the converted form of requested type
        // e.g., LocalRefType(i32) -> i1, where i1 converts to i32
        // Check if type is a small integer that would convert to innerType
        if (auto intType = dyn_cast<IntegerType>(type)) {
          Type expectedConverted = IntegerType::get(
              type.getContext(), intType.getWidth() <= 32 ? 32 : 64);
          if (innerType == expectedConverted) {
            // Extract via local_get, then create a reconcilable cast
            Value extracted = wasmssa::LocalGetOp::create(builder, loc, input);
            return UnrealizedConversionCastOp::create(builder, loc, type,
                                                      extracted)
                .getResult(0);
          }
        }
      }

      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    });

    // Target materialization: convert to target type when needed
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;

      Value input = inputs[0];
      Type inputType = input.getType();

      // If input is a LocalRefType and we need a value type, insert local_get
      if (auto localRefType = dyn_cast<wasmssa::LocalRefType>(inputType)) {
        Type innerType = localRefType.getElementType();

        // Case 1: innerType exactly matches requested type - use local_get
        if (innerType == type) {
          return wasmssa::LocalGetOp::create(builder, loc, input);
        }

        // Case 2: innerType is the converted form of requested type
        // Check if type is a small integer that would convert to innerType
        if (auto intType = dyn_cast<IntegerType>(type)) {
          Type expectedConverted = IntegerType::get(
              type.getContext(), intType.getWidth() <= 32 ? 32 : 64);
          if (innerType == expectedConverted) {
            Value extracted = wasmssa::LocalGetOp::create(builder, loc, input);
            return UnrealizedConversionCastOp::create(builder, loc, type,
                                                      extracted)
                .getResult(0);
          }
        }
      }

      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    });
  }
};

} // namespace mlir::wami

#endif // WAMI_WAMITYPECONVERTER_H
