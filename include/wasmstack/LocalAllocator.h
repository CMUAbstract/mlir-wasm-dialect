//===- LocalAllocator.h - Local variable allocation -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the LocalAllocator class which allocates local indices
// to values that need locals during stackification.
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_LOCALALLOCATOR_H
#define WASMSTACK_LOCALALLOCATOR_H

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::wasmstack {

/// Allocates local indices to values that need locals
/// Parameters: indices 0..N-1
/// Introduced locals: indices N..M
class LocalAllocator {
  /// Mapping from Value to local index
  DenseMap<Value, unsigned> localIndices;

  /// Type of each local (indexed by local index)
  SmallVector<Type> localTypes;

  /// Number of function parameters
  unsigned numParams = 0;

public:
  /// Unwrap a WasmSSA local ref type to get the underlying value type
  static Type unwrapLocalRefType(Type type);

  /// Convert source dialect reference types to wasmstack reference types.
  static Type normalizeType(Type type);

  /// Allocate locals for a function
  void allocate(wasmssa::FuncOp funcOp, ArrayRef<Value> needsLocalOrdered,
                ArrayRef<Value> needsTeeOrdered);

  /// Get the local index for a value, or -1 if not allocated
  int getLocalIndex(Value value) const;

  /// Check if a value has an allocated local
  bool hasLocal(Value value) const { return localIndices.count(value); }

  /// Get the number of parameters
  unsigned getNumParams() const { return numParams; }

  /// Get the total number of locals (including parameters)
  unsigned getNumLocals() const { return localTypes.size(); }

  /// Get the type of a local by index
  Type getLocalType(unsigned idx) const { return localTypes[idx]; }

  /// Get all local types
  ArrayRef<Type> getLocalTypes() const { return localTypes; }
};

} // namespace mlir::wasmstack

#endif // WASMSTACK_LOCALALLOCATOR_H
