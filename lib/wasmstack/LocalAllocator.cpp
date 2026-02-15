//===- LocalAllocator.cpp - Local variable allocation -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LocalAllocator class.
//
//===----------------------------------------------------------------------===//

#include "wasmstack/LocalAllocator.h"
#include "WAMI/WAMIOps.h"
#include "WAMI/WAMITypes.h"
#include "wasmstack/WasmStackTypes.h"

namespace mlir::wasmstack {

Type LocalAllocator::unwrapLocalRefType(Type type) {
  // Check if this is a LocalRefType (!wasmssa<local ref to T>)
  if (auto localRefType = dyn_cast<wasmssa::LocalRefType>(type)) {
    return localRefType.getElementType();
  }
  return type;
}

Type LocalAllocator::normalizeType(Type type) {
  if (auto cont = dyn_cast<wami::ContType>(type)) {
    if (cont.getNullable())
      return ContRefType::get(type.getContext(), cont.getTypeName());
    return ContRefNonNullType::get(type.getContext(), cont.getTypeName());
  }
  if (auto func = dyn_cast<wami::FuncRefType>(type)) {
    return FuncRefType::get(type.getContext(), func.getFuncName());
  }
  return type;
}

Type LocalAllocator::normalizeValueType(Value value) {
  return normalizeType(unwrapLocalRefType(value.getType()));
}

void LocalAllocator::allocate(wasmssa::FuncOp funcOp,
                              ArrayRef<Value> needsLocalOrdered,
                              ArrayRef<Value> needsTeeOrdered) {
  // First, assign indices to parameters
  // WasmSSA functions use !wasmssa<local ref to T> for parameters
  // The block arguments in the entry block are the parameters
  if (!funcOp.getBody().empty()) {
    Block &entryBlock = funcOp.getBody().front();
    for (BlockArgument arg : entryBlock.getArguments()) {
      localIndices[arg] = numParams;
      // Unwrap local ref types to get the underlying value type
      Type unwrappedType = unwrapLocalRefType(arg.getType());
      localTypes.push_back(normalizeType(unwrappedType));
      numParams++;
    }
  }

  // Then, assign indices to introduced locals
  unsigned nextLocalIdx = numParams;

  // Assign indices to values that need full locals (local.set/local.get)
  for (Value value : needsLocalOrdered) {
    if (!localIndices.count(value)) {
      localIndices[value] = nextLocalIdx++;
      localTypes.push_back(normalizeValueType(value));
    }
  }

  // Assign indices to values that use tee (local.tee + local.get)
  for (Value value : needsTeeOrdered) {
    if (!localIndices.count(value)) {
      localIndices[value] = nextLocalIdx++;
      localTypes.push_back(normalizeValueType(value));
    }
  }
}

int LocalAllocator::getLocalIndex(Value value) const {
  auto it = localIndices.find(value);
  if (it != localIndices.end())
    return it->second;
  return -1;
}

} // namespace mlir::wasmstack
