//===- StackificationPlan.h - Stackification planning state -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines shared planning state used by stackification analysis and
// local materialization.
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_STACKIFICATIONPLAN_H
#define WASMSTACK_STACKIFICATIONPLAN_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::wasmstack {

/// Stable plan describing which SSA values must be materialized as locals
/// and which can use tee semantics.
struct StackificationPlan {
  DenseSet<Value> needsLocal;
  DenseSet<Value> needsTee;

  /// Deterministic insertion order for local allocation.
  SmallVector<Value> localOrder;
  SmallVector<Value> teeOrder;

  /// Require local-only materialization.
  void requireLocal(Value v) {
    needsTee.erase(v);
    if (needsLocal.insert(v).second)
      localOrder.push_back(v);
  }

  /// Require tee materialization, unless the value is already local-only.
  void requireTee(Value v) {
    if (needsLocal.contains(v))
      return;
    if (needsTee.insert(v).second)
      teeOrder.push_back(v);
  }

  bool isLocal(Value v) const { return needsLocal.contains(v); }
  bool isTee(Value v) const { return needsTee.contains(v); }

  ArrayRef<Value> getLocalOrder() const { return localOrder; }
  ArrayRef<Value> getTeeOrder() const { return teeOrder; }
};

} // namespace mlir::wasmstack

#endif // WASMSTACK_STACKIFICATIONPLAN_H
