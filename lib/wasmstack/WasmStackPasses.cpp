//===- WasmStackPasses.cpp - WasmStack dialect passes -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the stackification pass that converts WasmSSA+WAMI
// dialects to the WasmStack dialect using LLVM-style stackification.
//
//===----------------------------------------------------------------------===//

#include "wasmstack/WasmStackPasses.h"
#include "WAMI/WAMIDialect.h"
#include "WAMI/WAMIOps.h"
#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackOps.h"

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::wasmstack {

#define GEN_PASS_DEF_CONVERTTOWASMSTACK
#include "wasmstack/WasmStackPasses.h.inc"

//===----------------------------------------------------------------------===//
// Dependency Analysis
//===----------------------------------------------------------------------===//

/// Query result for operation dependencies
struct DepInfo {
  bool readsMemory = false;
  bool writesMemory = false;
  bool hasSideEffects = false;
  // Note: WebAssembly has no volatile semantics - all memory accesses
  // can be reordered unless they have data dependencies
};

/// Analyze an operation for its memory and side effect dependencies
static DepInfo queryDependencies(Operation *op) {
  DepInfo info;

  // Memory loads
  if (isa<wami::LoadOp>(op)) {
    info.readsMemory = true;
  }

  // Memory stores - both write memory AND have side effects
  // (order of stores must be preserved)
  if (isa<wami::StoreOp>(op)) {
    info.writesMemory = true;
    info.hasSideEffects = true;
  }

  // Global reads - treat like memory reads for ordering purposes
  if (isa<wasmssa::GlobalGetOp>(op)) {
    info.readsMemory = true;
  }

  // Note: WasmSSA doesn't have GlobalSetOp - mutable globals are
  // handled differently. If added later, it would have side effects.

  // Calls have side effects and may read/write memory
  if (isa<wasmssa::FuncCallOp>(op)) {
    info.hasSideEffects = true;
    info.readsMemory = true;
    info.writesMemory = true;
  }

  // Local operations don't have side effects - they're SSA values
  // LocalGetOp, LocalSetOp, LocalTeeOp are fine to reorder

  // TODO: Add stack switching operations when implemented
  // suspend, resume, switch all have significant side effects

  return info;
}

/// Check if it's safe to move defOp to immediately before insertBefore
static bool isSafeToMove(Operation *defOp, Operation *insertBefore) {
  // Can't move across basic block boundaries
  if (defOp->getBlock() != insertBefore->getBlock())
    return false;

  DepInfo defDeps = queryDependencies(defOp);

  // Walk operations between defOp and insertBefore
  for (Operation *op = defOp->getNextNode(); op != insertBefore;
       op = op->getNextNode()) {
    if (!op)
      return false; // Reached end of block without finding insertBefore

    DepInfo opDeps = queryDependencies(op);

    // Check memory dependencies
    if (defDeps.readsMemory && opDeps.writesMemory)
      return false; // Read-after-write hazard
    if (defDeps.writesMemory && opDeps.readsMemory)
      return false; // Write-after-read hazard
    if (defDeps.writesMemory && opDeps.writesMemory)
      return false; // Write-after-write hazard

    // Check side effects (calls, traps, etc.)
    if (defDeps.hasSideEffects || opDeps.hasSideEffects)
      return false;
  }
  return true;
}

/// Check if an operation should be rematerialized instead of using a local
static bool shouldRematerialize(Operation *op) {
  // Constants are always cheap to rematerialize
  if (isa<wasmssa::ConstOp>(op))
    return true;

  // Local.get is cheap (in the source dialect)
  if (isa<wasmssa::LocalGetOp>(op))
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class WasmStackTypeConverter : public TypeConverter {
public:
  WasmStackTypeConverter(MLIRContext *ctx) {
    // Value types pass through unchanged
    addConversion([](IntegerType t) { return t; });
    addConversion([](FloatType t) { return t; });

    // MemRef types convert to i32 (linear memory pointer)
    addConversion(
        [ctx](MemRefType t) -> Type { return IntegerType::get(ctx, 32); });

    // Function types pass through
    addConversion([](FunctionType t) { return t; });

    // Index type converts to i32
    addConversion(
        [ctx](IndexType t) -> Type { return IntegerType::get(ctx, 32); });
  }
};

//===----------------------------------------------------------------------===//
// ConvertToWasmStack Pass
//===----------------------------------------------------------------------===//

class ConvertToWasmStack
    : public impl::ConvertToWasmStackBase<ConvertToWasmStack> {
public:
  using impl::ConvertToWasmStackBase<
      ConvertToWasmStack>::ConvertToWasmStackBase;

  void runOnOperation() final {
    auto module = getOperation();
    (void)module.getContext(); // Will be used in full implementation

    // For now, just mark the pass as successful
    // TODO: Implement the full stackification algorithm

    llvm::errs() << "ConvertToWasmStack pass running on module\n";

    // Walk all WasmSSA functions and analyze them
    module.walk([&](wasmssa::FuncOp funcOp) {
      llvm::errs() << "  Processing function: " << funcOp.getName() << "\n";
    });
  }
};

} // namespace mlir::wasmstack
