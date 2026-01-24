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
#include "wasmstack/LocalAllocator.h"
#include "wasmstack/StackificationAnalysis.h"
#include "wasmstack/TreeWalker.h"
#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackEmitter.h"
#include "wasmstack/WasmStackOps.h"

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wasmstack {

#define GEN_PASS_DEF_CONVERTTOWASMSTACK
#include "wasmstack/WasmStackPasses.h.inc"

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
    MLIRContext *ctx = module.getContext();

    llvm::errs() << "ConvertToWasmStack pass running on module\n";
    llvm::errs().flush();

    // Collect functions to process
    SmallVector<wasmssa::FuncOp> funcsToConvert;
    module.walk(
        [&](wasmssa::FuncOp funcOp) { funcsToConvert.push_back(funcOp); });

    // Create builder for emitting new operations
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    // Process each WasmSSA function
    for (wasmssa::FuncOp funcOp : funcsToConvert) {
      // Skip empty functions
      if (funcOp.getBody().empty())
        continue;

      // Analyze use counts for operation results in the function.
      // UseCountAnalysis recursively analyzes nested regions (loops, ifs, etc.)
      Block &entryBlock = funcOp.getBody().front();
      UseCountAnalysis useCount(entryBlock);

      // Run the TreeWalker to stackify operations
      TreeWalker walker(useCount);
      walker.processBlock(entryBlock);

      // Get values that need locals or tee
      DenseSet<Value> needsLocal = walker.getValuesNeedingLocals();
      const auto &needsTee = walker.getValuesNeedingTee();

      // IMPORTANT: TreeWalker processes operations and adds block args to
      // needsLocal when it encounters them as operands. However, it may miss
      // some block args (e.g., in function-level successor blocks). To ensure
      // correctness, we conservatively add ALL block arguments to needsLocal.
      // Block args arrive on the stack at block entry, but their stack position
      // is not controlled by code motion, so using locals is the safe approach.
      allocateLocalsForBlockArgs(funcOp.getBody(), needsLocal);

      // Allocate local indices
      LocalAllocator allocator;
      allocator.allocate(funcOp, needsLocal, needsTee);

      // Report allocation results
      if (!needsLocal.empty()) {
        llvm::errs() << "    Values needing locals: " << needsLocal.size()
                     << "\n";
      }
      if (!needsTee.empty()) {
        llvm::errs() << "    Values needing tee: " << needsTee.size() << "\n";
      }
      if (allocator.getNumLocals() > allocator.getNumParams()) {
        llvm::errs() << "    Allocated locals: "
                     << (allocator.getNumLocals() - allocator.getNumParams())
                     << " (params: " << allocator.getNumParams()
                     << ", total: " << allocator.getNumLocals() << ")\n";
      }

      // Emit WasmStack function
      WasmStackEmitter emitter(builder, allocator, needsTee);
      emitter.emitFunction(funcOp);

      // Restore insertion point to module body for next function
      builder.setInsertionPointToEnd(module.getBody());

      // Remove the original WasmSSA function
      funcOp.erase();
    }
  }
};

} // namespace mlir::wasmstack
