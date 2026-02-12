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
#include "wasmstack/LocalAllocator.h"
#include "wasmstack/StackificationAnalysis.h"
#include "wasmstack/StackificationPlan.h"
#include "wasmstack/TreeWalker.h"
#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackEmitter.h"
#include "wasmstack/WasmStackOps.h"

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

namespace mlir::wasmstack {

#define GEN_PASS_DEF_CONVERTTOWASMSTACK
#include "wasmstack/WasmStackPasses.h.inc"

namespace {

/// Analyze one function and produce a deterministic stackification plan.
static LogicalResult analyzeStackification(wasmssa::FuncOp funcOp,
                                           StackificationPlan &plan) {
  if (funcOp.getBody().empty())
    return success();

  UseCountAnalysis useCount(funcOp.getBody());
  TreeWalker walker(useCount);
  walker.processRegion(funcOp.getBody());

  // Preserve deterministic order from the tree walk.
  for (Value v : walker.getLocalOrder())
    if (walker.getValuesNeedingLocals().contains(v))
      plan.requireLocal(v);
  for (Value v : walker.getTeeOrder())
    if (walker.getValuesNeedingTee().contains(v))
      plan.requireTee(v);

  // Conservatively materialize all block arguments as locals.
  allocateLocalsForBlockArgsOrdered(funcOp.getBody(), plan.needsLocal,
                                    plan.localOrder);

  // Local-only policy takes precedence over tee.
  for (Value v : plan.needsLocal)
    plan.needsTee.erase(v);

  bool policyError = false;
  for (Value v : plan.needsTee) {
    if (plan.needsLocal.contains(v)) {
      funcOp.emitError("internal stackification policy error: value is both "
                       "local-only and tee-backed");
      policyError = true;
      break;
    }
  }
  if (policyError)
    return failure();

  // Control-flow interface operands must be local-backed.
  funcOp.walk([&](wasmssa::BlockReturnOp blockReturnOp) {
    for (Value v : blockReturnOp.getInputs()) {
      if (!plan.isLocal(v)) {
        blockReturnOp.emitError("block_return operand must be local-backed "
                                "after stackification analysis");
        policyError = true;
        return;
      }
    }
  });
  if (policyError)
    return failure();

  funcOp.walk([&](wasmssa::BranchIfOp branchIfOp) {
    if (!plan.isLocal(branchIfOp.getCondition())) {
      branchIfOp.emitError("branch_if condition must be local-backed after "
                           "stackification analysis");
      policyError = true;
      return;
    }
    for (Value v : branchIfOp.getInputs()) {
      if (!plan.isLocal(v)) {
        branchIfOp.emitError("branch_if operand must be local-backed after "
                             "stackification analysis");
        policyError = true;
        return;
      }
    }
  });

  return policyError ? failure() : success();
}

static SmallVector<Value> getFilteredTeeOrder(const StackificationPlan &plan) {
  SmallVector<Value> filtered;
  filtered.reserve(plan.teeOrder.size());
  for (Value v : plan.getTeeOrder()) {
    if (plan.isTee(v) && !plan.isLocal(v))
      filtered.push_back(v);
  }
  return filtered;
}

} // namespace

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

    // This pass expects pre-wasmstack input and materializes exactly one
    // wasmstack.module wrapper for emitted WasmStack functions.
    if (!module.getOps<wasmstack::ModuleOp>().empty()) {
      module.emitError("convert-to-wasmstack expects input without existing "
                       "wasmstack.module");
      signalPassFailure();
      return;
    }

    // Collect imports and functions to process.
    SmallVector<wasmssa::FuncImportOp> importsToConvert;
    for (auto importOp : module.getOps<wasmssa::FuncImportOp>())
      importsToConvert.push_back(importOp);

    SmallVector<wasmssa::FuncOp> funcsToConvert;
    for (auto funcOp : module.getOps<wasmssa::FuncOp>())
      funcsToConvert.push_back(funcOp);

    if (importsToConvert.empty() && funcsToConvert.empty())
      return;

    // Emit a default linear memory when memory ops or malloc/free runtime
    // imports are present.
    bool needsLinearMemory = false;
    for (auto importOp : importsToConvert) {
      StringRef symName = importOp.getSymName();
      if (symName == "malloc" || symName == "free") {
        needsLinearMemory = true;
        break;
      }
    }

    if (!needsLinearMemory) {
      for (auto funcOp : funcsToConvert) {
        funcOp.walk([&](Operation *nestedOp) {
          if (isa<wami::LoadOp, wami::StoreOp>(nestedOp))
            needsLinearMemory = true;
        });
        if (needsLinearMemory)
          break;
      }
    }

    // Create builder for emitting new operations
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    // Create the canonical WasmStack wrapper module for conversion output.
    auto wasmModule = wasmstack::ModuleOp::create(builder, module.getLoc(),
                                                  /*sym_name=*/StringAttr());
    if (wasmModule.getBody().empty())
      wasmModule.getBody().push_back(new Block());
    builder.setInsertionPointToEnd(&wasmModule.getBody().front());

    if (needsLinearMemory) {
      std::string memorySym = "__linear_memory";
      auto symbolIsTaken = [&](StringRef name) {
        for (auto importOp : importsToConvert)
          if (importOp.getSymName() == name)
            return true;
        for (auto funcOp : funcsToConvert)
          if (funcOp.getSymName() == name)
            return true;
        return false;
      };
      for (unsigned suffix = 0; symbolIsTaken(memorySym); ++suffix)
        memorySym = ("__linear_memory_" + llvm::Twine(suffix + 1)).str();

      wasmstack::MemoryOp::create(builder, module.getLoc(),
                                  builder.getStringAttr(memorySym),
                                  builder.getI32IntegerAttr(1), IntegerAttr(),
                                  builder.getStringAttr("memory"));
    }

    // Preserve wasmssa.import_func as wasmstack.import_func declarations.
    for (wasmssa::FuncImportOp importOp : importsToConvert) {
      wasmstack::FuncImportOp::create(
          builder, importOp.getLoc(), importOp.getSymNameAttr(),
          importOp.getModuleNameAttr(), importOp.getImportNameAttr(),
          TypeAttr::get(importOp.getType()));
      importOp.erase();
    }

    // Process each WasmSSA function
    for (wasmssa::FuncOp funcOp : funcsToConvert) {
      // Skip empty functions
      if (funcOp.getBody().empty())
        continue;

      // Analyze use counts for operation results in the function.
      StackificationPlan plan;
      if (failed(analyzeStackification(funcOp, plan))) {
        signalPassFailure();
        return;
      }

      SmallVector<Value> teeOrder = getFilteredTeeOrder(plan);

      // Allocate local indices
      LocalAllocator allocator;
      allocator.allocate(funcOp, plan.getLocalOrder(), teeOrder);

      // Report allocation results
      if (!plan.needsLocal.empty()) {
        llvm::errs() << "    Values needing locals: " << plan.needsLocal.size()
                     << "\n";
      }
      if (!plan.needsTee.empty()) {
        llvm::errs() << "    Values needing tee: " << plan.needsTee.size()
                     << "\n";
      }
      if (allocator.getNumLocals() > allocator.getNumParams()) {
        llvm::errs() << "    Allocated locals: "
                     << (allocator.getNumLocals() - allocator.getNumParams())
                     << " (params: " << allocator.getNumParams()
                     << ", total: " << allocator.getNumLocals() << ")\n";
      }

      // Emit WasmStack function
      WasmStackEmitter emitter(builder, allocator, plan.needsTee);
      emitter.emitFunction(funcOp);
      if (emitter.hasFailed()) {
        signalPassFailure();
        return;
      }

      // Restore insertion point to wasmstack.module body for next function.
      builder.setInsertionPointToEnd(&wasmModule.getBody().front());

      // Remove the original WasmSSA function
      funcOp.erase();
    }
  }
};

} // namespace mlir::wasmstack
