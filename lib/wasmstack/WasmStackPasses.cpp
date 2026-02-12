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
#include "mlir/IR/BuiltinAttributes.h"
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

static LogicalResult emitGlobalConstExpr(wasmssa::ConstOp constOp,
                                         OpBuilder &builder) {
  Location loc = constOp.getLoc();
  Type resultType = constOp.getResult().getType();
  Attribute value = constOp.getValueAttr();

  if (resultType.isInteger(32)) {
    auto intVal = cast<IntegerAttr>(value).getInt();
    wasmstack::I32ConstOp::create(
        builder, loc, builder.getI32IntegerAttr(static_cast<int32_t>(intVal)));
    return success();
  }
  if (resultType.isInteger(64)) {
    auto intVal = cast<IntegerAttr>(value).getInt();
    wasmstack::I64ConstOp::create(builder, loc,
                                  builder.getI64IntegerAttr(intVal));
    return success();
  }
  if (resultType.isF32()) {
    auto floatVal = cast<FloatAttr>(value).getValueAsDouble();
    wasmstack::F32ConstOp::create(
        builder, loc, builder.getF32FloatAttr(static_cast<float>(floatVal)));
    return success();
  }
  if (resultType.isF64()) {
    auto floatVal = cast<FloatAttr>(value).getValueAsDouble();
    wasmstack::F64ConstOp::create(builder, loc,
                                  builder.getF64FloatAttr(floatVal));
    return success();
  }

  constOp.emitError("unsupported wasmssa.const type in global initializer: ")
      << resultType;
  return failure();
}

static LogicalResult lowerGlobalInitializer(wasmssa::GlobalOp srcGlobal,
                                            wasmstack::GlobalOp dstGlobal,
                                            OpBuilder &builder) {
  Region &srcInit = srcGlobal.getInitializer();
  if (srcInit.empty()) {
    srcGlobal.emitError("expected non-empty initializer region");
    return failure();
  }
  Region &dstInit = dstGlobal.getInit();
  if (dstInit.empty())
    dstInit.push_back(new Block());

  builder.setInsertionPointToEnd(&dstInit.front());

  bool sawReturn = false;
  for (Operation &op : srcInit.front()) {
    if (auto constOp = dyn_cast<wasmssa::ConstOp>(op)) {
      if (failed(emitGlobalConstExpr(constOp, builder)))
        return failure();
      continue;
    }
    if (auto globalGetOp = dyn_cast<wasmssa::GlobalGetOp>(op)) {
      wasmstack::GlobalGetOp::create(
          builder, globalGetOp.getLoc(), globalGetOp.getGlobalAttr(),
          TypeAttr::get(globalGetOp.getResult().getType()));
      continue;
    }
    if (auto returnOp = dyn_cast<wasmssa::ReturnOp>(op)) {
      if (returnOp.getNumOperands() != 1) {
        returnOp.emitError(
            "wasmssa.global initializer return must have exactly one operand");
        return failure();
      }
      sawReturn = true;
      continue;
    }

    op.emitError("unsupported operation in wasmssa.global initializer for "
                 "convert-to-wasmstack");
    return failure();
  }

  if (!sawReturn) {
    srcGlobal.emitError(
        "wasmssa.global initializer must end with wasmssa.return");
    return failure();
  }
  return success();
}

static FailureOr<StringAttr> getDataBytesAttr(wami::DataOp dataOp,
                                              OpBuilder &builder) {
  auto denseData = dyn_cast<DenseElementsAttr>(dataOp.getValue());
  if (!denseData) {
    dataOp.emitError("wami.data requires dense elements for conversion to "
                     "wasmstack.data");
    return failure();
  }

  ArrayRef<char> rawData = denseData.getRawData();
  return builder.getStringAttr(StringRef(rawData.data(), rawData.size()));
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

    SmallVector<wasmssa::GlobalOp> globalsToConvert;
    for (auto globalOp : module.getOps<wasmssa::GlobalOp>())
      globalsToConvert.push_back(globalOp);

    SmallVector<wami::DataOp> dataToConvert;
    for (auto dataOp : module.getOps<wami::DataOp>())
      dataToConvert.push_back(dataOp);

    if (importsToConvert.empty() && funcsToConvert.empty() &&
        globalsToConvert.empty() && dataToConvert.empty())
      return;

    // Emit a default linear memory when memory ops or malloc/free runtime
    // imports are present.
    bool needsLinearMemory = !dataToConvert.empty();
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

    std::string memorySym;
    if (needsLinearMemory) {
      memorySym = "__linear_memory";
      auto symbolIsTaken = [&](StringRef name) {
        for (auto importOp : importsToConvert)
          if (importOp.getSymName() == name)
            return true;
        for (auto funcOp : funcsToConvert)
          if (funcOp.getSymName() == name)
            return true;
        for (auto globalOp : globalsToConvert)
          if (globalOp.getSymName() == name)
            return true;
        for (auto dataOp : dataToConvert)
          if (dataOp.getSymName() == name)
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

    for (wasmssa::GlobalOp globalOp : globalsToConvert) {
      StringAttr exportNameAttr;
      if (globalOp.getExported())
        exportNameAttr = builder.getStringAttr(globalOp.getSymName());

      auto newGlobal = wasmstack::GlobalOp::create(
          builder, globalOp.getLoc(), globalOp.getSymName(), globalOp.getType(),
          globalOp.getIsMutable(), exportNameAttr);

      if (failed(lowerGlobalInitializer(globalOp, newGlobal, builder))) {
        signalPassFailure();
        return;
      }

      // Restore insertion point to module body after emitting init region.
      builder.setInsertionPointToEnd(&wasmModule.getBody().front());
      globalOp.erase();
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

    if (!dataToConvert.empty() && memorySym.empty()) {
      module.emitError(
          "internal error: data conversion requires generated linear memory");
      signalPassFailure();
      return;
    }

    for (wami::DataOp dataOp : dataToConvert) {
      FailureOr<StringAttr> dataBytes = getDataBytesAttr(dataOp, builder);
      if (failed(dataBytes)) {
        signalPassFailure();
        return;
      }

      wasmstack::DataOp::create(
          builder, dataOp.getLoc(), FlatSymbolRefAttr::get(ctx, memorySym),
          builder.getI32IntegerAttr(dataOp.getOffset()), *dataBytes);
      dataOp.erase();
    }
  }
};

} // namespace mlir::wasmstack
