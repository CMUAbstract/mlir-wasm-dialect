//===- WasmPasses.cpp - Wasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Wasm/BaseAddrAnalysis.h"
#include "Wasm/ConversionPatterns/ArithToWasmPatterns.h"
#include "Wasm/ConversionPatterns/FuncToWasmPatterns.h"
#include "Wasm/ConversionPatterns/MemRefToWasmPatterns.h"
#include "Wasm/ConversionPatterns/ScfToWasmPatterns.h"
#include "Wasm/ConversionPatterns/WasmFinalizePatterns.h"
#include "Wasm/WasmFinalizeAnalysis.h"
#include "Wasm/WasmPasses.h"

namespace mlir::wasm {
#define GEN_PASS_DEF_CONVERTTOWASM
#define GEN_PASS_DEF_WASMFINALIZE
#include "Wasm/WasmPasses.h.inc"

class WasmTypeConverter : public TypeConverter {
public:
  WasmTypeConverter(MLIRContext *ctx) {
    addConversion(
        [ctx](IntegerType type) -> Type { return LocalType::get(ctx, type); });
    addConversion(
        [ctx](FloatType type) -> Type { return LocalType::get(ctx, type); });
    addConversion(
        [ctx](IndexType type) -> Type { return LocalType::get(ctx, type); });
    addConversion([ctx](MemRefType type) -> Type {
      return LocalType::get(ctx, IntegerType::get(ctx, 32));
    });

    addArgumentMaterialization([](OpBuilder &builder, Type type,
                                  ValueRange inputs,
                                  Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });

    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs,
                                Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });

    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs,
                                Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });
  }
};

class ConvertToWasm : public impl::ConvertToWasmBase<ConvertToWasm> {
public:
  using impl::ConvertToWasmBase<ConvertToWasm>::ConvertToWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    BaseAddrAnalysis analysis(module);

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    WasmTypeConverter typeConverter(context);
    populateArithToWasmPatterns(typeConverter, patterns);
    populateFuncToWasmPatterns(typeConverter, patterns);
    populateMemRefToWasmPatterns(typeConverter, patterns, analysis);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // we need to apply scf conversion separately because calling
    // `replaceAllUsesWith()` on induction variable will cause the conversion to
    // fail with the error message "Assertion failed: (!impl->wasOpReplaced(op)
    // && "attempting to modify a replaced/erased op")"
    RewritePatternSet nextPatterns(context);
    target.addIllegalDialect<scf::SCFDialect>();
    populateScfToWasmPatterns(typeConverter, nextPatterns);
    if (failed(
            applyPartialConversion(module, target, std::move(nextPatterns)))) {
      signalPassFailure();
    }
  }
};

class WasmFinalize : public impl::WasmFinalizeBase<WasmFinalize> {
public:
  using impl::WasmFinalizeBase<WasmFinalize>::WasmFinalizeBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    WasmFinalizeAnalysis analysis(moduleOp);

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addIllegalOp<wasm::TempGlobalOp>();
    target.addIllegalOp<wasm::TempGlobalGetOp>();
    target.addIllegalOp<wasm::TempGlobalSetOp>();
    target.addIllegalOp<wasm::TempLocalOp>();
    target.addIllegalOp<wasm::TempLocalGetOp>();
    target.addIllegalOp<wasm::TempLocalSetOp>();

    RewritePatternSet patterns(context);
    populateWasmFinalizePatterns(context, analysis, patterns);

    // TODO: place it somewhere else
    // declare local at the beginning of each function
    // note that analysis.getLocalTypesRef() must be called before applying the
    // conversion because mlir Values are erased during the conversion
    PatternRewriter rewriter(context);
    moduleOp.walk([&](WasmFuncOp funcOp) {
      Operation *funcFirstOp =
          &(*funcOp->getRegion(0).getBlocks().begin()->begin());
      rewriter.setInsertionPoint(funcFirstOp);
      auto localTypesRef = analysis.getLocalTypesRef(funcOp.getOperation());
      rewriter.create<wasm::LocalOp>(funcOp.getLoc(),
                                     rewriter.getArrayAttr(localTypesRef));
    });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::wasm
