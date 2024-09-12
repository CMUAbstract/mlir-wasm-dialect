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

#include "Wasm/ConversionPatterns/ArithToWasmPatterns.h"
#include "Wasm/ConversionPatterns/FuncToWasmPatterns.h"
#include "Wasm/ConversionPatterns/MemRefToWasmPatterns.h"
#include "Wasm/ConversionPatterns/ScfToWasmPatterns.h"
#include "Wasm/ConversionPatterns/WasmFinalizePatterns.h"
#include "Wasm/VariableAnalysis.h"
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

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    WasmTypeConverter typeConverter(context);
    populateArithToWasmPatterns(typeConverter, patterns);
    populateFuncToWasmPatterns(typeConverter, patterns);
    populateScfToWasmPatterns(typeConverter, patterns);
    populateMemRefToWasmPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class WasmFinalize : public impl::WasmFinalizeBase<WasmFinalize> {
public:
  using impl::WasmFinalizeBase<WasmFinalize>::WasmFinalizeBase;

  void runOnOperation() final {
    wasm::WasmFuncOp func = getOperation();
    MLIRContext *context = func.getContext();
    VariableAnalysis analysis(func);

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalOp<wasm::TempLocalOp>();
    target.addIllegalOp<wasm::TempLocalGetOp>();
    target.addIllegalOp<wasm::TempLocalSetOp>();
    // TODO: mark this as illegal after implementing function argument handling
    // target.addIllegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWasmFinalizePatterns(context, analysis, patterns);

    // TODO: place it somewhere else
    // declare local at the beginning of the function
    // note that analysis.getTypesAttrs() must be called before applying the
    // conversion because mlir Values are erased during the conversion
    PatternRewriter rewriter(context);
    Operation *firstOp = &(*func->getRegion(0).getBlocks().begin()->begin());
    rewriter.setInsertionPoint(firstOp);
    std::vector<mlir::Attribute> types;
    for (auto typeAttr : analysis.getTypeAttrs()) {
      types.push_back(typeAttr);
    }
    llvm::ArrayRef<mlir::Attribute> typesRef(types);
    rewriter.create<wasm::LocalOp>(func.getLoc(),
                                   rewriter.getArrayAttr(typesRef));

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::wasm
