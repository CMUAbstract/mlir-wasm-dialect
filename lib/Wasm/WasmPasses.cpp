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
#include "Wasm/WasmPasses.h"

namespace mlir::wasm {
#define GEN_PASS_DEF_CONVERTTOWASM
#include "Wasm/WasmPasses.h.inc"

class ConvertToWasm : public impl::ConvertToWasmBase<ConvertToWasm> {
public:
  using impl::ConvertToWasmBase<ConvertToWasm>::ConvertToWasmBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    MLIRContext *context = func.getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateArithToWasmPatterns(context, patterns);
    populateFuncToWasmPatterns(context, patterns);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::wasm
