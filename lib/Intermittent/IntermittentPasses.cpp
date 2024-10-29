//===- IntermittentPasses.cpp - Wasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Intermittent/IntermittentPasses.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_CONVERTINTERMITTENTTOWASM
#include "Intermittent/IntermittentPasses.h.inc"

class ConvertIntermittentToWasm
    : public impl::ConvertIntermittentToWasmBase<ConvertIntermittentToWasm> {
public:
  using impl::ConvertIntermittentToWasmBase<
      ConvertIntermittentToWasm>::ConvertIntermittentToWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<wasm::WasmDialect>();
    target.addLegalDialect<arith::ArithDialect>();
  }
};

} // namespace mlir::intermittent
