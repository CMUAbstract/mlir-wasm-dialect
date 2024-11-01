//===- IntermittentPasses.cpp - Wasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Intermittent/IntermittentPasses.h"
#include "Wasm/WasmOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_PREPAREFORINTERMITTENT
#define GEN_PASS_DEF_CONVERTINTERMITTENTTASKTOWASM
#include "Intermittent/IntermittentPasses.h.inc"

void importHostFunctions(PatternRewriter &rewriter, Location loc) {
  auto simpleFuncType = rewriter.getFunctionType({}, {});
  rewriter.create<wasm::ImportFuncOp>(loc, "begin_commit", simpleFuncType);
  rewriter.create<wasm::ImportFuncOp>(loc, "end_commit", simpleFuncType);
  rewriter.create<wasm::ImportFuncOp>(
      loc, "set_i32",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI32Type()},
                               {}));
  rewriter.create<wasm::ImportFuncOp>(
      loc, "set_i64",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI64Type()},
                               {}));
  rewriter.create<wasm::ImportFuncOp>(
      loc, "set_f32",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getF32Type()},
                               {}));
  rewriter.create<wasm::ImportFuncOp>(
      loc, "set_f64",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getF64Type()},
                               {}));
  rewriter.create<wasm::ImportFuncOp>(
      loc, "get_i32",
      rewriter.getFunctionType({rewriter.getI32Type()},
                               {rewriter.getI32Type()}));
  rewriter.create<wasm::ImportFuncOp>(
      loc, "get_i64",
      rewriter.getFunctionType({rewriter.getI32Type()},
                               {rewriter.getI64Type()}));
  rewriter.create<wasm::ImportFuncOp>(
      loc, "get_f32",
      rewriter.getFunctionType({rewriter.getI32Type()},
                               {rewriter.getF32Type()}));
  rewriter.create<wasm::ImportFuncOp>(
      loc, "get_f64",
      rewriter.getFunctionType({rewriter.getI32Type()},
                               {rewriter.getF64Type()}));
}

void addTypes(PatternRewriter &rewriter, Location loc) {
  rewriter.create<wasm::FuncTypeOp>(loc, "$ft",
                                    rewriter.getFunctionType({}, {}));
  rewriter.create<wasm::ContinuationTypeOp>(loc, "$ct", "$ft");
}

void addTag(PatternRewriter &rewriter, Location loc) {
  rewriter.create<wasm::TagOp>(loc, "$yield");
}

void addGlobalVariables(PatternRewriter &rewriter, Location loc) {
  // TODO
}

void addTable(PatternRewriter &rewriter, Location loc) {
  // TODO
}

class PrepareForIntermittent
    : public impl::PrepareForIntermittentBase<PrepareForIntermittent> {
public:
  using impl::PrepareForIntermittentBase<
      PrepareForIntermittent>::PrepareForIntermittentBase;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    PatternRewriter rewriter(context);

    auto loc = moduleOp.getLoc();
    importHostFunctions(rewriter, loc);
    addTypes(rewriter, loc);
    addTag(rewriter, loc);
    addGlobalVariables(rewriter, loc);
  }
};

struct IdempotentTaskOpLowering {};

struct TransitionToOpLowering {};

class ConvertIntermittentTaskToWasm
    : public impl::ConvertIntermittentTaskToWasmBase<
          ConvertIntermittentTaskToWasm> {
public:
  using impl::ConvertIntermittentTaskToWasmBase<
      ConvertIntermittentTaskToWasm>::ConvertIntermittentTaskToWasmBase;
  void runOnOperation() final {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<wasm::WasmDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<intermittent::IntermittentDialect>();
  }
};

} // namespace mlir::intermittent
