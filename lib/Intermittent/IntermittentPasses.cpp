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

void importHostFunctions(OpBuilder &builder, Location loc) {
  auto simpleFuncType = builder.getFunctionType({}, {});
  builder.create<wasm::ImportFuncOp>(loc, "begin_commit", simpleFuncType);
  builder.create<wasm::ImportFuncOp>(loc, "end_commit", simpleFuncType);
  builder.create<wasm::ImportFuncOp>(
      loc, "set_i32",
      builder.getFunctionType({builder.getI32Type(), builder.getI32Type()},
                              {}));
  builder.create<wasm::ImportFuncOp>(
      loc, "set_i64",
      builder.getFunctionType({builder.getI32Type(), builder.getI64Type()},
                              {}));
  builder.create<wasm::ImportFuncOp>(
      loc, "set_f32",
      builder.getFunctionType({builder.getI32Type(), builder.getF32Type()},
                              {}));
  builder.create<wasm::ImportFuncOp>(
      loc, "set_f64",
      builder.getFunctionType({builder.getI32Type(), builder.getF64Type()},
                              {}));
  builder.create<wasm::ImportFuncOp>(
      loc, "get_i32",
      builder.getFunctionType({builder.getI32Type()}, {builder.getI32Type()}));
  builder.create<wasm::ImportFuncOp>(
      loc, "get_i64",
      builder.getFunctionType({builder.getI32Type()}, {builder.getI64Type()}));
  builder.create<wasm::ImportFuncOp>(
      loc, "get_f32",
      builder.getFunctionType({builder.getI32Type()}, {builder.getF32Type()}));
  builder.create<wasm::ImportFuncOp>(
      loc, "get_f64",
      builder.getFunctionType({builder.getI32Type()}, {builder.getF64Type()}));
}

void addTypes(OpBuilder &builder, Location loc) {
  builder.create<wasm::FuncTypeOp>(loc, "ft", builder.getFunctionType({}, {}));
  builder.create<wasm::ContinuationTypeOp>(loc, "ct", "ft");
}

void addTag(OpBuilder &builder, Location loc) {
  builder.create<wasm::TagOp>(loc, "yield");
}

void addGlobalVariables(OpBuilder &builder, Location loc) {
  builder.create<wasm::GlobalOp>(loc, "curr_task", builder.getI32Type());
}
void addTable(OpBuilder &builder, Location loc) {
  // TODO: Configure size
  builder.create<wasm::ContinuationTableOp>(loc, "task_table", 100, "ct");
  // TODO: Add each task to the table
}

class PrepareForIntermittent
    : public impl::PrepareForIntermittentBase<PrepareForIntermittent> {
public:
  using impl::PrepareForIntermittentBase<
      PrepareForIntermittent>::PrepareForIntermittentBase;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    OpBuilder builder(context);
    builder.setInsertionPoint(moduleOp.getBody(), moduleOp.getBody()->begin());

    auto loc = moduleOp.getLoc();
    importHostFunctions(builder, loc);
    addTypes(builder, loc);
    addTag(builder, loc);
    addGlobalVariables(builder, loc);
    addTable(builder, loc);
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
