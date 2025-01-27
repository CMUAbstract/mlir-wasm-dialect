//===- PreProcessForIntermittentToWasm.cpp - Preprocess for Intermittent to Wasm
// pass
//-----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Intermittent/IntermittentPasses.h"
#include "Wasm/WasmOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_PREPROCESSINTERMITTENTTOWASM
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
  auto contType = wasm::ContinuationType::get(builder.getContext(), "ct", "ft");
  builder.create<wasm::FuncTypeDeclOp>(loc, "ft",
                                       builder.getFunctionType({contType}, {}));
  builder.create<wasm::ContinuationTypeDeclOp>(loc, contType);
}

void addTag(OpBuilder &builder, Location loc) {
  builder.create<wasm::TagOp>(loc, "yield");
}

void addTable(ModuleOp &moduleOp, MLIRContext *context, OpBuilder &builder,
              Location loc) {
  int numTasks = 0;
  llvm::SmallVector<mlir::Attribute, 4> taskNames;
  moduleOp.walk([&](IdempotentTaskOp taskOp) {
    numTasks++;
    taskNames.push_back(FlatSymbolRefAttr::get(context, taskOp.getSymName()));
  });
  builder.create<wasm::ContinuationTableOp>(loc, "task_table", numTasks, "ct");
}

void addMainFunction(ModuleOp &moduleOp, MLIRContext *context,
                     OpBuilder &builder, Location loc) {

  // add a global variable to store the current task
  auto currTaskGlobalOp =
      builder.create<wasm::TempGlobalOp>(loc, true, builder.getI32Type());
  // global_name is a temporary attribute used to reference this global variable
  currTaskGlobalOp->setAttr("global_name", builder.getStringAttr("curr_task"));

  auto mainFuncOp = builder.create<wasm::WasmFuncOp>(
      loc, "main", builder.getFunctionType({}, {}));
  auto &entryRegion = mainFuncOp.getBody();
  auto *entryBlock = builder.createBlock(&entryRegion);
  builder.setInsertionPointToEnd(entryBlock);

  // Set table elements (TODO: Ideally tables should be initialized through the
  // elem segment)

  size_t taskIndex = 0;
  for (auto taskOp : moduleOp.getOps<IdempotentTaskOp>()) {
    taskOp->setAttr("task_index", builder.getI32IntegerAttr(taskIndex));
    builder.create<wasm::ConstantOp>(loc, builder.getI32IntegerAttr(taskIndex));
    taskIndex++;
    builder.create<wasm::FuncRefOp>(loc, taskOp.getSymName());
    builder.create<wasm::ContNewOp>(loc, "ct");
    builder.create<wasm::TableSetOp>(loc, "task_table");
  }

  // Restore global variables Restore the current task
  for (auto tempGlobalOp : moduleOp.getOps<wasm::TempGlobalOp>()) {
    // The TempGlobalIndexOp will be replaced by wasm::ConstantOp
    // representing the index value of the global variable.
    builder.create<wasm::TempGlobalIndexOp>(loc, tempGlobalOp.getResult());
    builder.create<wasm::CallOp>(loc, "get_i32");
    builder.create<wasm::TempGlobalSetOp>(loc, tempGlobalOp.getResult());
  }
  // restore the current task
  builder.create<wasm::TempGlobalGetOp>(loc, currTaskGlobalOp.getResult());
  builder.create<wasm::TableGetOp>(loc, "task_table");

  // FIXME: We should also add one more argument of type (ref null $ct)

  // Run the current task
  builder.create<wasm::ResumeSwitchOp>(loc, "ct", "yield");

  builder.create<wasm::WasmReturnOp>(loc);
}

class PreprocessIntermittentToWasm
    : public impl::PreprocessIntermittentToWasmBase<
          PreprocessIntermittentToWasm> {
public:
  using impl::PreprocessIntermittentToWasmBase<
      PreprocessIntermittentToWasm>::PreprocessIntermittentToWasmBase;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    OpBuilder builder(context);
    builder.setInsertionPoint(moduleOp.getBody(), moduleOp.getBody()->begin());

    auto loc = moduleOp.getLoc();
    importHostFunctions(builder, loc);
    addTypes(builder, loc);
    addTag(builder, loc);
    addTable(moduleOp, context, builder, loc);
    addMainFunction(moduleOp, context, builder, loc);
  }
};

} // namespace mlir::intermittent