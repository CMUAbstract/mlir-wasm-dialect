//===- ConvertIntermittentToLLVM.cpp - Convert Intermittent to LLVM
//-----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Intermittent/IntermittentPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_CONVERTINTERMITTENTTOLLVM
#include "Intermittent/IntermittentPasses.h.inc"

struct ConvertIntermittentToLLVM
    : public impl::ConvertIntermittentToLLVMBase<ConvertIntermittentToLLVM> {
  using ConvertIntermittentToLLVMBase<
      ConvertIntermittentToLLVM>::ConvertIntermittentToLLVMBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();

    // A builder for top-level insertion (moduleOp scope).
    OpBuilder topBuilder(ctx);

    // We'll gather tasks, then transform them in one pass so
    // we don't invalidate iterators while walking.
    SmallVector<IdempotentTaskOp, 4> tasks;
    SmallVector<std::string, 4> taskNames;
    moduleOp.walk([&](IdempotentTaskOp op) {
      tasks.push_back(op);
      taskNames.push_back(op.getSymName().str());
    });

    for (auto taskOp : tasks) {
      convertTaskOp(taskNames, taskOp, moduleOp);
    }
  }

private:
  /// Replace every TransitionToOp in 'block' with a branch to 'saveBlock'.
  /// If there are multiple TransitionToOps, each is replaced with a BranchOp.
  void rewriteTransitionToOps(
      llvm::function_ref<Value(mlir::StringRef)> getTaskToken, Block &block,
      Block *saveBlock, Location loc) {
    for (auto &op : llvm::make_early_inc_range(block)) {
      if (auto transOp = dyn_cast<TransitionToOp>(&op)) {
        OpBuilder builder(transOp);
        // FIXME: save the arguments of TransitionToOp here

        auto nextTaskName = transOp.getNextTask();
        Value nextTaskToken = getTaskToken(nextTaskName);
        builder.create<cf::BranchOp>(loc, saveBlock, nextTaskToken);
        transOp.erase();
      }
    }
  }

  void convertTaskOp(SmallVector<std::string, 4> &taskNames,
                     IdempotentTaskOp taskOp, ModuleOp module) {
    // 1) Get the symbol name
    auto symNameAttr = taskOp->getAttrOfType<StringAttr>("sym_name");
    if (!symNameAttr) {
      taskOp.emitError("IdempotentTaskOp has no sym_name attribute");
      return;
    }
    StringRef taskName = symNameAttr.getValue();

    OpBuilder builder(module.getContext());
    auto loc = taskOp.getLoc();

    auto asyncTokenType = mlir::async::TokenType::get(module.getContext());
    SmallVector<Type, 4> tokenTypes;
    for (auto _ : taskNames) {
      tokenTypes.push_back(asyncTokenType);
    }

    auto funcType = builder.getFunctionType(tokenTypes, {});

    // Create the function: func.func @taskName(!async.token) -> !async.token
    auto newFunc = func::FuncOp::create(loc, taskName, funcType);
    module.push_back(newFunc);

    llvm::dbgs() << "Creating blocks\n";

    // Create blocks: ^header, ^mainbody, ^saveblock, ^resume, ^cleanup,
    // ^suspend
    Block *headerBlock = newFunc.addEntryBlock();
    Block *mainBodyBlock = new Block();
    Block *saveBlock = new Block();
    saveBlock->addArgument(asyncTokenType, newFunc.getLoc());
    Block *resumeBlock = new Block();
    Block *cleanupBlock = new Block();
    Block *suspendBlock = new Block();

    newFunc.getBody().push_back(mainBodyBlock);
    newFunc.getBody().push_back(saveBlock);
    newFunc.getBody().push_back(resumeBlock);
    newFunc.getBody().push_back(cleanupBlock);
    newFunc.getBody().push_back(suspendBlock);

    // 3) Fill ^header:
    llvm::dbgs() << "Filling ^header\n";
    builder.setInsertionPointToStart(headerBlock);

    auto idOp = builder.create<async::CoroIdOp>(loc);
    Value idValue = idOp.getResult();
    auto handleOp = builder.create<async::CoroBeginOp>(loc, idValue);
    Value handleValue = handleOp.getResult();

    // Branch to ^mainbody
    builder.create<cf::BranchOp>(loc, mainBodyBlock);

    // 4) Fill ^mainbody with user ops
    llvm::dbgs() << "Filling ^mainbody\n";
    builder.setInsertionPointToStart(mainBodyBlock);

    auto getTaskToken = [&](StringRef taskName) -> Value {
      for (size_t i = 0; i < taskNames.size(); ++i) {
        if (taskNames[i] == taskName)
          return newFunc.getArgument(i);
      }
      return nullptr; // TODO: Error handling
    };

    // FIXME: do not unify multiple blocks
    // Move or clone the original task body. For simplicity,
    // we assume a single block or unify multiple blocks.
    Region &taskBody = taskOp.getBody();
    if (!taskBody.empty()) {
      // splice or clone all ops into mainBodyBlock
      for (Block &oldBlock : llvm::make_early_inc_range(taskBody)) {
        auto &ops = oldBlock.getOperations();
        mainBodyBlock->getOperations().splice(mainBodyBlock->end(), ops,
                                              ops.begin(), ops.end());
      }
      rewriteTransitionToOps(getTaskToken, *mainBodyBlock, saveBlock, loc);
    }
    // 5) Fill ^saveblock (where we do async.coro.save, await, suspend, etc.)
    llvm::dbgs() << "Filling ^saveblock\n";
    builder.setInsertionPointToStart(saveBlock);

    auto stateOp = builder.create<async::CoroSaveOp>(
        loc, async::CoroStateType::get(builder.getContext()), handleValue);
    Value stateVal = stateOp.getResult();

    Value nextTaskToken = saveBlock->getArgument(0);
    builder.create<async::RuntimeAwaitAndResumeOp>(loc, nextTaskToken,
                                                   handleValue);

    builder.create<async::CoroSuspendOp>(loc, stateVal, suspendBlock,
                                         resumeBlock, cleanupBlock);

    // 6) Fill ^resume:
    llvm::dbgs() << "Filling ^resume\n";
    builder.setInsertionPointToStart(resumeBlock);

    // FIXME
    builder.create<async::RuntimeSetAvailableOp>(loc, newFunc.getArgument(0));

    builder.create<cf::BranchOp>(loc, mainBodyBlock);

    // 7) Fill ^cleanup
    llvm::dbgs() << "Filling ^cleanup\n";
    builder.setInsertionPointToStart(cleanupBlock);
    builder.create<async::CoroFreeOp>(loc, idValue, handleValue);
    builder.create<cf::BranchOp>(loc, suspendBlock);

    // 8) Fill ^suspend
    llvm::dbgs() << "Filling ^suspend\n";
    builder.setInsertionPointToStart(suspendBlock);
    builder.create<async::CoroEndOp>(loc, handleValue);
    builder.create<func::ReturnOp>(loc);

    // 9) Erase old TaskOp
    llvm::dbgs() << "Removing the old IdempotentTaskOp \n";
    taskOp.erase();
  }
};

} // namespace mlir::intermittent