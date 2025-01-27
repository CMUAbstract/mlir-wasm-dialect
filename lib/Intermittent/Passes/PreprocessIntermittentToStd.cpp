//===- PreprocessIntermittentToStd.cpp - Preprocess Intermittent to Std
//-----------------*- C++ -*-===//

#include "Intermittent/IntermittentPasses.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_PREPROCESSINTERMITTENTTOSTD
#include "Intermittent/IntermittentPasses.h.inc"

struct PreprocessIntermittentToStd
    : public impl::PreprocessIntermittentToStdBase<
          PreprocessIntermittentToStd> {
  using impl::PreprocessIntermittentToStdBase<
      PreprocessIntermittentToStd>::PreprocessIntermittentToStdBase;
  void runOnOperation() final {
    // The Operation we are running on is ModuleOp because
    // in the .td file we wrote: Pass<"create-main-fn", "ModuleOp">
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    std::string entryTaskName = entryTask.getValue();

    // Collect all tasks (and see if there's an "entry" task).
    SmallVector<StringRef, 4> taskNames;

    // FIXME: Here we are assuming that all functions are tasks
    moduleOp.walk([&](func::FuncOp funcOp) {
      auto name = funcOp.getName();
      taskNames.push_back(name);
    });

    // Look up or create a @main function
    func::FuncOp mainFunc = moduleOp.lookupSymbol<func::FuncOp>("main");
    if (!mainFunc) {
      // Create it if it doesn't exist
      auto funcType = builder.getFunctionType({}, {});
      mainFunc = func::FuncOp::create(moduleOp.getLoc(), "main", funcType);
      moduleOp.push_back(mainFunc);
    }

    // If @main is empty, build the body
    if (mainFunc.empty()) {
      Block *entryBlock = mainFunc.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      SmallVector<Value, 4> entryTaskArgs;

      // Insert async.runtime.create for each discovered task
      for (auto &name : taskNames) {
        // Typically, you'd specify return types or arguments to the create op.
        // Adjust as necessary for your environment.
        auto runtimeCreateOp = builder.create<async::RuntimeCreateOp>(
            mainFunc.getLoc(), async::TokenType::get(context));
        runtimeCreateOp->setAttr("task_name", builder.getStringAttr(name));
        entryTaskArgs.push_back(runtimeCreateOp.getResult());
      }

      // call entry task
      builder.create<func::CallOp>(mainFunc.getLoc(), entryTaskName,
                                   TypeRange{} /*results*/, entryTaskArgs
                                   /*arguments*/);

      // add a return
      builder.create<func::ReturnOp>(mainFunc.getLoc());
    }
  }
};

} // namespace mlir::intermittent