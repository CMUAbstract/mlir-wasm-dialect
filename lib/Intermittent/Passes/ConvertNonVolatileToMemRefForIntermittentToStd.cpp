//===- ConvertNonVolatileToMemRefForIntermittentToStd.cpp - Convert NonVolatile
// to MemRef -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Intermittent/IntermittentPasses.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::intermittent {
#define GEN_PASS_DEF_CONVERTNONVOLATILETOMEMREFFORINTERMITTENTTOSTD
#include "Intermittent/IntermittentPasses.h.inc"

struct ConvertNonVolatileToMemRefForIntermittentToStd
    : public impl::ConvertNonVolatileToMemRefForIntermittentToStdBase<
          ConvertNonVolatileToMemRefForIntermittentToStd> {
  using impl::ConvertNonVolatileToMemRefForIntermittentToStdBase<
      ConvertNonVolatileToMemRefForIntermittentToStd>::
      ConvertNonVolatileToMemRefForIntermittentToStdBase;
  void runOnOperation() final {
    static int globalCounter = 0;

    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    OpBuilder builder(context);
    // Create a memref.global for each NonVolatileNewOp
    DenseMap<Value, memref::GlobalOp> newToGlobal;
    moduleOp.walk([&](NonVolatileNewOp op) {
      builder.setInsertionPoint(op);
      auto memRefType = MemRefType::get({}, op.getType().getElementType());
      std::string globalName = "_nv_global_" + std::to_string(globalCounter++);
      auto globalOp = builder.create<memref::GlobalOp>(
          op.getLoc(), builder.getStringAttr(globalName),
          /*sym_visibility=*/builder.getStringAttr("private"),
          /*type=*/memRefType,
          /*initial_value=*/Attribute(), // no initial value
          /*constant=*/false,
          /*alignment=*/IntegerAttr());
      newToGlobal[op.getResult()] = globalOp;
    });

    llvm::dbgs() << "Created memref globals\n";

    // for each task
    moduleOp.walk([&](IdempotentTaskOp taskOp) {
      llvm::dbgs() << "Task: " << taskOp.getSymName() << "\n";
      builder.setInsertionPointToStart(&taskOp.getBody().front());
      llvm::dbgs() << "For all nonvolatile variables accessed in the task "
                      "body, create a memref.get_global\n";
      DenseMap<Value, Value> nonVolatileToMemRefGetGlobal;
      taskOp.walk([&](Operation *op) {
        if (isa<NonVolatileLoadOp>(op) || isa<NonVolatileStoreOp>(op)) {
          Value nonVolatile = op->getOperand(0);
          if (!nonVolatileToMemRefGetGlobal.count(nonVolatile)) {
            auto globalValue = newToGlobal[nonVolatile];
            auto memRefGlobal = builder.create<memref::GetGlobalOp>(
                op->getLoc(), globalValue.getType(), globalValue.getName());
            nonVolatileToMemRefGetGlobal[nonVolatile] = memRefGlobal;
          }
        }
      });

      llvm::dbgs() << "Replaced all nonvolatile loads and stores with "
                      "memref.load and memref.store\n";
      taskOp.walk([&](NonVolatileLoadOp loadOp) {
        builder.setInsertionPoint(loadOp);
        auto memRefGlobal = nonVolatileToMemRefGetGlobal[loadOp.getVar()];
        auto memRefLoad = builder.create<memref::LoadOp>(
            loadOp.getLoc(), memRefGlobal, ValueRange{} // empty index list
        );
        // replace all uses of the load op with the memref.load
        loadOp.getResult().replaceAllUsesWith(memRefLoad.getResult());
        loadOp.erase();
      });
      taskOp.walk([&](NonVolatileStoreOp storeOp) {
        builder.setInsertionPoint(storeOp);
        auto memRefGlobal = nonVolatileToMemRefGetGlobal[storeOp.getVar()];
        builder.create<memref::StoreOp>(storeOp.getLoc(), storeOp.getValue(),
                                        memRefGlobal,
                                        ValueRange{} // empty index list
        );
        storeOp.erase();
      });
    });

    llvm::dbgs() << "Removed all NonVolatileNewOps\n";
    moduleOp.dump();
    moduleOp.walk([&](NonVolatileNewOp newOp) { newOp->erase(); });
  }
};

} // namespace mlir::intermittent