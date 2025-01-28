#ifndef INTERMITTENT_UTILITY_H
#define INTERMITTENT_UTILITY_H

#include "Intermittent/IntermittentOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace mlir::intermittent {

static Value lookupOrCreateGlobalI32(ModuleOp module, OpBuilder &builder,
                                     StringRef globalName) {
  MLIRContext *ctx = builder.getContext();
  // Check if it already exists.
  if (auto globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
    // Use AddressOfOp to get a pointer to it.
    return builder.create<LLVM::AddressOfOp>(module.getLoc(),
                                             LLVM::LLVMPointerType::get(ctx),
                                             globalOp.getSymName());
  }

  // Otherwise, create a new global with an initial value of 0.
  auto i32Ty = IntegerType::get(ctx, 32);
  auto newGlobal =
      OpBuilder::atBlockBegin(module.getBody())
          .create<LLVM::GlobalOp>(module.getLoc(), i32Ty,
                                  /*isConstant=*/false, LLVM::Linkage::External,
                                  globalName, builder.getI32IntegerAttr(0),
                                  /*alignment=*/0,
                                  /*addr_space=*/0);

  // And return a pointer to it.
  return builder.create<LLVM::AddressOfOp>(
      module.getLoc(), LLVM::LLVMPointerType::get(ctx), newGlobal.getSymName());
}

static SmallVector<StringRef> collectTaskSymbols(ModuleOp module) {
  SmallVector<StringRef> taskNames;
  for (auto taskOp : module.getOps</*intermittent::*/ IdempotentTaskOp>()) {
    taskNames.push_back(taskOp.getSymName());
  }
  return taskNames;
}

} // namespace mlir::intermittent

#endif // INTERMITTENT_UTILITY_H