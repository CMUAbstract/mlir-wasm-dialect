//===- SsaWasmPasses.cpp - SsaWasm passes -----------------*- C++ -*-===//
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

#include "SsaWasm/ConversionPatterns/ArithToSsaWasm.h"
#include "SsaWasm/ConversionPatterns/FuncToSsaWasm.h"
#include "SsaWasm/SsaWasmPasses.h"
#include "SsaWasm/SsaWasmTypeConverter.h"
#include <vector>

using namespace std;

namespace mlir::ssawasm {
#define GEN_PASS_DEF_CONVERTTOSSAWASM
#include "SsaWasm/SsaWasmPasses.h.inc"

class ConvertToSsaWasm : public impl::ConvertToSsaWasmBase<ConvertToSsaWasm> {
public:
  using impl::ConvertToSsaWasmBase<ConvertToSsaWasm>::ConvertToSsaWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<func::FuncDialect>();

    RewritePatternSet patterns(context);
    SsaWasmTypeConverter typeConverter(context);
    populateArithToSsaWasmPatterns(typeConverter, patterns);
    populateFuncToSsaWasmPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class UseCount {
public:
  UseCount(ModuleOp module) {
    module.walk([&](FuncOp funcOp) {
      for (auto &block : funcOp.getBody().getBlocks()) {
        for (auto &op : block) {
          useCount[&op] = op.getUsers().size();
        }
      }
    });
  }
  int getUseCount(Operation *op) const { return useCount.at(op); }

private:
  DenseMap<Operation *, int> useCount;
};

namespace {}

class ReplaceMultiUseOpsWithLocals
    : public impl::ReplaceMultiUseOpsWithLocalsBase<
          ReplaceMultiUseOpsWithLocals> {
public:
  using impl::ReplaceMultiUseOpsWithLocalsBase<
      ReplaceMultiUseOpsWithLocals>::ReplaceMultiUseOpsWithLocalsBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder builder(context);

    UseCount useCount(module);
    // TODO
  }
};

class ConvertSsaWasmToWasm
    : public impl::ConvertSsaWasmToWasmBase<ConvertSsaWasmToWasm> {
public:
  using impl::ConvertSsaWasmToWasmBase<
      ConvertSsaWasmToWasm>::ConvertSsaWasmToWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder builder(context);

    UseCount useCount(module);

    module.walk([&](FuncOp funcOp) {
      for (auto &block : funcOp.getBody().getBlocks()) {
        stackifyBlock(builder, &block);
      }
    });
    // TODO: Convert SsaWasm loop ops to Wasm loop ops
    // TODO: Convert ssawasm::Func to wasm::WasmFunc
  }

private:
  void stackifyBlock(OpBuilder &builder, Block *block, UseCount &useCount) {
    vector<Operation *> ops;
    for (auto &op : *block) {
      ops.push_back(&op);
    }

    int index = ops.size() - 1;

    while (index >= 0) {
      recoverTree(builder, index, ops, useCount);
    }
  }

  void recoverTree(OpBuilder &builder, int &index, vector<Operation *> &ops,
                   UseCount &useCount) {
    Operation *currentOp = ops[index];
    SmallVector<Value, 4> nestedOperands;

    ops.pop_back();
    for (int operandIdx = currentOp->getNumOperands() - 1; operandIdx >= 0;
         --operandIdx) {
      Value operand = currentOp->getOperand(operandIdx);
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp) {
        assert(isa<BlockArgument>(operand) && "Expected a block argument");
        // Block argument can be used directly.
        nestedOperands.insert(nestedOperands.begin(), operand);
      } else if (useCount.getUseCount(definingOp) == 1 && index > 0 &&
                 ops[index - 1] == definingOp) {
        // This is a value defined by an operation that is used only once
        // and the operation is the previous operation in the block.
        index--;
        Operation *nestedOp = recoverTree(builder, index, ops, useCount);
        nestedOperands.insert(nestedOperands.begin(), nestedOp->getResult(0));
      } else {
        // TODO
      }
    }

    stackifyOperation(builder, currentOp, nestedOperands);
    // TODO
  }
  void stackifyOperation(OpBuilder &builder, Operation *op,
                         SmallVector<Value, 4> &operands) {
    // TODO
  }
};
} // namespace mlir::ssawasm
