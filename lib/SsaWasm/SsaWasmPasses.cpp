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
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "SsaWasm/ConversionPatterns/ArithToSsaWasm.h"
#include "SsaWasm/ConversionPatterns/FuncToSsaWasm.h"
#include "SsaWasm/SsaWasmPasses.h"
#include "SsaWasm/SsaWasmTypeConverter.h"
#include "Wasm/WasmOps.h"
#include <vector>

using namespace std;

namespace mlir::ssawasm {
#define GEN_PASS_DEF_CONVERTTOSSAWASM
#define GEN_PASS_DEF_INTRODUCELOCALS
#define GEN_PASS_DEF_STACKIFY
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

namespace {

class UseCountAnalysis {
public:
  UseCountAnalysis(ModuleOp module) {
    module.walk([&](FuncOp funcOp) {
      for (auto &block : funcOp.getBody().getBlocks()) {
        for (auto &op : block) {
          size_t numUsers =
              std::distance(op.getUsers().begin(), op.getUsers().end());
          useCount[&op] = numUsers;
        }
      }
    });
  }
  int getUseCount(Operation *op) const { return useCount.at(op); }

private:
  DenseMap<Operation *, int> useCount;
};

class IntroduceLocalAnalysis {
public:
  IntroduceLocalAnalysis(ModuleOp module) {
    UseCountAnalysis useCount(module);

    module.walk([&](FuncOp funcOp) {
      for (auto &block : funcOp.getBody().getBlocks()) {
        analyzeBlock(&block, useCount);
      }
    });
  }
  SmallVector<Operation *> getLocalRequiredOps() const {
    return localRequiredOps;
  }
  bool isLocalRequired(Operation *op) const {
    return std::find(localRequiredOps.begin(), localRequiredOps.end(), op) !=
           localRequiredOps.end();
  }

private:
  void analyzeBlock(Block *block, UseCountAnalysis &useCount) {
    vector<Operation *> ops;
    for (auto &op : *block) {
      ops.push_back(&op);
    }

    int index = ops.size() - 1;

    while (index >= 0) {
      index = traverseTree(index, ops, useCount);
    }
  }
  int traverseTree(int index, vector<Operation *> &ops,
                   UseCountAnalysis &useCount) {
    Operation *currentOp = ops[index];
    llvm::dbgs() << "Index: " << index << "\n";
    llvm::dbgs() << "Current operation: ";
    currentOp->print(llvm::dbgs());
    llvm::dbgs() << "\n";
    if (currentOp->getNumOperands() == 0) {
      return index - 1;
    }

    int newIndex = index - 1;
    for (int operandIdx = currentOp->getNumOperands() - 1; operandIdx >= 0;
         operandIdx--) {
      Value operand = currentOp->getOperand(operandIdx);
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp) {
        assert(isa<BlockArgument>(operand) && "Expected a block argument");
        llvm::dbgs() << "Block argument\n";
      } else if (useCount.getUseCount(definingOp) == 1 && newIndex >= 0 &&
                 ops[newIndex] == definingOp) {
        // This is a value defined by an operation that is used only once
        // and the operation is the previous operation in the block.
        llvm::dbgs() << "Single use operation\n";
        newIndex--;
      } else {
        llvm::dbgs() << "Local required operation\n";
        // We should introduce a local for this operation.
        if (std::find(localRequiredOps.begin(), localRequiredOps.end(),
                      definingOp) == localRequiredOps.end()) {
          localRequiredOps.push_back(definingOp);
        }
        break;
      }
    }
    return newIndex;
  }
  SmallVector<Operation *> localRequiredOps;
};

struct IntroduceLocalPattern : public RewritePattern {
  IntroduceLocalPattern(MLIRContext *context,
                        IntroduceLocalAnalysis &introduceLocal)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        introduceLocal(introduceLocal) {}

  LogicalResult match(Operation *op) const override {
    assert(op->getDialect() ==
           op->getContext()->getLoadedDialect<ssawasm::SsaWasmDialect>());

    if (introduceLocal.isLocalRequired(op)) {
      return success();
    }
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    llvm::dbgs() << "Rewrite operation\n";
    op->dump();

    rewriter.setInsertionPointAfter(op);
    auto localOp = rewriter.create<LocalOp>(op->getLoc(), op->getResult(0));
    // we assume that the op has one operand
    rewriter.replaceAllUsesExcept(op->getResult(0), localOp.getResult(),
                                  localOp);
  }

private:
  IntroduceLocalAnalysis &introduceLocal;
};

} // namespace

class IntroduceLocals : public impl::IntroduceLocalsBase<IntroduceLocals> {
public:
  using impl::IntroduceLocalsBase<IntroduceLocals>::IntroduceLocalsBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    IntroduceLocalAnalysis introduceLocal(module);

    for (auto op : introduceLocal.getLocalRequiredOps()) {
      llvm::dbgs() << "Local required operation\n";
      op->dump();
    }

    // introduce locals
    RewritePatternSet patterns(context);
    patterns.add<IntroduceLocalPattern>(context, introduceLocal);
    walkAndApplyPatterns(module, std::move(patterns));
  }
};

class LocalIndexAnalysis {
public:
  LocalIndexAnalysis(ModuleOp module) {
    module.walk([&](FuncOp funcOp) {
      int index = 0;
      // each wasm function argument is treated as a local
      for (auto &arg : funcOp.getArguments()) {
        localIndex[funcOp][arg] = index;
        index++;
      }

      for (auto &block : funcOp.getBody().getBlocks()) {
        for (auto &op : block) {
          if (isa<LocalOp>(op)) {
            localIndex[funcOp][op.getResult(0)] = index;
            index++;
          }
        }
      }
    });
  }
  int getLocalIndex(FuncOp funcOp, Value value) const {
    return localIndex.at(funcOp).at(value);
  }

private:
  DenseMap<FuncOp, DenseMap<Value, int>> localIndex;
};

class Stackify : public impl::StackifyBase<Stackify> {
public:
  using impl::StackifyBase<Stackify>::StackifyBase;

  void runOnOperation() final {
    auto module = getOperation();
    UseCountAnalysis useCount(module);
    LocalIndexAnalysis localIndex(module);
    OpBuilder builder(module.getContext());

    module.walk([&](FuncOp funcOp) {
      llvm::dbgs() << "Stackifying function\n";
      funcOp.dump();
      for (auto &block : funcOp.getBody().getBlocks()) {
        llvm::dbgs() << "Stackifying block\n";
        block.dump();
        stackifyBlock(funcOp, &block, useCount, localIndex, builder);
      }
      // TODO: convert funcop and add local declarations
    });
  }

private:
  void stackifyBlock(FuncOp funcOp, Block *block,
                     UseCountAnalysis &useCountAnalysis,
                     LocalIndexAnalysis &localIndexAnalysis,
                     OpBuilder &builder) {
    vector<Operation *> ops;
    for (auto &op : *block) {
      ops.push_back(&op);
    }

    int index = ops.size() - 1;

    while (index >= 0) {
      stackify(funcOp, index, ops, useCountAnalysis, localIndexAnalysis,
               builder);
    }
  }

  // This modifies index in place
  Operation *stackify(FuncOp funcOp, int &index, vector<Operation *> ops,
                      UseCountAnalysis &useCountAnalysis,
                      LocalIndexAnalysis &localIndexAnalysis,
                      OpBuilder &builder) {
    Operation *currentOp = ops[index];
    llvm::dbgs() << "Stackifying operation\n";
    currentOp->dump();
    index--;

    Operation *newOp =
        stackifyOperation(currentOp, localIndexAnalysis, builder);
    builder.setInsertionPoint(newOp);

    if (isa<LocalOp>(currentOp)) {
      currentOp->erase();
      return newOp;
    }

    SmallVector<Value> operands;
    for (int operandIdx = currentOp->getNumOperands() - 1; operandIdx >= 0;
         operandIdx--) {
      operands.push_back(currentOp->getOperand(operandIdx));
    }
    currentOp->erase();

    for (auto operand : operands) {
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp) {
        assert(isa<BlockArgument>(operand) && "Expected a block argument");
        llvm::dbgs() << "Block argument. creating localget\n";
        int localIndex = localIndexAnalysis.getLocalIndex(funcOp, operand);
        builder.create<wasm::LocalGetOp>(newOp->getLoc(),
                                         builder.getIndexAttr(localIndex));

      } else if (isa<LocalOp>(definingOp)) {
        definingOp->dump();
        llvm::dbgs() << "Local operation. creating localget\n";
        int localIndex =
            localIndexAnalysis.getLocalIndex(funcOp, definingOp->getResult(0));
        builder.create<wasm::LocalGetOp>(newOp->getLoc(),
                                         builder.getIndexAttr(localIndex));
      } else {
        definingOp->dump();
        llvm::dbgs() << "stackify success\n";
        Operation *newOp = stackify(funcOp, index, ops, useCountAnalysis,
                                    localIndexAnalysis, builder);
        builder.setInsertionPoint(newOp);
      }
    }
    return newOp;
  }

  Operation *stackifyOperation(Operation *op,
                               LocalIndexAnalysis &localIndexAnalysis,
                               OpBuilder &builder) {
    Operation *newOp;
    builder.setInsertionPoint(op);
    if (isa<AddOp>(op)) {
      TypeAttr typeAttr = TypeAttr::get(op->getResult(0).getType());
      newOp = builder.create<wasm::AddOp>(op->getLoc(), typeAttr);
    } else if (isa<ConstantOp>(op)) {
      newOp = builder.create<wasm::ConstantOp>(op->getLoc(),
                                               cast<ConstantOp>(op).getValue());
    } else if (isa<LocalOp>(op)) {
      FuncOp funcOp = op->getParentOfType<FuncOp>();
      newOp = builder.create<wasm::LocalSetOp>(
          op->getLoc(), builder.getIndexAttr(localIndexAnalysis.getLocalIndex(
                            funcOp, op->getResult(0))));
    } else if (isa<ReturnOp>(op)) {
      newOp = builder.create<wasm::WasmReturnOp>(op->getLoc());
    } else {
      newOp = nullptr;
    }
    return newOp;
  }
};
} // namespace mlir::ssawasm
