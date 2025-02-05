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
      } else if (useCount.getUseCount(definingOp) == 1 && newIndex >= 0 &&
                 ops[newIndex] == definingOp) {
        // This is a value defined by an operation that is used only once
        // and the operation is the previous operation in the block.
        newIndex--;
      } else {
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
    rewriter.setInsertionPointAfter(op);
    auto localOp = rewriter.create<LocalOp>(op->getLoc(), op->getResult(0));
    // we assume that the op has one operand
    rewriter.replaceAllUsesExcept(op->getResult(0), localOp.getResult(),
                                  localOp);
  }

private:
  IntroduceLocalAnalysis &introduceLocal;
};

class IntroduceLocals : public impl::IntroduceLocalsBase<IntroduceLocals> {
public:
  using impl::IntroduceLocalsBase<IntroduceLocals>::IntroduceLocalsBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    IntroduceLocalAnalysis introduceLocal(module);

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

Type convertSsaWasmTypeToWasmType(Type type, MLIRContext *ctx) {
  if (isa<WasmIntegerType>(type)) {
    auto bitWidth = cast<WasmIntegerType>(type).getBitWidth();
    return IntegerType::get(ctx, bitWidth);
  } else if (isa<WasmFloatType>(type)) {
    auto bitWidth = cast<WasmFloatType>(type).getBitWidth();
    if (bitWidth == 32) {
      return FloatType::getF32(ctx);
    } else if (bitWidth == 64) {
      return FloatType::getF64(ctx);
    } else {
      assert(false && "Unsupported float type");
    }
  } else {
    assert(false && "Unsupported type");
  }
}

class StackifyTypeConverter : public TypeConverter {
public:
  StackifyTypeConverter(MLIRContext *ctx) {
    addConversion([ctx](WasmIntegerType type) -> Type {
      return wasm::LocalType::get(ctx, convertSsaWasmTypeToWasmType(type, ctx));
    });
    addConversion([ctx](WasmFloatType type) -> Type {
      return wasm::LocalType::get(ctx, convertSsaWasmTypeToWasmType(type, ctx));
    });
  }
};

} // namespace

class Stackify : public impl::StackifyBase<Stackify> {
public:
  using impl::StackifyBase<Stackify>::StackifyBase;

  void runOnOperation() final {
    auto module = getOperation();
    UseCountAnalysis useCount(module);
    LocalIndexAnalysis localIndex(module);
    IRRewriter rewriter(module.getContext());
    StackifyTypeConverter typeConverter(module.getContext());

    module.walk([&](FuncOp funcOp) {
      for (auto &block : funcOp.getBody().getBlocks()) {
        stackifyBlock(funcOp, &block, useCount, localIndex, rewriter);
      }
      convertFuncOp(funcOp, rewriter, typeConverter);
    });
  }

private:
  void convertFuncOp(FuncOp funcOp, IRRewriter &rewriter,
                     TypeConverter &typeConverter) {
    TypeConverter::SignatureConversion signatureConverter(
        funcOp.getFunctionType().getNumInputs());
    for (const auto &inputType :
         enumerate(funcOp.getFunctionType().getInputs())) {
      signatureConverter.addInputs(
          inputType.index(), typeConverter.convertType(inputType.value()));
    }

    // we should return i32 for memref types
    llvm::SmallVector<Type, 4> newResultTypes;
    for (auto resultType : funcOp.getFunctionType().getResults()) {
      if (isa<MemRefType>(resultType)) {
        newResultTypes.push_back(rewriter.getI32Type());
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    auto newFuncType = rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), newResultTypes);
    rewriter.setInsertionPoint(funcOp);
    auto newFuncOp = rewriter.create<wasm::WasmFuncOp>(
        funcOp.getLoc(), funcOp.getName(), newFuncType);

    for (Block &block : funcOp.getBody()) {
      // Convert block argument types
      for (BlockArgument arg : block.getArguments()) {
        Type newType = typeConverter.convertType(arg.getType());
        if (!newType) {
          llvm::dbgs() << "Failed to convert block argument type\n";
          assert(false && "Failed to convert block argument type");
        }
        arg.setType(newType);
      }
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // create locals for the function arguments
    Block &entryBlock = newFuncOp.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);
    vector<Attribute> typesAttr;
    for (auto inputType : newFuncType.getInputs()) {
      typesAttr.push_back(
          TypeAttr::get(cast<wasm::LocalType>(inputType).getInner()));
    }
    ArrayRef<Attribute> types(typesAttr);
    rewriter.create<wasm::LocalOp>(funcOp.getLoc(),
                                   rewriter.getArrayAttr(types));

    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.eraseOp(funcOp);
  }
  void stackifyBlock(FuncOp funcOp, Block *block,
                     UseCountAnalysis &useCountAnalysis,
                     LocalIndexAnalysis &localIndexAnalysis,
                     IRRewriter &rewriter) {
    vector<Operation *> ops;
    for (auto &op : *block) {
      ops.push_back(&op);
    }

    int index = ops.size() - 1;

    while (index >= 0) {
      stackify(funcOp, index, ops, useCountAnalysis, localIndexAnalysis,
               rewriter);
    }
  }

  // This modifies index in place
  Operation *stackify(FuncOp funcOp, int &index, vector<Operation *> ops,
                      UseCountAnalysis &useCountAnalysis,
                      LocalIndexAnalysis &localIndexAnalysis,
                      IRRewriter &rewriter) {
    Operation *currentOp = ops[index];
    index--;

    Operation *newOp =
        stackifyOperation(currentOp, localIndexAnalysis, rewriter);
    rewriter.setInsertionPoint(newOp);

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
        int localIndex = localIndexAnalysis.getLocalIndex(funcOp, operand);
        rewriter.create<wasm::LocalGetOp>(newOp->getLoc(),
                                          rewriter.getIndexAttr(localIndex));

      } else if (isa<LocalOp>(definingOp)) {
        int localIndex =
            localIndexAnalysis.getLocalIndex(funcOp, definingOp->getResult(0));
        rewriter.create<wasm::LocalGetOp>(newOp->getLoc(),
                                          rewriter.getIndexAttr(localIndex));
      } else {
        Operation *newOp = stackify(funcOp, index, ops, useCountAnalysis,
                                    localIndexAnalysis, rewriter);
        rewriter.setInsertionPoint(newOp);
      }
    }
    return newOp;
  }

  Operation *stackifyOperation(Operation *op,
                               LocalIndexAnalysis &localIndexAnalysis,
                               IRRewriter &rewriter) {
    Operation *newOp;
    rewriter.setInsertionPoint(op);
    if (isa<AddOp>(op)) {
      TypeAttr typeAttr = TypeAttr::get(convertSsaWasmTypeToWasmType(
          op->getResult(0).getType(), op->getContext()));
      newOp = rewriter.create<wasm::AddOp>(op->getLoc(), typeAttr);
    } else if (isa<ConstantOp>(op)) {
      newOp = rewriter.create<wasm::ConstantOp>(
          op->getLoc(), cast<ConstantOp>(op).getValue());
    } else if (isa<LocalOp>(op)) {
      FuncOp funcOp = op->getParentOfType<FuncOp>();
      newOp = rewriter.create<wasm::LocalSetOp>(
          op->getLoc(), rewriter.getIndexAttr(localIndexAnalysis.getLocalIndex(
                            funcOp, op->getResult(0))));
    } else if (isa<ReturnOp>(op)) {
      newOp = rewriter.create<wasm::WasmReturnOp>(op->getLoc());
    } else {
      newOp = nullptr;
    }
    return newOp;
  }
};
} // namespace mlir::ssawasm
