//===- SsaWasmPasses.cpp - SsaWasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "SsaWasm/ConversionPatterns/ArithToSsaWasm.h"
#include "SsaWasm/ConversionPatterns/FuncToSsaWasm.h"
#include "SsaWasm/ConversionPatterns/MathToSsaWasm.h"
#include "SsaWasm/ConversionPatterns/MemRefToSsaWasm.h"
#include "SsaWasm/ConversionPatterns/ScfToSsaWasm.h"
#include "SsaWasm/SsaWasmPasses.h"
#include "SsaWasm/SsaWasmTypeConverter.h"
#include "Wasm/WasmOps.h"
#include <map>
#include <vector>

using namespace std;

namespace mlir::ssawasm {
#define GEN_PASS_DEF_CONVERTMATHTOSSAWASM
#define GEN_PASS_DEF_CONVERTARITHTOSSAWASM
#define GEN_PASS_DEF_CONVERTFUNCTOSSAWASM
#define GEN_PASS_DEF_CONVERTMEMREFTOSSAWASM
#define GEN_PASS_DEF_CONVERTSCFTOSSAWASM
#define GEN_PASS_DEF_INTRODUCELOCALS
#define GEN_PASS_DEF_CONVERTSSAWASMTOWASM
#define GEN_PASS_DEF_CONVERTSSAWASMGLOBALTOWASM
#include "SsaWasm/SsaWasmPasses.h.inc"

class ConvertMathToSsaWasm
    : public impl::ConvertMathToSsaWasmBase<ConvertMathToSsaWasm> {
public:
  using impl::ConvertMathToSsaWasmBase<
      ConvertMathToSsaWasm>::ConvertMathToSsaWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    SsaWasmTypeConverter typeConverter(context);
    ConversionTarget target(*context);

    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateMathToSsaWasmPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class ConvertArithToSsaWasm
    : public impl::ConvertArithToSsaWasmBase<ConvertArithToSsaWasm> {
public:
  using impl::ConvertArithToSsaWasmBase<
      ConvertArithToSsaWasm>::ConvertArithToSsaWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    SsaWasmTypeConverter typeConverter(context);
    ConversionTarget target(*context);

    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateArithToSsaWasmPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class ConvertFuncToSsaWasm
    : public impl::ConvertFuncToSsaWasmBase<ConvertFuncToSsaWasm> {
public:
  using impl::ConvertFuncToSsaWasmBase<
      ConvertFuncToSsaWasm>::ConvertFuncToSsaWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    SsaWasmTypeConverter typeConverter(context);
    ConversionTarget target(*context);

    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateFuncToSsaWasmPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class ConvertMemRefToSsaWasm
    : public impl::ConvertMemRefToSsaWasmBase<ConvertMemRefToSsaWasm> {
public:
  using impl::ConvertMemRefToSsaWasmBase<
      ConvertMemRefToSsaWasm>::ConvertMemRefToSsaWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    SsaWasmTypeConverter typeConverter(context);
    ConversionTarget target(*context);
    BaseAddressAnalysis baseAddressAnalysis(module);

    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateMemRefToSsaWasmPatterns(typeConverter, patterns,
                                    baseAddressAnalysis);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class ConvertScfToSsaWasm
    : public impl::ConvertScfToSsaWasmBase<ConvertScfToSsaWasm> {
public:
  using impl::ConvertScfToSsaWasmBase<
      ConvertScfToSsaWasm>::ConvertScfToSsaWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    SsaWasmTypeConverter typeConverter(context);
    ConversionTarget target(*context);

    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateScfToSsaWasmPatterns(typeConverter, patterns);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

namespace {

class UseCountAnalysis {
public:
  UseCountAnalysis(ModuleOp module) {
    module.walk([&](FuncOp funcOp) { analyzeRegion(funcOp.getBody()); });
  }
  int getUseCount(Operation *op) const { return useCount.at(op); }

private:
  void analyzeRegion(Region &region) {
    for (auto &block : region.getBlocks()) {
      for (auto &op : block) {
        // Count direct users of this operation
        size_t numUsers =
            std::distance(op.getUsers().begin(), op.getUsers().end());
        useCount[&op] = numUsers;

        // Recursively analyze any nested regions
        for (Region &nestedRegion : op.getRegions()) {
          analyzeRegion(nestedRegion);
        }
      }
    }
  }

  DenseMap<Operation *, int> useCount;
};

// Returns the operation that effectively defines a value by looking through
// AsPointerOp and AsMemRefOp wrappers. When a value is defined by an
// AsPointerOp or AsMemRefOp, this function returns the operation that defines
// the underlying memref instead, since AsPointerOp and AsMemRefOp are just
// type conversion wrappers that don't create new data.
//
// Example:
//   %1 = some_op ... : memref<...>
//   %2 = ssawasm.as_pointer %1 : memref<...> to ptr
//   getUnderlyingValue(%2) returns %1
Value getUnderlyingValue(Value value) {
  auto definingOp = value.getDefiningOp();
  if (!definingOp) {
    return value;
  }
  if (isa<UnrealizedConversionCastOp>(definingOp)) {
    return getUnderlyingValue(definingOp->getOperand(0));
  }
  if (auto asPointerOp = dyn_cast<AsPointerOp>(definingOp)) {
    return getUnderlyingValue(asPointerOp.getValue());
  }
  if (auto asMemRefOp = dyn_cast<AsMemRefOp>(definingOp)) {
    return getUnderlyingValue(asMemRefOp.getValue());
  }
  return value;
}

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
  vector<Operation *> getLocalRequiredOps() const { return localRequiredOps; }
  bool isLocalRequired(Operation *op) const {
    return std::find(localRequiredOps.begin(), localRequiredOps.end(), op) !=
           localRequiredOps.end();
  }
  bool isLocalGetRequired(Operation *op, int operandIdx) {
    return std::find(localGetRequiredOps[op].begin(),
                     localGetRequiredOps[op].end(),
                     operandIdx) != localGetRequiredOps[op].end();
  }
  vector<int> getLocalGetRequiredOperandIndices(Operation *op) {
    return localGetRequiredOps[op];
  }

private:
  // TODO: We have to analyze block recursively
  void analyzeBlock(Block *block, UseCountAnalysis &useCount) {
    vector<Operation *> ops;
    for (auto &op : *block) {
      ops.push_back(&op);
    }

    for (int index = ops.size() - 1; index >= 0; index--) {
      traverseTree(ops, index, useCount);
    }
  }
  void traverseTree(vector<Operation *> ops, int index,
                    UseCountAnalysis &useCount) {
    Operation *op = ops[index];

    // skip as_pointer operations
    // we should not introduce locals for these operations
    if (isa<AsPointerOp>(op) || isa<AsMemRefOp>(op) ||
        isa<UnrealizedConversionCastOp>(op)) {
      return;
    }

    for (int operandIdx = op->getNumOperands() - 1; operandIdx >= 0;
         operandIdx--) {
      Value operand = op->getOperand(operandIdx);
      Value underlyingValue = getUnderlyingValue(operand);
      Operation *definingOp = underlyingValue.getDefiningOp();
      if (!definingOp) {
        // FIX THIS: go to definingOp as far as possible
        assert((isa<BlockArgument>(underlyingValue) ||
                isa<AsPointerOp>(underlyingValue.getDefiningOp()) ||
                isa<AsMemRefOp>(underlyingValue.getDefiningOp())) &&
               "Expected a block argument or an AsPointerOp or AsMemRefOp");

        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getParentBlock() !=
              &blockArg.getParentRegion()->front()) {
            // if this is not the function argument
            // This must be placed in the first position of the block
            continue;
          }
        }

        if (!isa<LocalSetOp>(op) || operandIdx != 0) {
          localGetRequiredOps[op].push_back(operandIdx);
        }
      } else if (isa<LocalDeclOp>(definingOp)) {
        // if the defining operation is already a local, we do not need to
        // introduce a new local
        // let's introduce a local_get for the operand only
        if (!isa<LocalSetOp>(op) || operandIdx != 0) {
          localGetRequiredOps[op].push_back(operandIdx);
        }
      } else if (useCount.getUseCount(definingOp) == 1 && index > 0 &&
                 ops[index - 1] == definingOp) {
        // This is a value defined by an operation that is used only once
        // and the operation is the previous operation in the block.
      } else {
        // We should introduce a local for this operation.
        if (std::find(localRequiredOps.begin(), localRequiredOps.end(),
                      definingOp) == localRequiredOps.end() &&
            !isa<LocalDeclOp>(definingOp)) {
          localRequiredOps.push_back(definingOp);
        }
        if (!isa<LocalSetOp>(op) || operandIdx != 0) {
          localGetRequiredOps[op].push_back(operandIdx);
        }
      }
    }

    for (auto &region : op->getRegions()) {
      for (auto &block : region.getBlocks()) {
        analyzeBlock(&block, useCount);
      }
    }
  }
  vector<Operation *> localRequiredOps;
  map<Operation *, vector<int>> localGetRequiredOps;
};

struct IntroduceLocalPattern : public RewritePattern {
  IntroduceLocalPattern(MLIRContext *context,
                        IntroduceLocalAnalysis &introduceLocal)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        introduceLocal(introduceLocal) {}

  LogicalResult match(Operation *op) const override {
    if (op->getDialect() !=
            op->getContext()->getLoadedDialect<ssawasm::SsaWasmDialect>() &&
        op->getDialect() !=
            op->getContext()->getLoadedDialect<wasm::WasmDialect>() &&
        op->getName().getDialectNamespace() !=
            BuiltinDialect::getDialectNamespace()) {
      llvm::errs() << "Unsupported operation: ";
      op->dump();
      llvm::errs() << "\n";
      return failure();
    }

    if (introduceLocal.isLocalRequired(op)) {
      return success();
    }
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    auto local =
        rewriter
            .create<LocalDeclOp>(
                op->getLoc(),
                LocalType::get(op->getContext(), op->getResult(0).getType()))
            .getResult();
    auto castedLocal = rewriter
                           .create<UnrealizedConversionCastOp>(
                               op->getLoc(), op->getResult(0).getType(), local)
                           .getResult(0);
    auto localSetOp =
        rewriter.create<LocalSetOp>(op->getLoc(), local, op->getResult(0));
    // we assume that the op has one operand
    rewriter.replaceAllUsesExcept(op->getResult(0), castedLocal, localSetOp);
  }

private:
  IntroduceLocalAnalysis &introduceLocal;
};

unsigned computeOffset(Operation *op, int operandIdx) {
  Value underlyingValue = getUnderlyingValue(op->getOperand(operandIdx));
  Operation *definingOp = underlyingValue.getDefiningOp();
  if (!definingOp) {
    return 0;
  }
  if (isa<LocalGetOp>(definingOp) || isa<AsPointerOp>(definingOp) ||
      isa<AsMemRefOp>(definingOp)) {
    return 1;
  }
  int offset = 1;
  for (unsigned i = 0; i < definingOp->getNumOperands(); i++) {
    offset += computeOffset(definingOp, i);
  }
  return offset;
}

struct IntroduceLocalGetPattern : public RewritePattern {
  IntroduceLocalGetPattern(MLIRContext *context,
                           IntroduceLocalAnalysis &introduceLocal)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        introduceLocal(introduceLocal) {}

  LogicalResult match(Operation *op) const override {
    assert(op->getDialect() ==
               op->getContext()->getLoadedDialect<ssawasm::SsaWasmDialect>() ||
           op->getDialect() ==
               op->getContext()->getLoadedDialect<wasm::WasmDialect>() ||
           op->getName().getDialectNamespace() ==
               BuiltinDialect::getDialectNamespace());

    if (introduceLocal.getLocalGetRequiredOperandIndices(op).size() > 0) {
      return success();
    }
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    unsigned offset = 0;
    for (int operandIdx = op->getNumOperands() - 1; operandIdx >= 0;
         operandIdx--) {
      if (introduceLocal.isLocalGetRequired(op, operandIdx)) {
        auto underlyingValue = getUnderlyingValue(op->getOperand(operandIdx));
        Type underlyingValueType = underlyingValue.getType();

        rewriter.setInsertionPoint(op);
        auto ip = rewriter.getInsertionPoint();
        for (unsigned i = 0; i < offset; i++) {
          rewriter.setInsertionPoint(rewriter.getInsertionBlock(), --ip);
        }
        Value castedUnderlyingValue;
        Type resultType;
        if (auto localType = dyn_cast<LocalType>(underlyingValueType)) {
          castedUnderlyingValue = underlyingValue;
          resultType = localType.getInnerType();
        } else {
          castedUnderlyingValue =
              rewriter
                  .create<UnrealizedConversionCastOp>(
                      op->getLoc(),
                      LocalType::get(op->getContext(), underlyingValueType),
                      underlyingValue)
                  .getResult(0);
          resultType = underlyingValueType;
        }
        // if the result type is memref,
        // we convert it to integer type to make operations
        // on it legal
        if (isa<MemRefType>(resultType)) {
          resultType = IntegerType::get(op->getContext(), 32);
        }
        auto localGet = rewriter.create<ssawasm::LocalGetOp>(
            op->getLoc(), resultType, castedUnderlyingValue);
        // replace the operand with the
        op->setOperand(operandIdx, localGet.getResult());
        offset += 1;

      } else {
        offset += computeOffset(op, operandIdx);
      }
    }
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

    // introduce local gets
    RewritePatternSet localGetPatterns(context);
    localGetPatterns.add<IntroduceLocalGetPattern>(context, introduceLocal);
    walkAndApplyPatterns(module, std::move(localGetPatterns));
  }
};

Type convertSsaWasmTypeToWasmType(Type type, MLIRContext *ctx) {
  if (isa<IntegerType>(type)) {
    auto bitWidth = cast<IntegerType>(type).getWidth();
    return IntegerType::get(ctx, bitWidth);
  } else if (isa<FloatType>(type)) {
    auto bitWidth = cast<FloatType>(type).getWidth();
    if (bitWidth == 32) {
      return FloatType::getF32(ctx);
    } else if (bitWidth == 64) {
      return FloatType::getF64(ctx);
    } else {
      assert(false && "Unsupported float type");
    }
  } else if (isa<MemRefType>(type)) {
    return IntegerType::get(ctx, 32);
  } else if (isa<WasmContinuationType>(type)) {
    // TODO: We should not hardcode this
    return wasm::ContinuationType::get(ctx, "ct", "ft");
  } else {
    type.dump();
    assert(false && "Unsupported type");
  }
}

class LocalIndexAnalysis {
public:
  LocalIndexAnalysis(ModuleOp module) {
    module.walk([&](FuncOp funcOp) {
      localTypes[funcOp.getName().str()] = vector<Type>();
      int index = 0;
      // each wasm function argument is treated as a local
      for (auto &arg : funcOp.getArguments()) {
        localIndex[funcOp.getName().str()][arg] = index;
        index++;
      }

      std::function<void(Block *)> traverse = [&](Block *block) {
        for (auto &op : *block) {
          if (isa<LocalDeclOp>(op)) {
            localIndex[funcOp.getName().str()][op.getResult(0)] = index;
            localTypes[funcOp.getName().str()].push_back(
                convertSsaWasmTypeToWasmType(
                    cast<LocalType>(op.getResult(0).getType()).getInnerType(),
                    op.getContext()));
            index++;
          }
          // Recursively traverse any nested regions
          for (Region &nestedRegion : op.getRegions()) {
            for (auto &nestedBlock : nestedRegion.getBlocks()) {
              traverse(&nestedBlock);
            }
          }
        }
      };

      for (auto &block : funcOp.getBody().getBlocks()) {
        traverse(&block);
      }
    });
  }
  int getLocalIndex(FuncOp funcOp, Value value) const {
    Value underlyingValue = getUnderlyingValue(value);
    auto funcIt = localIndex.find(funcOp.getName().str());
    assert(funcIt != localIndex.end() &&
           "Function not found in local index map");

    auto valueIt = funcIt->second.find(underlyingValue);
    if (valueIt == funcIt->second.end()) {
      llvm::errs() << "value not found in local index map for function:\n";
      funcOp.dump();
      value.dump();
      assert(false && "Value not found in local index map");
    }
    return valueIt->second;
  }
  vector<Type> getLocalTypes(string funcName) const {
    return localTypes.at(funcName);
  }

private:
  map<string, DenseMap<Value, int>> localIndex;
  map<string, vector<Type>> localTypes;
};

class SsaWasmToWasmTypeConverter : public TypeConverter {
public:
  SsaWasmToWasmTypeConverter(MLIRContext *ctx) {
    addConversion([ctx](IntegerType type) -> Type {
      return convertSsaWasmTypeToWasmType(type, ctx);
    });
    addConversion([ctx](FloatType type) -> Type {
      return convertSsaWasmTypeToWasmType(type, ctx);
    });
    addConversion([ctx](MemRefType type) -> Type {
      return convertSsaWasmTypeToWasmType(type, ctx);
    });
    addConversion([ctx](LocalType type) -> Type {
      return convertSsaWasmTypeToWasmType(type.getInnerType(), ctx);
    });
    addConversion([ctx](WasmContinuationType type) -> Type {
      // FIXME: We should not hardcode this
      assert(type.getId() == "ct");
      return wasm::ContinuationType::get(ctx, "ct", "ft");
    });
  }
};

} // namespace

namespace {
static const uint8_t s_is_char_escaped[] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static const char s_hexdigits[] = "0123456789abcdef";

std::string generateStr(const void *bytes, size_t length) {
  const uint8_t *u8_data = static_cast<const uint8_t *>(bytes);

  std::stringstream ss;

  for (size_t i = 0; i < length; ++i) {
    uint8_t c = u8_data[i];
    if (s_is_char_escaped[c]) {
      ss << "\\";
      ss << s_hexdigits[c >> 4];
      ss << s_hexdigits[c & 0xf];
    } else {
      ss << c;
    }
  }

  return ss.str();
}

class DataOpLowering : public OpConversionPattern<DataOp> {
public:
  using OpConversionPattern<DataOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto denseElementsAttr =
            dyn_cast<DenseElementsAttr>(adaptor.getInitialValue())) {
      auto rawData = denseElementsAttr.getRawData();
      std::string bytes = generateStr(rawData.data(), rawData.size());

      rewriter.replaceOpWithNewOp<wasm::DataOp>(
          op, adaptor.getSymName(), adaptor.getBaseAddr(),
          rewriter.getStringAttr(bytes.c_str()),
          TypeAttr::get(adaptor.getMemref()));

      return success();
    }
    return failure();
  }
};

class GetDataOpLowering : public OpConversionPattern<GetDataOp> {
public:
  using OpConversionPattern<GetDataOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetDataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto dataOp = SymbolTable::lookupNearestSymbolFrom<ssawasm::DataOp>(
        op, rewriter.getStringAttr(op.getName()));
    auto addressConstantOp = rewriter.create<ConstantOp>(
        loc, rewriter.getI32IntegerAttr(dataOp.getBaseAddr()));
    auto asMemRefOp = rewriter.create<AsMemRefOp>(
        loc, op.getResult().getType(), addressConstantOp.getResult());

    rewriter.replaceOp(op, asMemRefOp.getResult());

    return success();
  }
};

class TagOpLowering : public OpConversionPattern<TagOp> {
public:
  using OpConversionPattern<TagOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<wasm::TagOp>(op, op.getSymName());
    return success();
  }
};

class RecContFuncDeclOpLowering
    : public OpConversionPattern<RecContFuncDeclOp> {
public:
  using OpConversionPattern<RecContFuncDeclOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(RecContFuncDeclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<wasm::RecContFuncDeclOp>(op, op.getFuncTypeId(),
                                                         op.getContTypeId());
    return success();
  }
};

class ElemDeclFuncOpLowering : public OpConversionPattern<ElemDeclFuncOp> {
public:
  using OpConversionPattern<ElemDeclFuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ElemDeclFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<wasm::ElemDeclareFuncOp>(op,
                                                         adaptor.getFuncName());
    return success();
  }
};

class FuncTypeDeclOpLowering : public OpConversionPattern<FuncTypeDeclOp> {
public:
  using OpConversionPattern<FuncTypeDeclOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FuncTypeDeclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<wasm::FuncTypeDeclOp>(
        op, adaptor.getFuncTypeId(), adaptor.getFuncType());
    return success();
  }
};

class ContTypeDeclOpLowering : public OpConversionPattern<ContTypeDeclOp> {
public:
  using OpConversionPattern<ContTypeDeclOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ContTypeDeclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<wasm::ContinuationTypeDeclOp>(
        op,
        wasm::ContinuationType::get(op.getContext(), adaptor.getContTypeId(),
                                    adaptor.getFuncTypeId()));
    return success();
  }
};

class ImportFuncOpLowering : public OpConversionPattern<ImportFuncOp> {
public:
  using OpConversionPattern<ImportFuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ImportFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert function type
    auto funcType = adaptor.getFuncType();
    SmallVector<Type, 4> convertedInputTypes;
    SmallVector<Type, 4> convertedResultTypes;

    for (auto inputType : funcType.getInputs()) {
      convertedInputTypes.push_back(getTypeConverter()->convertType(inputType));
    }

    for (auto resultType : funcType.getResults()) {
      convertedResultTypes.push_back(
          getTypeConverter()->convertType(resultType));
    }

    auto newFuncType =
        rewriter.getFunctionType(convertedInputTypes, convertedResultTypes);

    rewriter.replaceOpWithNewOp<wasm::ImportFuncOp>(op, adaptor.getFuncName(),
                                                    newFuncType);
    return success();
  }
};

class ConvertSsaWasmGlobalToWasm
    : public impl::ConvertSsaWasmGlobalToWasmBase<ConvertSsaWasmGlobalToWasm> {
public:
  using impl::ConvertSsaWasmGlobalToWasmBase<
      ConvertSsaWasmGlobalToWasm>::ConvertSsaWasmGlobalToWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    RewritePatternSet globalDependencyLoweringPattern(context);
    SsaWasmToWasmTypeConverter typeConverter(context);
    globalDependencyLoweringPattern.add<GetDataOpLowering>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addLegalDialect<wasm::WasmDialect>();
    target.addIllegalOp<GetDataOp>();

    if (failed(applyPartialConversion(
            module, target, std::move(globalDependencyLoweringPattern)))) {
      signalPassFailure();
    }

    RewritePatternSet globalLoweringPattern(context);
    globalLoweringPattern
        .add<DataOpLowering, TagOpLowering, RecContFuncDeclOpLowering,
             ElemDeclFuncOpLowering, ContTypeDeclOpLowering,
             ImportFuncOpLowering, FuncTypeDeclOpLowering>(typeConverter,
                                                           context);
    target.addIllegalOp<DataOp>();
    target.addIllegalOp<TagOp>();
    target.addIllegalOp<RecContFuncDeclOp>();
    target.addIllegalOp<ElemDeclFuncOp>();
    target.addIllegalOp<ContTypeDeclOp>();
    target.addIllegalOp<FuncTypeDeclOp>();
    target.addIllegalOp<ImportFuncOp>();
    if (failed(applyPartialConversion(module, target,
                                      std::move(globalLoweringPattern)))) {
      signalPassFailure();
    }
  }
};
} // namespace

class ConvertSsaWasmToWasm
    : public impl::ConvertSsaWasmToWasmBase<ConvertSsaWasmToWasm> {
public:
  using impl::ConvertSsaWasmToWasmBase<
      ConvertSsaWasmToWasm>::ConvertSsaWasmToWasmBase;

  void runOnOperation() final {
    auto module = getOperation();
    UseCountAnalysis useCount(module);
    LocalIndexAnalysis localIndex(module);
    IRRewriter rewriter(module.getContext());
    SsaWasmToWasmTypeConverter typeConverter(module.getContext());

    int newLabelIndex = 0;

    module.walk([&](FuncOp funcOp) {
      for (auto &block : funcOp.getBody().getBlocks()) {
        convertBlock(funcOp, &block, localIndex, rewriter, newLabelIndex);
      }
      convertFuncOp(funcOp, rewriter, typeConverter, localIndex);
    });
  }

private:
  void convertFuncOp(FuncOp funcOp, IRRewriter &rewriter,
                     TypeConverter &typeConverter,
                     LocalIndexAnalysis &localIndexAnalysis) {
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
          assert(false && "Failed to convert block argument type");
        }
        arg.setType(newType);
      }
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // create locals
    Block &entryBlock = newFuncOp.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);
    vector<Attribute> typeAttrs;
    for (auto type : localIndexAnalysis.getLocalTypes(funcOp.getName().str())) {
      typeAttrs.push_back(TypeAttr::get(type));
    }
    if (typeAttrs.size() > 0) {
      ArrayRef<Attribute> types(typeAttrs);
      rewriter.create<wasm::LocalOp>(funcOp.getLoc(),
                                     rewriter.getArrayAttr(types));
    }

    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.eraseOp(funcOp);
  }

  void convertBlock(FuncOp funcOp, Block *block,
                    LocalIndexAnalysis &localIndexAnalysis,
                    IRRewriter &rewriter, int &newLabelIndex) {
    vector<Operation *> ops;
    for (auto &op : *block) {
      ops.push_back(&op);
    }
    // we need to convert the operations in reverse order to avoid
    // Assertion failed: (op->use_empty() && "expected 'op' to have no uses"),
    // function eraseOp
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      convertOperation(funcOp, *op, localIndexAnalysis, rewriter,
                       newLabelIndex);
    }
  }

  void convertOperation(FuncOp funcOp, Operation *op,
                        LocalIndexAnalysis &localIndexAnalysis,
                        IRRewriter &rewriter, int &newLabelIndex) {

    if (isa<UnrealizedConversionCastOp>(op)) {
      rewriter.eraseOp(op);
      return;
    }

    if (op->getDialect() !=
        op->getContext()->getLoadedDialect<ssawasm::SsaWasmDialect>()) {
      return;
    }

    rewriter.setInsertionPoint(op);
    if (isa<OnStackOp>(op)) {
      // do nothing
    } else if (isa<AddOp>(op)) {
      rewriter.create<wasm::AddOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<SubOp>(op)) {
      rewriter.create<wasm::SubOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<MulOp>(op)) {
      rewriter.create<wasm::MulOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<MinOp>(op)) {
      rewriter.create<wasm::FMinOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<MaxOp>(op)) {
      rewriter.create<wasm::FMaxOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<RemUIOp>(op)) {
      rewriter.create<wasm::IRemUOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<RemSIOp>(op)) {
      rewriter.create<wasm::IRemSOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<NegFOp>(op)) {
      rewriter.create<wasm::NegFOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      if (isa<IndexType>(constantOp.getValue().getType())) {
        rewriter.create<wasm::ConstantOp>(
            op->getLoc(),
            rewriter.getI32IntegerAttr(cast<IntegerAttr>(constantOp.getValue())
                                           .getValue()
                                           .getSExtValue()));
      } else {
        rewriter.create<wasm::ConstantOp>(op->getLoc(), constantOp.getValue());
      }
    } else if (isa<ConvertSIToFPOp>(op)) {
      rewriter.create<wasm::ConvertSIToFPOp>(
          op->getLoc(),
          convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                       op->getContext()),
          convertSsaWasmTypeToWasmType(op->getOperand(0).getType(),
                                       op->getContext()));
    } else if (isa<TruncateFPToSIOp>(op)) {
      rewriter.create<wasm::TruncateFPToSIOp>(
          op->getLoc(),
          convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                       op->getContext()),
          convertSsaWasmTypeToWasmType(op->getOperand(0).getType(),
                                       op->getContext()));
    } else if (isa<ShlOp>(op)) {
      rewriter.create<wasm::IShlOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<ShrSOp>(op)) {
      rewriter.create<wasm::IShrSOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<SqrtOp>(op)) {
      rewriter.create<wasm::FSqrtOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<LocalDeclOp>(op)) {
      // This is handled by ConvertSsaWasmToWasm. Do nothing.
    } else if (auto localSetOp = dyn_cast<LocalSetOp>(op)) {
      rewriter.create<wasm::LocalSetOp>(
          op->getLoc(), rewriter.getIndexAttr(localIndexAnalysis.getLocalIndex(
                            funcOp, localSetOp.getLocal())));
    } else if (auto localGetOp = dyn_cast<LocalGetOp>(op)) {
      rewriter.create<wasm::LocalGetOp>(
          op->getLoc(), rewriter.getIndexAttr(localIndexAnalysis.getLocalIndex(
                            funcOp, localGetOp.getLocal())));
    } else if (isa<ReturnOp>(op)) {
      rewriter.create<wasm::WasmReturnOp>(op->getLoc());
    } else if (auto blockLoopOp = dyn_cast<BlockLoopOp>(op)) {
      convertBlockLoopOp(funcOp, blockLoopOp, rewriter, localIndexAnalysis,
                         newLabelIndex);
    } else if (auto blockBlockOp = dyn_cast<BlockBlockOp>(op)) {
      convertBlockBlockOp(funcOp, blockBlockOp, rewriter, localIndexAnalysis,
                          newLabelIndex);
    } else if (auto ifElseOp = dyn_cast<IfElseOp>(op)) {
      convertIfElseOp(funcOp, ifElseOp, rewriter, localIndexAnalysis,
                      newLabelIndex);
    } else if (auto ifElseTerminatorOp = dyn_cast<IfElseTerminatorOp>(op)) {
      rewriter.create<wasm::IfElseEndOp>(op->getLoc());
    } else if (isa<ILeUOp>(op)) {
      TypeAttr typeAttr = TypeAttr::get(convertSsaWasmTypeToWasmType(
          op->getResult(0).getType(), op->getContext()));
      rewriter.create<wasm::ILeUOp>(op->getLoc(), typeAttr);
    } else if (isa<FLeOp>(op)) {
      rewriter.create<wasm::FLeOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            op->getOperand(0).getType(), op->getContext())));
    } else if (isa<EqOp>(op)) {
      rewriter.create<wasm::EqOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            op->getResult(0).getType(), op->getContext())));
    } else if (isa<LtSOp>(op)) {
      rewriter.create<wasm::LtSOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            op->getResult(0).getType(), op->getContext())));
    } else if (isa<GeSOp>(op)) {
      rewriter.create<wasm::GeSOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            op->getResult(0).getType(), op->getContext())));
    } else if (isa<AndIOp>(op)) {
      rewriter.create<wasm::AndIOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            op->getResult(0).getType(), op->getContext())));
    } else if (isa<DivSOp>(op)) {
      rewriter.create<wasm::IDivSOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            op->getResult(0).getType(), op->getContext())));
    } else if (isa<DivFOp>(op)) {
      rewriter.create<wasm::FDivOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            op->getResult(0).getType(), op->getContext())));
    } else if (auto callOp = dyn_cast<CallOp>(op)) {
      rewriter.create<wasm::CallOp>(op->getLoc(), callOp.getCallee());
    } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
      rewriter.create<wasm::LoadOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            op->getResult(0).getType(), op->getContext())));
    } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
      rewriter.create<wasm::StoreOp>(
          op->getLoc(),
          TypeAttr::get(convertSsaWasmTypeToWasmType(
              storeOp.getValue().getType(), storeOp.getContext())));
    } else if (auto suspendOp = dyn_cast<SuspendOp>(op)) {
      rewriter.create<wasm::SuspendOp>(op->getLoc(), suspendOp.getTag());
    } else if (auto contNewOp = dyn_cast<ContNewOp>(op)) {
      rewriter.create<wasm::ContNewOp>(op->getLoc(),
                                       contNewOp.getResult().getType().getId());
    } else if (auto funcRefOp = dyn_cast<FuncRefOp>(op)) {
      rewriter.create<wasm::FuncRefOp>(op->getLoc(), funcRefOp.getFunc());
    } else if (auto nullContRefOp = dyn_cast<NullContRefOp>(op)) {
      rewriter.create<wasm::NullContRefOp>(
          op->getLoc(), nullContRefOp.getResult().getType().getId());
    } else if (auto selectOp = dyn_cast<SelectOp>(op)) {
      rewriter.create<wasm::SelectOp>(
          op->getLoc(), TypeAttr::get(convertSsaWasmTypeToWasmType(
                            selectOp.getResult().getType(), op->getContext())));
    } else if (isa<AsPointerOp>(op) || isa<AsMemRefOp>(op)) {
      // do nothing
      // This is already handled by the SsaWasmDataToLocal pass
    } else {
      llvm::errs() << "Unsupported operation: " << op->getName() << "\n";
    }
    rewriter.eraseOp(op);
    return;
  }
  void convertBlockLoopOp(FuncOp funcOp, BlockLoopOp blockLoopOp,
                          IRRewriter &rewriter,
                          LocalIndexAnalysis &localIndexAnalysis,
                          int &newLabelIndex) {
    Location loc = blockLoopOp.getLoc();
    std::string blockLabel = "block_" + std::to_string(newLabelIndex++);
    // TODO: Support arguments
    auto blockOp = rewriter.create<wasm::BlockOp>(loc, blockLabel);

    // move the entry block of the loop into the block
    rewriter.moveBlockBefore(blockLoopOp.getEntryBlock(), &blockOp.getBody(),
                             blockOp.getBody().begin());
    Block *blockBody = &blockOp.getBody().front();
    rewriter.eraseOp(blockBody->getTerminator());
    convertBlock(funcOp, blockBody, localIndexAnalysis, rewriter,
                 newLabelIndex);

    // loop inside block
    rewriter.setInsertionPointToEnd(blockBody);
    std::string loopLabel = "loop_" + std::to_string(newLabelIndex++);
    auto loopOp = rewriter.create<wasm::LoopOp>(loc, loopLabel);
    rewriter.create<wasm::BlockEndOp>(loc);

    Block *loopBody = rewriter.createBlock(&loopOp.getBody());
    rewriter.setInsertionPointToStart(loopBody);

    for (Block &block : blockLoopOp.getRegion()) {
      if (&block != blockLoopOp.getExitBlock()) {
        moveAndMergeBlockInBlockLoop(&block, loopBody, rewriter, loopLabel,
                                     blockLabel);
      }
    }

    convertBlock(funcOp, loopBody, localIndexAnalysis, rewriter, newLabelIndex);

    rewriter.setInsertionPointToEnd(loopBody);
    rewriter.create<wasm::LoopEndOp>(loc);
  }
  void moveAndMergeBlockInBlockLoop(Block *from, Block *to,
                                    IRRewriter &rewriter, std::string loopLabel,
                                    std::string blockLabel) {
    rewriter.setInsertionPointToEnd(to);
    vector<Operation *> opsToMove;
    for (auto &op : *from) {
      opsToMove.push_back(&op);
    }
    for (auto &op : opsToMove) {
      Location loc = op->getLoc();
      if (isa<TempBranchOp>(op)) {
        // do not clone this
        rewriter.eraseOp(op);
      } else if (auto blockLoopBranchOp = dyn_cast<BlockLoopBranchOp>(op)) {
        if (blockLoopBranchOp.isBranchingToBegin()) {
          rewriter.create<wasm::BranchOp>(loc, loopLabel);
        } else {
          rewriter.create<wasm::BranchOp>(loc, blockLabel);
        }
        rewriter.eraseOp(op);
      } else if (auto blockLoopCondBranchOp =
                     dyn_cast<BlockLoopCondBranchOp>(op)) {
        if (blockLoopCondBranchOp.isBranchingToBegin()) {
          // need to add local get here?
          rewriter.create<wasm::CondBranchOp>(loc, loopLabel);
        } else {
          // need to add local get here?
          rewriter.create<wasm::CondBranchOp>(loc, blockLabel);
        }
        rewriter.eraseOp(op);
      } else {
        rewriter.moveOpBefore(op, to, to->end());
      }
    }
  }

  void convertBlockBlockOp(FuncOp funcOp, BlockBlockOp blockBlockOp,
                           IRRewriter &rewriter,
                           LocalIndexAnalysis &localIndexAnalysis,
                           int &newLabelIndex) {
    Location loc = blockBlockOp.getLoc();

    auto it = blockBlockOp->getRegion(0).begin();
    Block *outerEntryBlock = &*it;
    ++it;
    // Block *innerEntryBlock = &*it;
    ++it;
    auto revIt = blockBlockOp->getRegion(0).rbegin();
    Block *outerExitBlock = &*revIt;
    ++revIt;
    Block *innerExitBlock = &*revIt;

    // TODO: Support arguments
    auto outerBlockLabel = "block_" + std::to_string(newLabelIndex++);
    auto outerBlockOp = rewriter.create<wasm::BlockOp>(
        loc, rewriter.getStringAttr(outerBlockLabel));
    rewriter.moveBlockBefore(outerEntryBlock, &outerBlockOp.getBody(),
                             outerBlockOp.getBody().begin());
    Block *outerBlockBody = &outerBlockOp.getBody().front();
    rewriter.eraseOp(outerBlockBody->getTerminator());
    convertBlock(funcOp, outerBlockBody, localIndexAnalysis, rewriter,
                 newLabelIndex);

    rewriter.setInsertionPointToEnd(outerBlockBody);

    auto innerBlockLabel = "block_" + std::to_string(newLabelIndex++);
    auto innerBlockOp = rewriter.create<wasm::BlockOp>(
        loc, rewriter.getStringAttr(innerBlockLabel));
    // if innerExitBlock has arguments, set return_types of innerBlockOp to the
    // argument types
    SmallVector<Attribute> typeAttrs;
    for (Attribute attr : blockBlockOp.getInnerBlockResultTypes()) {
      TypeAttr typeAttr = cast<TypeAttr>(attr);
      typeAttrs.push_back(TypeAttr::get(convertSsaWasmTypeToWasmType(
          typeAttr.getValue(), innerBlockOp.getContext())));
    }
    innerBlockOp.setReturnTypesAttr(rewriter.getArrayAttr(typeAttrs));

    Block *tempBlock = new Block();
    moveAndMergeBlocksInBlockBlock(blockBlockOp, innerExitBlock, tempBlock,
                                   rewriter, innerBlockLabel, outerBlockLabel);
    convertBlock(funcOp, tempBlock, localIndexAnalysis, rewriter,
                 newLabelIndex);
    vector<Operation *> opsToMove;
    for (auto &op : *tempBlock) {
      opsToMove.push_back(&op);
    }
    for (auto &op : opsToMove) {
      rewriter.moveOpBefore(op, outerBlockBody, outerBlockBody->end());
    }

    Block *innerBlockBody = rewriter.createBlock(&innerBlockOp.getBody());
    rewriter.setInsertionPointToStart(innerBlockBody);

    SmallVector<Block *> blocksToMove;
    for (Block &block : blockBlockOp.getRegion()) {
      if (&block != outerExitBlock && &block != innerExitBlock) {
        blocksToMove.push_back(&block);
      }
    }
    // sort blocks by topological order
    std::sort(blocksToMove.begin(), blocksToMove.end(),
              [&](Block *a, Block *b) {
                for (Block *successor : a->getSuccessors()) {
                  if (successor == b) {
                    return true;
                  }
                }
                return false;
              });
    for (Block *block : blocksToMove) {
      moveAndMergeBlocksInBlockBlock(blockBlockOp, block, innerBlockBody,
                                     rewriter, innerBlockLabel,
                                     outerBlockLabel);
    }

    convertBlock(funcOp, innerBlockBody, localIndexAnalysis, rewriter,
                 newLabelIndex);
    rewriter.setInsertionPointToEnd(innerBlockBody);
    rewriter.create<wasm::BlockEndOp>(loc);
    rewriter.setInsertionPointToEnd(outerBlockBody);
    rewriter.create<wasm::BlockEndOp>(loc);
  }

  void moveAndMergeBlocksInBlockBlock(BlockBlockOp blockBlockOp, Block *from,
                                      Block *to, IRRewriter &rewriter,
                                      std::string innerBlockLabel,
                                      std::string outerBlockLabel) {
    rewriter.setInsertionPointToEnd(to);
    vector<Operation *> opsToMove;
    for (auto &op : *from) {
      opsToMove.push_back(&op);
    }
    for (auto &op : opsToMove) {
      Location loc = op->getLoc();
      if (isa<TempBranchOp>(op)) {
        // do not clone this
        rewriter.eraseOp(op);
      } else if (auto blockBlockBranchOp = dyn_cast<BlockBlockBranchOp>(op)) {
        if (blockBlockBranchOp.isBranchingToOuter()) {
          rewriter.create<wasm::BranchOp>(loc, outerBlockLabel);
        } else {
          rewriter.create<wasm::BranchOp>(loc, innerBlockLabel);
        }
        rewriter.eraseOp(op);
        // FIXME: We should not assume that resumeOp is always
        // in a BlockBlockOp
      } else if (auto resumeOp = dyn_cast<ResumeOp>(op)) {
        // FIXME: Move this to somewhere else

        Block *outerBlock = &blockBlockOp.getRegion().getBlocks().back();
        std::string onTagLabel;
        if (resumeOp.getOnTag() == outerBlock) {
          onTagLabel = outerBlockLabel;
        } else {
          onTagLabel = innerBlockLabel;
        }
        rewriter.create<wasm::ResumeOp>(op->getLoc(),
                                        resumeOp.getCont().getType().getId(),
                                        resumeOp.getTag(), onTagLabel);
        rewriter.eraseOp(op);
      } else {
        rewriter.moveOpBefore(op, to, to->end());
      }
    }
  }
  void convertIfElseOp(FuncOp funcOp, IfElseOp ifElseOp, IRRewriter &rewriter,
                       LocalIndexAnalysis &localIndexAnalysis,
                       int &newLabelIndex) {
    Location loc = ifElseOp.getLoc();
    MLIRContext *context = ifElseOp.getContext();

    // Create wasm if-else operation with the same result types
    auto wasmIfElseOp = rewriter.create<wasm::IfElseOp>(loc);
    if (ifElseOp.getResults().size() > 0) {
      SmallVector<Attribute> typeAttrs;
      for (Value result : ifElseOp.getResults()) {
        typeAttrs.push_back(TypeAttr::get(
            convertSsaWasmTypeToWasmType(result.getType(), context)));
      }
      wasmIfElseOp.setReturnTypesAttr(rewriter.getArrayAttr(typeAttrs));
    }

    // Move the then region
    rewriter.inlineRegionBefore(ifElseOp.getThenRegion(),
                                wasmIfElseOp.getThenRegion(),
                                wasmIfElseOp.getThenRegion().end());

    // Move the else region
    rewriter.inlineRegionBefore(ifElseOp.getElseRegion(),
                                wasmIfElseOp.getElseRegion(),
                                wasmIfElseOp.getElseRegion().end());

    // Convert blocks in then region
    Block &thenBlock = wasmIfElseOp.getThenRegion().front();
    convertBlock(funcOp, &thenBlock, localIndexAnalysis, rewriter,
                 newLabelIndex);

    // Convert blocks in else region
    Block &elseBlock = wasmIfElseOp.getElseRegion().front();
    convertBlock(funcOp, &elseBlock, localIndexAnalysis, rewriter,
                 newLabelIndex);
  }
};
} // namespace mlir::ssawasm
