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
#include "SsaWasm/ConversionPatterns/MemRefToSsaWasm.h"
#include "SsaWasm/ConversionPatterns/ScfToSsaWasm.h"
#include "SsaWasm/SsaWasmPasses.h"
#include "SsaWasm/SsaWasmTypeConverter.h"
#include "Wasm/WasmOps.h"
#include <vector>

using namespace std;

namespace mlir::ssawasm {
#define GEN_PASS_DEF_CONVERTTOSSAWASM
#define GEN_PASS_DEF_INTRODUCELOCALS
#define GEN_PASS_DEF_CONVERTSSAWASMTOWASM
#define GEN_PASS_DEF_SSAWASMDATATOLOCAL
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
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    SsaWasmTypeConverter typeConverter(context);
    populateArithToSsaWasmPatterns(typeConverter, patterns);
    populateFuncToSsaWasmPatterns(typeConverter, patterns);
    populateMemRefToSsaWasmPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // we need to apply scf conversion separately because calling
    // `replaceAllUsesWith()` on induction variable will cause the conversion to
    // fail with the error message "Assertion failed: (!impl->wasOpReplaced(op)
    // && "attempting to modify a replaced/erased op")"
    RewritePatternSet nextPatterns(context);
    target.addIllegalDialect<scf::SCFDialect>();
    populateScfToSsaWasmPatterns(typeConverter, nextPatterns);

    if (failed(
            applyPartialConversion(module, target, std::move(nextPatterns)))) {
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
// AsPointerOp wrappers. When a value is defined by an AsPointerOp, this
// function returns the operation that defines the underlying memref instead,
// since AsPointerOp is just a type conversion wrapper that doesn't create new
// data.
//
// Example:
//   %1 = some_op ... : memref<...>
//   %2 = ssawasm.as_pointer %1 : memref<...> to ptr
//   getUnderlyingDefiningOp(%2) returns the operation defining %1
Operation *getUnderlyingDefiningOp(Value value) {
  auto definingOp = value.getDefiningOp();
  if (!definingOp) {
    return nullptr;
  }
  if (auto asPointerOp = dyn_cast<AsPointerOp>(definingOp)) {
    return asPointerOp.getValue().getDefiningOp();
  }
  return definingOp;
}

Value getUnderlyingValue(Value value) {
  auto definingOp = value.getDefiningOp();
  if (!definingOp) {
    return value;
  }
  if (auto asPointerOp = dyn_cast<AsPointerOp>(definingOp)) {
    return asPointerOp.getValue();
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
  vector<int> getLocalRequiredOperandIndices(Operation *op) {
    return localGetRequiredOps[op];
  }

private:
  // TODO: We have to analyze block recursively
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
    int newIndex = index - 1;

    // skip as_pointer operations
    // we should not introduce locals for these operations
    if (isa<AsPointerOp>(currentOp)) {
      return newIndex;
    }

    for (int operandIdx = currentOp->getNumOperands() - 1; operandIdx >= 0;
         operandIdx--) {
      Value operand = currentOp->getOperand(operandIdx);
      Operation *definingOp = getUnderlyingDefiningOp(operand);
      if (!definingOp) {
        assert((isa<BlockArgument>(operand) ||
                isa<AsPointerOp>(operand.getDefiningOp())) &&
               "Expected a block argument or an AsPointerOp");
        localGetRequiredOps[currentOp].push_back(operandIdx);
      } else if (isa<LocalOp>(definingOp)) {
        // if the defining operation is already a local, we do not need to
        // introduce a new local
        if (!isa<LocalSetOp>(currentOp) || operandIdx != 0) {
          localGetRequiredOps[currentOp].push_back(operandIdx);
        }
      } else if (useCount.getUseCount(definingOp) == 1 && newIndex >= 0 &&
                 ops[newIndex] == definingOp) {
        // This is a value defined by an operation that is used only once
        // and the operation is the previous operation in the block.
      } else {
        // We should introduce a local for this operation.
        if (std::find(localRequiredOps.begin(), localRequiredOps.end(),
                      definingOp) == localRequiredOps.end() &&
            !isa<LocalOp>(definingOp)) {
          localRequiredOps.push_back(definingOp);
        }
        localGetRequiredOps[currentOp].push_back(operandIdx);
      }
    }

    for (auto &region : currentOp->getRegions()) {
      for (auto &block : region.getBlocks()) {
        analyzeBlock(&block, useCount);
      }
    }

    return newIndex;
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

struct IntroduceLocalGetPattern : public RewritePattern {
  IntroduceLocalGetPattern(MLIRContext *context,
                           IntroduceLocalAnalysis &introduceLocal)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        introduceLocal(introduceLocal) {}

  LogicalResult match(Operation *op) const override {
    assert(op->getDialect() ==
           op->getContext()->getLoadedDialect<ssawasm::SsaWasmDialect>());

    if (introduceLocal.getLocalRequiredOperandIndices(op).size() > 0) {
      return success();
    }
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    for (int operandIdx : introduceLocal.getLocalRequiredOperandIndices(op)) {
      auto underlyingValue = getUnderlyingValue(op->getOperand(operandIdx));

      // if the underlying value is of sswawasm<memref> type,
      // we convert it to ssawasm<integer> type to make operations
      // on it legal
      Type localType = underlyingValue.getType();
      if (isa<WasmMemRefType>(localType)) {
        localType = ssawasm::WasmIntegerType::get(op->getContext(), 32);
      }
      auto local = rewriter
                       .create<ssawasm::LocalGetOp>(op->getLoc(), localType,
                                                    underlyingValue)
                       .getResult();
      // replace the operand with the local
      op->setOperand(operandIdx, local);
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

      std::function<void(Block *)> traverse = [&](Block *block) {
        for (auto &op : *block) {
          if (isa<LocalOp>(op)) {
            localIndex[funcOp][op.getResult(0)] = index;
            index++;
          } else if (auto blockLoopOp = dyn_cast<BlockLoopOp>(op)) {
            for (auto &nestedBlock : blockLoopOp.getRegion()) {
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
    return localIndex.at(funcOp).at(getUnderlyingValue(value));
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
  } else if (isa<WasmMemRefType>(type)) {
    return IntegerType::get(ctx, 32);
  } else {
    assert(false && "Unsupported type");
  }
}

class SsaWasmToWasmTypeConverter : public TypeConverter {
public:
  SsaWasmToWasmTypeConverter(MLIRContext *ctx) {
    addConversion([ctx](WasmIntegerType type) -> Type {
      return wasm::LocalType::get(ctx, convertSsaWasmTypeToWasmType(type, ctx));
    });
    addConversion([ctx](WasmFloatType type) -> Type {
      return wasm::LocalType::get(ctx, convertSsaWasmTypeToWasmType(type, ctx));
    });
    addConversion([ctx](WasmMemRefType type) -> Type {
      return wasm::LocalType::get(ctx, convertSsaWasmTypeToWasmType(type, ctx));
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
          TypeAttr::get(adaptor.getType()));

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
    // FIXME: This might have been already converted to wasm::DataOp
    auto dataOp = SymbolTable::lookupNearestSymbolFrom<ssawasm::DataOp>(
        op, rewriter.getStringAttr(op.getName()));
    auto addressConstantOp = rewriter.create<ConstantOp>(
        loc, rewriter.getI32IntegerAttr(dataOp.getBaseAddr()));
    auto localOp = rewriter.create<LocalOp>(loc, addressConstantOp.getResult());

    rewriter.replaceOp(op, localOp.getResult());

    return success();
  }
};

class SsaWasmDataToLocal
    : public impl::SsaWasmDataToLocalBase<SsaWasmDataToLocal> {
public:
  using impl::SsaWasmDataToLocalBase<
      SsaWasmDataToLocal>::SsaWasmDataToLocalBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    RewritePatternSet getDataOpLoweringPattern(context);
    getDataOpLoweringPattern.add<GetDataOpLowering>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addLegalDialect<wasm::WasmDialect>();
    target.addIllegalOp<GetDataOp>();

    if (failed(applyPartialConversion(module, target,
                                      std::move(getDataOpLoweringPattern)))) {
      signalPassFailure();
    }

    RewritePatternSet dataOpLoweringPattern(context);
    dataOpLoweringPattern.add<DataOpLowering>(context);
    target.addIllegalOp<DataOp>();
    if (failed(applyPartialConversion(module, target,
                                      std::move(dataOpLoweringPattern)))) {
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
    for (auto &op : llvm::reverse(ops)) {
      convertOperation(funcOp, op, localIndexAnalysis, rewriter, newLabelIndex);
    }
  }

  void convertOperation(FuncOp funcOp, Operation *op,
                        LocalIndexAnalysis &localIndexAnalysis,
                        IRRewriter &rewriter, int &newLabelIndex) {

    if (op->getDialect() !=
        op->getContext()->getLoadedDialect<ssawasm::SsaWasmDialect>()) {
      return;
    }

    rewriter.setInsertionPoint(op);
    if (isa<AddOp>(op)) {
      rewriter.create<wasm::AddOp>(
          op->getLoc(), convertSsaWasmTypeToWasmType(op->getResult(0).getType(),
                                                     op->getContext()));
    } else if (isa<MulOp>(op)) {
      rewriter.create<wasm::MulOp>(
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
    } else if (isa<LocalOp>(op)) {
      rewriter.create<wasm::LocalSetOp>(
          op->getLoc(), rewriter.getIndexAttr(localIndexAnalysis.getLocalIndex(
                            funcOp, op->getResult(0))));
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
    } else if (isa<ILeUOp>(op)) {
      TypeAttr typeAttr = TypeAttr::get(convertSsaWasmTypeToWasmType(
          op->getResult(0).getType(), op->getContext()));
      rewriter.create<wasm::ILeUOp>(op->getLoc(), typeAttr);
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
    } else if (isa<AsPointerOp>(op)) {
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
        moveAndMergeBlock(&block, loopBody, rewriter, loopLabel, blockLabel);
      }
    }

    convertBlock(funcOp, loopBody, localIndexAnalysis, rewriter, newLabelIndex);

    rewriter.setInsertionPointToEnd(loopBody);
    rewriter.create<wasm::LoopEndOp>(loc);
  }

  void moveAndMergeBlock(Block *from, Block *to, IRRewriter &rewriter,
                         std::string loopLabel, std::string blockLabel) {
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
};
} // namespace mlir::ssawasm
