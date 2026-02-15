//===- WAMIPasses.cpp - WAMI dialect passes ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements passes for the WAMI dialect and conversion passes
// to/from the upstream WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/WAMIPasses.h"
#include "WAMI/ConversionPatterns/WAMIConvertArith.h"
#include "WAMI/ConversionPatterns/WAMIConvertFunc.h"
#include "WAMI/ConversionPatterns/WAMIConvertMath.h"
#include "WAMI/ConversionPatterns/WAMIConvertMemref.h"
#include "WAMI/ConversionPatterns/WAMIConvertScf.h"
#include "WAMI/WAMIDialect.h"
#include "WAMI/WAMIOps.h"
#include "WAMI/WAMITypeConverter.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <cstring>
#include <optional>
#include <string>

namespace mlir::wami {

#define GEN_PASS_DEF_WAMICONVERTARITH
#define GEN_PASS_DEF_WAMICONVERTMATH
#define GEN_PASS_DEF_WAMICONVERTFUNC
#define GEN_PASS_DEF_WAMICONVERTMEMREF
#define GEN_PASS_DEF_WAMICONVERTSCF
#define GEN_PASS_DEF_WAMICONVERTALL
#define GEN_PASS_DEF_COROVERIFYINTRINSICS
#define GEN_PASS_DEF_CORONORMALIZE
#define GEN_PASS_DEF_COROTOWAMI
#define GEN_PASS_DEF_COROTOLLVM
#include "WAMI/WAMIPasses.h.inc"

//===----------------------------------------------------------------------===//
// WAMIConvertArith Pass
//===----------------------------------------------------------------------===//

class WAMIConvertArith : public impl::WAMIConvertArithBase<WAMIConvertArith> {
public:
  using impl::WAMIConvertArithBase<WAMIConvertArith>::WAMIConvertArithBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // WasmSSA and WAMI dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();
    target.addLegalDialect<WAMIDialect>();

    // Arith dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<arith::ArithDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertArithPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// WAMIConvertMath Pass
//===----------------------------------------------------------------------===//

class WAMIConvertMath : public impl::WAMIConvertMathBase<WAMIConvertMath> {
public:
  using impl::WAMIConvertMathBase<WAMIConvertMath>::WAMIConvertMathBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // WasmSSA and WAMI dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();
    target.addLegalDialect<WAMIDialect>();

    // Math dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<math::MathDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertMathPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// WAMIConvertFunc Pass
//===----------------------------------------------------------------------===//

class WAMIConvertFunc : public impl::WAMIConvertFuncBase<WAMIConvertFunc> {
public:
  using impl::WAMIConvertFuncBase<WAMIConvertFunc>::WAMIConvertFuncBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // WasmSSA dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();

    // Func dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<func::FuncDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertFuncPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// WAMIConvertScf Pass
//===----------------------------------------------------------------------===//

class WAMIConvertScf : public impl::WAMIConvertScfBase<WAMIConvertScf> {
public:
  using impl::WAMIConvertScfBase<WAMIConvertScf>::WAMIConvertScfBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // WasmSSA dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();

    // SCF dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<scf::SCFDialect>();

    // Arith and Func dialects are legal (may be used in loop bodies)
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertScfPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// WAMIConvertMemref Pass
//===----------------------------------------------------------------------===//

class WAMIConvertMemref
    : public impl::WAMIConvertMemrefBase<WAMIConvertMemref> {
public:
  using impl::WAMIConvertMemrefBase<WAMIConvertMemref>::WAMIConvertMemrefBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // Analyze module to assign base addresses to globals
    WAMIBaseAddressAnalysis baseAddressAnalysis(module);

    // WasmSSA and WAMI dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();
    target.addLegalDialect<WAMIDialect>();

    // MemRef dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<memref::MemRefDialect>();

    // Arith dialect is legal (used for address computation)
    target.addLegalDialect<arith::ArithDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertMemrefPatterns(typeConverter, patterns,
                                      baseAddressAnalysis);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// WAMIConvertAll Pass
//===----------------------------------------------------------------------===//

class WAMIConvertAll : public impl::WAMIConvertAllBase<WAMIConvertAll> {
public:
  using impl::WAMIConvertAllBase<WAMIConvertAll>::WAMIConvertAllBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // Analyze module to assign base addresses to globals
    WAMIBaseAddressAnalysis baseAddressAnalysis(module);

    // WasmSSA and WAMI dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();
    target.addLegalDialect<WAMIDialect>();

    // All source dialects are illegal (we want to convert them)
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    // Collect all conversion patterns
    RewritePatternSet patterns(context);
    populateWAMIConvertArithPatterns(typeConverter, patterns);
    populateWAMIConvertMathPatterns(typeConverter, patterns);
    populateWAMIConvertFuncPatterns(typeConverter, patterns);
    populateWAMIConvertScfPatterns(typeConverter, patterns);
    populateWAMIConvertMemrefPatterns(typeConverter, patterns,
                                      baseAddressAnalysis);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

namespace {

enum class CoroIntrinsicKind {
  Spawn,
  Resume,
  Yield,
  IsDone,
  Cancel,
};

struct CoroIntrinsicRef {
  CoroIntrinsicKind kind;
  std::string suffix;
};

struct KindUsage {
  Operation *spawnSite = nullptr;
  Operation *resumeSite = nullptr;
  SmallVector<Type> spawnArgTypes;
  SmallVector<Type> resumeArgTypes;
  SmallVector<Type> resumePayloadTypes;
};

struct RuntimeSlot {
  int64_t handleId = 0;
  FlatSymbolRefAttr srcContType;
  FlatSymbolRefAttr resumeContType;
  FlatSymbolRefAttr stateGlobal;
  FlatSymbolRefAttr contGlobal;
  SmallVector<Type> expectedResumeArgs;
  SmallVector<Type> expectedResumePayloads;
  std::string kind;
  std::string implSym;
};

struct KindRuntimeInfo {
  FlatSymbolRefAttr tagSym;
  FlatSymbolRefAttr helperSym;
  FlatSymbolRefAttr resumeContType;
  SmallVector<Type> resumeArgTypes;
  SmallVector<Type> payloadTypes;
};

static std::optional<CoroIntrinsicRef> parseCoroIntrinsicName(StringRef name) {
  if (!name.starts_with("coro."))
    return std::nullopt;

  StringRef rest = name.drop_front(strlen("coro."));
  auto parts = rest.split('.');
  if (parts.second.empty())
    return std::nullopt;

  CoroIntrinsicKind kind;
  if (parts.first == "spawn") {
    kind = CoroIntrinsicKind::Spawn;
  } else if (parts.first == "resume") {
    kind = CoroIntrinsicKind::Resume;
  } else if (parts.first == "yield") {
    kind = CoroIntrinsicKind::Yield;
  } else if (parts.first == "is_done") {
    kind = CoroIntrinsicKind::IsDone;
  } else if (parts.first == "cancel") {
    kind = CoroIntrinsicKind::Cancel;
  } else {
    return std::nullopt;
  }

  return CoroIntrinsicRef{kind, parts.second.str()};
}

static bool isCoroIntrinsicSymbol(StringRef name) {
  return parseCoroIntrinsicName(name).has_value();
}

static bool typeSequencesEqual(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::equal(lhs, rhs);
}

static std::string sanitizeSymbolPiece(StringRef raw) {
  std::string out;
  out.reserve(raw.size());
  for (char c : raw) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty())
    out = "anon";
  return out;
}

static Type parseWamiTypeOrEmit(Operation *at, StringRef typeText) {
  Type parsed = mlir::parseType(typeText, at->getContext());
  if (parsed)
    return parsed;
  at->emitError("failed to parse generated WAMI type: ") << typeText;
  return Type();
}

static Type buildContValueTypeOrEmit(Operation *at, FlatSymbolRefAttr contType,
                                     bool nullable) {
  std::string text = ("!wami.cont<@" + contType.getValue().str() +
                      (nullable ? ", true>" : ">"));
  return parseWamiTypeOrEmit(at, text);
}

static FailureOr<FlatSymbolRefAttr>
ensureWamiTypeFunc(ModuleOp module, Operation *diagOp, StringRef symName,
                   FunctionType type, OpBuilder &moduleBuilder) {
  Operation *existing = SymbolTable::lookupSymbolIn(module, symName);
  if (existing) {
    if (existing->getName().getStringRef() != "wami.type.func")
      return diagOp->emitError("symbol @")
             << symName << " already exists and is not wami.type.func";

    auto existingType = existing->getAttrOfType<TypeAttr>("type");
    if (!existingType || existingType.getValue() != type)
      return diagOp->emitError("symbol @")
             << symName << " has mismatched wami.type.func signature";
    return FlatSymbolRefAttr::get(module.getContext(), symName);
  }

  OperationState state(diagOp->getLoc(), "wami.type.func");
  state.addAttribute("sym_name", moduleBuilder.getStringAttr(symName));
  state.addAttribute("type", TypeAttr::get(type));
  moduleBuilder.create(state);
  return FlatSymbolRefAttr::get(module.getContext(), symName);
}

static FailureOr<FlatSymbolRefAttr>
ensureWamiTypeCont(ModuleOp module, Operation *diagOp, StringRef symName,
                   FlatSymbolRefAttr funcTypeRef, OpBuilder &moduleBuilder) {
  Operation *existing = SymbolTable::lookupSymbolIn(module, symName);
  if (existing) {
    if (existing->getName().getStringRef() != "wami.type.cont")
      return diagOp->emitError("symbol @")
             << symName << " already exists and is not wami.type.cont";

    auto existingRef = existing->getAttrOfType<FlatSymbolRefAttr>("func_type");
    if (!existingRef || existingRef != funcTypeRef)
      return diagOp->emitError("symbol @")
             << symName << " has mismatched wami.type.cont func_type";
    return FlatSymbolRefAttr::get(module.getContext(), symName);
  }

  OperationState state(diagOp->getLoc(), "wami.type.cont");
  state.addAttribute("sym_name", moduleBuilder.getStringAttr(symName));
  state.addAttribute("func_type", funcTypeRef);
  moduleBuilder.create(state);
  return FlatSymbolRefAttr::get(module.getContext(), symName);
}

static FailureOr<FlatSymbolRefAttr>
ensureWamiTag(ModuleOp module, Operation *diagOp, StringRef symName,
              FunctionType tagType, OpBuilder &moduleBuilder) {
  Operation *existing = SymbolTable::lookupSymbolIn(module, symName);
  if (existing) {
    if (existing->getName().getStringRef() != "wami.tag")
      return diagOp->emitError("symbol @")
             << symName << " already exists and is not wami.tag";

    auto existingType = existing->getAttrOfType<TypeAttr>("type");
    if (!existingType || existingType.getValue() != tagType)
      return diagOp->emitError("symbol @")
             << symName << " has mismatched wami.tag signature";
    return FlatSymbolRefAttr::get(module.getContext(), symName);
  }

  OperationState state(diagOp->getLoc(), "wami.tag");
  state.addAttribute("sym_name", moduleBuilder.getStringAttr(symName));
  state.addAttribute("type", TypeAttr::get(tagType));
  moduleBuilder.create(state);
  return FlatSymbolRefAttr::get(module.getContext(), symName);
}

static FailureOr<FlatSymbolRefAttr>
ensureContTypeForBoundPrefix(ModuleOp module, Operation *diagOp, StringRef kind,
                             FunctionType implType, unsigned boundPrefix,
                             OpBuilder &moduleBuilder) {
  if (boundPrefix > implType.getNumInputs())
    return diagOp->emitError(
        "bound argument count exceeds implementation arity");

  SmallVector<Type> dstInputs;
  dstInputs.append(implType.getInputs().begin() + boundPrefix,
                   implType.getInputs().end());
  SmallVector<Type> dstResults(implType.getResults().begin(),
                               implType.getResults().end());
  auto dstFuncType =
      FunctionType::get(module.getContext(), dstInputs, dstResults);

  std::string kindPart = sanitizeSymbolPiece(kind);
  std::string ftSym = ("coro_ft_" + kindPart + "_b" + Twine(boundPrefix)).str();
  std::string ctSym = ("coro_ct_" + kindPart + "_b" + Twine(boundPrefix)).str();

  FailureOr<FlatSymbolRefAttr> ftRef =
      ensureWamiTypeFunc(module, diagOp, ftSym, dstFuncType, moduleBuilder);
  if (failed(ftRef))
    return failure();

  return ensureWamiTypeCont(module, diagOp, ctSym, *ftRef, moduleBuilder);
}

static std::string buildImplSymbolForKind(StringRef kind) {
  return ("coro.impl." + kind).str();
}

static std::string buildResumeHelperSymbolForKind(StringRef kind) {
  return "coro.rt.resume." + sanitizeSymbolPiece(kind);
}

static Value createI32Const(OpBuilder &b, Location loc, int32_t value) {
  return wasmssa::ConstOp::create(b, loc, b.getI32IntegerAttr(value));
}

static Value createI64Const(OpBuilder &b, Location loc, int64_t value) {
  return wasmssa::ConstOp::create(b, loc, b.getI64IntegerAttr(value));
}

static Value createWasmssaEq(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  OperationState state(loc, "wasmssa.eq");
  state.addOperands({lhs, rhs});
  state.addTypes(b.getI32Type());
  Operation *op = b.create(state);
  return op->getResult(0);
}

static Value createWasmssaEqz(OpBuilder &b, Location loc, Value value) {
  OperationState state(loc, "wasmssa.eqz");
  state.addOperands(value);
  state.addTypes(b.getI32Type());
  Operation *op = b.create(state);
  return op->getResult(0);
}

static Value createWamiSelect(OpBuilder &b, Location loc, Value trueValue,
                              Value falseValue, Value condition) {
  OperationState state(loc, "wami.select");
  state.addOperands({trueValue, falseValue, condition});
  state.addTypes(trueValue.getType());
  Operation *op = b.create(state);
  return op->getResult(0);
}

static FailureOr<Value> createWasmssaRefNull(Operation *diagOp, OpBuilder &b,
                                             Type type) {
  OperationState state(diagOp->getLoc(), "wasmssa.ref_null");
  state.addTypes(type);
  Operation *op = b.create(state);
  if (!op)
    return failure();
  return op->getResult(0);
}

static FailureOr<Value> createWamiRefNull(Operation *diagOp, OpBuilder &b,
                                          Type type) {
  OperationState state(diagOp->getLoc(), "wami.ref.null");
  state.addTypes(type);
  Operation *op = b.create(state);
  if (!op)
    return failure();
  return op->getResult(0);
}

static FailureOr<Value> createWasmssaLocalGet(Operation *diagOp, OpBuilder &b,
                                              Value localRef) {
  auto localRefType = dyn_cast<wasmssa::LocalRefType>(localRef.getType());
  if (!localRefType)
    return diagOp->emitError("expected wasmssa local reference type");

  OperationState state(diagOp->getLoc(), "wasmssa.local_get");
  state.addOperands(localRef);
  state.addTypes(localRefType.getElementType());
  Operation *op = b.create(state);
  if (!op)
    return failure();
  return op->getResult(0);
}

static FailureOr<Value> createWasmssaGlobalGet(Operation *diagOp, OpBuilder &b,
                                               StringRef globalSym,
                                               Type resultType) {
  OperationState state(diagOp->getLoc(), "wasmssa.global_get");
  state.addAttribute("global",
                     FlatSymbolRefAttr::get(b.getContext(), globalSym));
  state.addTypes(resultType);
  Operation *op = b.create(state);
  if (!op)
    return failure();
  return op->getResult(0);
}

static LogicalResult createWasmssaGlobalSet(Operation *diagOp, OpBuilder &b,
                                            StringRef globalSym, Value value) {
  OperationState state(diagOp->getLoc(), "wasmssa.global_set");
  state.addAttribute("global",
                     FlatSymbolRefAttr::get(b.getContext(), globalSym));
  state.addOperands(value);
  Operation *op = b.create(state);
  if (!op)
    return diagOp->emitError("failed to create wasmssa.global_set");
  return success();
}

static FailureOr<SmallVector<Value>>
createWasmssaCall(Operation *diagOp, OpBuilder &b, StringRef callee,
                  TypeRange resultTypes, ValueRange operands) {
  OperationState state(diagOp->getLoc(), "wasmssa.call");
  state.addAttribute("callee", FlatSymbolRefAttr::get(b.getContext(), callee));
  state.addOperands(operands);
  state.addTypes(resultTypes);
  Operation *op = b.create(state);
  if (!op)
    return failure();
  return SmallVector<Value>(op->getResults().begin(), op->getResults().end());
}

static FailureOr<FlatSymbolRefAttr>
ensureI32MutableGlobal(ModuleOp module, Operation *diagOp, StringRef symName,
                       OpBuilder &moduleBuilder) {
  if (Operation *existing = SymbolTable::lookupSymbolIn(module, symName)) {
    if (existing->getName().getStringRef() != "wasmssa.global")
      return diagOp->emitError("symbol @")
             << symName << " already exists and is not wasmssa.global";
    auto typeAttr = existing->getAttrOfType<TypeAttr>("type");
    if (!typeAttr || !typeAttr.getValue().isInteger(32))
      return diagOp->emitError("symbol @")
             << symName << " has mismatched global type";
    if (!existing->hasAttr("isMutable"))
      return diagOp->emitError("symbol @") << symName << " must be mutable";
    return FlatSymbolRefAttr::get(module.getContext(), symName);
  }

  auto global = wasmssa::GlobalOp::create(
      moduleBuilder, diagOp->getLoc(), moduleBuilder.getStringAttr(symName),
      TypeAttr::get(moduleBuilder.getI32Type()), moduleBuilder.getUnitAttr(),
      /*exported=*/nullptr);

  OpBuilder::InsertionGuard guard(moduleBuilder);
  Block *initBlock = moduleBuilder.createBlock(&global.getInitializer());
  OpBuilder initBuilder = OpBuilder::atBlockBegin(initBlock);
  Value zero = createI32Const(initBuilder, diagOp->getLoc(), 0);
  wasmssa::ReturnOp::create(initBuilder, diagOp->getLoc(), zero);
  return FlatSymbolRefAttr::get(module.getContext(), symName);
}

static FailureOr<FlatSymbolRefAttr>
ensureNullableContMutableGlobal(ModuleOp module, Operation *diagOp,
                                StringRef symName, Type contNullableType,
                                OpBuilder &moduleBuilder) {
  if (Operation *existing = SymbolTable::lookupSymbolIn(module, symName)) {
    if (existing->getName().getStringRef() != "wasmssa.global")
      return diagOp->emitError("symbol @")
             << symName << " already exists and is not wasmssa.global";
    auto typeAttr = existing->getAttrOfType<TypeAttr>("type");
    if (!typeAttr || typeAttr.getValue() != contNullableType)
      return diagOp->emitError("symbol @")
             << symName << " has mismatched continuation global type";
    if (!existing->hasAttr("isMutable"))
      return diagOp->emitError("symbol @") << symName << " must be mutable";
    return FlatSymbolRefAttr::get(module.getContext(), symName);
  }

  auto global = wasmssa::GlobalOp::create(
      moduleBuilder, diagOp->getLoc(), moduleBuilder.getStringAttr(symName),
      TypeAttr::get(contNullableType), moduleBuilder.getUnitAttr(),
      /*exported=*/nullptr);

  OpBuilder::InsertionGuard guard(moduleBuilder);
  Block *initBlock = moduleBuilder.createBlock(&global.getInitializer());
  OpBuilder initBuilder = OpBuilder::atBlockBegin(initBlock);
  FailureOr<Value> initNull =
      createWasmssaRefNull(diagOp, initBuilder, contNullableType);
  if (failed(initNull))
    return failure();
  wasmssa::ReturnOp::create(initBuilder, diagOp->getLoc(), *initNull);
  return FlatSymbolRefAttr::get(module.getContext(), symName);
}

static FailureOr<FlatSymbolRefAttr>
ensureCoroResumeHelper(ModuleOp module, Operation *diagOp, StringRef kind,
                       FlatSymbolRefAttr resumeContTypeRef,
                       ArrayRef<Type> resumeArgTypes,
                       ArrayRef<Type> payloadTypes, FlatSymbolRefAttr tagSym,
                       OpBuilder &moduleBuilder) {
  Type contNullableType =
      buildContValueTypeOrEmit(diagOp, resumeContTypeRef, /*nullable=*/true);
  if (!contNullableType)
    return failure();

  SmallVector<Type> helperInputs;
  helperInputs.push_back(contNullableType);
  helperInputs.append(resumeArgTypes.begin(), resumeArgTypes.end());
  SmallVector<Type> helperResults;
  helperResults.push_back(moduleBuilder.getI32Type());
  helperResults.push_back(contNullableType);
  helperResults.append(payloadTypes.begin(), payloadTypes.end());
  auto helperType =
      FunctionType::get(module.getContext(), helperInputs, helperResults);

  std::string helperSym = buildResumeHelperSymbolForKind(kind);
  if (Operation *existing = SymbolTable::lookupSymbolIn(module, helperSym)) {
    if (existing->getName().getStringRef() != "wasmssa.func")
      return diagOp->emitError("symbol @")
             << helperSym << " already exists and is not wasmssa.func";
    auto existingFunc = cast<wasmssa::FuncOp>(existing);
    if (existingFunc.getFunctionType() != helperType)
      return diagOp->emitError("symbol @")
             << helperSym << " has mismatched helper signature";
    return FlatSymbolRefAttr::get(module.getContext(), helperSym);
  }

  auto helperFunc = wasmssa::FuncOp::create(moduleBuilder, diagOp->getLoc(),
                                            helperSym, helperType);
  Block *entry = helperFunc.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);

  SmallVector<Value> helperArgs;
  helperArgs.reserve(entry->getNumArguments());
  for (BlockArgument argRef : entry->getArguments()) {
    FailureOr<Value> argVal =
        createWasmssaLocalGet(diagOp, entryBuilder, argRef);
    if (failed(argVal))
      return failure();
    helperArgs.push_back(*argVal);
  }
  if (helperArgs.empty())
    return diagOp->emitError("internal error: helper missing continuation arg");

  Value contValue = helperArgs.front();
  ArrayRef<Value> resumeArgs(helperArgs.begin() + 1, helperArgs.end());

  Block *onSuspend = new Block();
  for (Type payloadType : payloadTypes)
    onSuspend->addArgument(payloadType, diagOp->getLoc());
  onSuspend->addArgument(contNullableType, diagOp->getLoc());
  helperFunc.getBody().push_back(onSuspend);

  OperationState blockState(diagOp->getLoc(), "wasmssa.block");
  blockState.addRegion();
  blockState.addSuccessors(onSuspend);
  Operation *blockOp = entryBuilder.create(blockState);
  Region &bodyRegion = blockOp->getRegion(0);
  Block *resumeBlock = new Block();
  bodyRegion.push_back(resumeBlock);
  OpBuilder resumeBuilder = OpBuilder::atBlockBegin(resumeBlock);

  ArrayAttr handlers = ArrayAttr::get(
      module.getContext(),
      {wami::OnLabelHandlerAttr::get(module.getContext(), tagSym, 0)});
  OperationState resumeState(diagOp->getLoc(), "wami.resume");
  resumeState.addOperands(contValue);
  resumeState.addOperands(resumeArgs);
  resumeState.addAttribute("cont_type", resumeContTypeRef);
  resumeState.addAttribute("handlers", handlers);
  resumeState.addTypes(payloadTypes);
  Operation *resumeOp = resumeBuilder.create(resumeState);

  Value doneOne = createI32Const(resumeBuilder, diagOp->getLoc(), 1);
  FailureOr<Value> nullCont =
      createWamiRefNull(diagOp, resumeBuilder, contNullableType);
  if (failed(nullCont))
    return failure();

  SmallVector<Value> completeReturn;
  completeReturn.push_back(doneOne);
  completeReturn.push_back(*nullCont);
  completeReturn.append(resumeOp->getResults().begin(),
                        resumeOp->getResults().end());
  wasmssa::ReturnOp::create(resumeBuilder, diagOp->getLoc(), completeReturn);

  OpBuilder suspendBuilder = OpBuilder::atBlockBegin(onSuspend);
  Value doneZero = createI32Const(suspendBuilder, diagOp->getLoc(), 0);
  SmallVector<Value> suspendReturn;
  suspendReturn.push_back(doneZero);
  suspendReturn.push_back(onSuspend->getArgument(payloadTypes.size()));
  for (unsigned i = 0; i < payloadTypes.size(); ++i)
    suspendReturn.push_back(onSuspend->getArgument(i));
  wasmssa::ReturnOp::create(suspendBuilder, diagOp->getLoc(), suspendReturn);

  return FlatSymbolRefAttr::get(module.getContext(), helperSym);
}

} // namespace

//===----------------------------------------------------------------------===//
// CoroVerifyIntrinsics Pass
//===----------------------------------------------------------------------===//

class CoroVerifyIntrinsics
    : public impl::CoroVerifyIntrinsicsBase<CoroVerifyIntrinsics> {
public:
  using impl::CoroVerifyIntrinsicsBase<
      CoroVerifyIntrinsics>::CoroVerifyIntrinsicsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    llvm::StringMap<func::FuncOp> implFuncs;
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (auto maybeIntrinsic = parseCoroIntrinsicName(funcOp.getSymName())) {
        if (!funcOp.isDeclaration()) {
          funcOp.emitError("coro intrinsic symbol must be declaration-only");
          signalPassFailure();
          return;
        }
      }

      if (funcOp.getSymName().starts_with("coro.impl.")) {
        StringRef kind = funcOp.getSymName().drop_front(strlen("coro.impl."));
        implFuncs[kind] = funcOp;
      }
    }

    llvm::StringMap<KindUsage> usages;
    bool failedVerify = false;

    module.walk([&](func::CallOp callOp) {
      if (failedVerify)
        return;

      FlatSymbolRefAttr callee = callOp.getCalleeAttr();
      if (!callee)
        return;

      auto intrinsic = parseCoroIntrinsicName(callee.getValue());
      if (!intrinsic)
        return;

      switch (intrinsic->kind) {
      case CoroIntrinsicKind::Spawn: {
        KindUsage &usage = usages[intrinsic->suffix];
        if (callOp.getNumResults() != 1 ||
            !callOp.getResult(0).getType().isInteger(64)) {
          callOp.emitError("coro.spawn.* must return a single i64 handle");
          failedVerify = true;
          return;
        }

        if (!usage.spawnArgTypes.empty() &&
            !typeSequencesEqual(callOp.getOperandTypes(),
                                usage.spawnArgTypes)) {
          callOp.emitError(
              "inconsistent coro.spawn.* argument types for kind '")
              << intrinsic->suffix << "'";
          failedVerify = true;
          return;
        }
        usage.spawnArgTypes.assign(callOp.getOperandTypes().begin(),
                                   callOp.getOperandTypes().end());
        usage.spawnSite = callOp;
        return;
      }
      case CoroIntrinsicKind::Resume: {
        KindUsage &usage = usages[intrinsic->suffix];
        if (callOp.getNumOperands() < 1) {
          callOp.emitError("coro.resume.* requires a leading handle operand");
          failedVerify = true;
          return;
        }
        if (!callOp.getOperand(0).getType().isInteger(64)) {
          callOp.emitError("coro.resume.* handle operand must be i64");
          failedVerify = true;
          return;
        }

        SmallVector<Type> resumeArgTypes(callOp.getOperandTypes().begin() + 1,
                                         callOp.getOperandTypes().end());
        if (callOp.getNumResults() < 2) {
          callOp.emitError(
              "coro.resume.* must return (i64, i1, payload...) tuple");
          failedVerify = true;
          return;
        }
        if (!callOp.getResult(0).getType().isInteger(64)) {
          callOp.emitError("coro.resume.* first result must be i64 handle");
          failedVerify = true;
          return;
        }
        if (!callOp.getResult(1).getType().isInteger(1)) {
          callOp.emitError("coro.resume.* second result must be i1 done flag");
          failedVerify = true;
          return;
        }
        SmallVector<Type> resumePayloadTypes(
            callOp.getResultTypes().begin() + 2, callOp.getResultTypes().end());
        if (!usage.resumeArgTypes.empty() &&
            !typeSequencesEqual(resumeArgTypes, usage.resumeArgTypes)) {
          callOp.emitError(
              "inconsistent coro.resume.* argument types for kind '")
              << intrinsic->suffix << "'";
          failedVerify = true;
          return;
        }
        if (!usage.resumePayloadTypes.empty() &&
            !typeSequencesEqual(resumePayloadTypes, usage.resumePayloadTypes)) {
          callOp.emitError("inconsistent coro.resume.* result types for kind '")
              << intrinsic->suffix << "'";
          failedVerify = true;
          return;
        }

        usage.resumeArgTypes.assign(resumeArgTypes.begin(),
                                    resumeArgTypes.end());
        usage.resumePayloadTypes.assign(resumePayloadTypes.begin(),
                                        resumePayloadTypes.end());
        usage.resumeSite = callOp;
        return;
      }
      case CoroIntrinsicKind::Yield:
        return;
      case CoroIntrinsicKind::IsDone: {
        if (callOp.getNumOperands() != 1 ||
            !callOp.getOperand(0).getType().isInteger(64) ||
            callOp.getNumResults() != 1 ||
            !callOp.getResult(0).getType().isInteger(1)) {
          callOp.emitError("coro.is_done.* must have type (i64) -> i1");
          failedVerify = true;
          return;
        }
        return;
      }
      case CoroIntrinsicKind::Cancel: {
        if (callOp.getNumOperands() != 1 ||
            !callOp.getOperand(0).getType().isInteger(64) ||
            callOp.getNumResults() != 0) {
          callOp.emitError("coro.cancel.* must have type (i64) -> ()");
          failedVerify = true;
          return;
        }
        return;
      }
      }
    });

    if (failedVerify) {
      signalPassFailure();
      return;
    }

    for (auto &entry : usages) {
      StringRef kind = entry.getKey();
      KindUsage &usage = entry.getValue();

      auto it = implFuncs.find(kind);
      if (it == implFuncs.end()) {
        module.emitError("missing coroutine implementation symbol @")
            << buildImplSymbolForKind(kind) << " for kind '" << kind << "'";
        signalPassFailure();
        return;
      }

      FunctionType implType = it->second.getFunctionType();
      if (!usage.resumePayloadTypes.empty() &&
          !typeSequencesEqual(implType.getResults(),
                              usage.resumePayloadTypes)) {
        module.emitError("coroutine implementation @")
            << it->second.getSymName()
            << " result types do not match coro.resume." << kind;
        signalPassFailure();
        return;
      }

      if (!usage.spawnArgTypes.empty() || !usage.resumeArgTypes.empty()) {
        SmallVector<Type> expectedInputs;
        expectedInputs.append(usage.spawnArgTypes.begin(),
                              usage.spawnArgTypes.end());
        expectedInputs.append(usage.resumeArgTypes.begin(),
                              usage.resumeArgTypes.end());
        if (!typeSequencesEqual(implType.getInputs(), expectedInputs)) {
          module.emitError("coroutine implementation @")
              << it->second.getSymName()
              << " input types must equal spawn args + resume args for kind '"
              << kind << "'";
          signalPassFailure();
          return;
        }
      }
    }

    for (auto &entry : implFuncs) {
      StringRef kind = entry.getKey();
      func::FuncOp implFunc = entry.getValue();

      auto usageIt = usages.find(kind);
      if (usageIt == usages.end())
        continue;
      KindUsage &usage = usageIt->second;

      bool localFailure = false;
      implFunc.walk([&](func::CallOp callOp) {
        if (localFailure)
          return;
        FlatSymbolRefAttr callee = callOp.getCalleeAttr();
        if (!callee)
          return;
        auto intrinsic = parseCoroIntrinsicName(callee.getValue());
        if (!intrinsic || intrinsic->kind != CoroIntrinsicKind::Yield)
          return;

        if (intrinsic->suffix != kind) {
          callOp.emitError("coro.yield.* inside @")
              << implFunc.getSymName() << " must use suffix '" << kind << "'";
          localFailure = true;
          return;
        }

        if (!typeSequencesEqual(callOp.getOperandTypes(),
                                usage.resumePayloadTypes)) {
          callOp.emitError("coro.yield.")
              << kind
              << " operand types must match coro.resume payload types for kind "
                 "'"
              << kind << "'";
          localFailure = true;
          return;
        }
        if (!typeSequencesEqual(callOp.getResultTypes(),
                                usage.resumeArgTypes)) {
          callOp.emitError("coro.yield.")
              << kind
              << " result types must match coro.resume argument types for kind "
                 "'"
              << kind << "'";
          localFailure = true;
          return;
        }
      });

      if (localFailure) {
        signalPassFailure();
        return;
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// CoroNormalize Pass
//===----------------------------------------------------------------------===//

class CoroNormalize : public impl::CoroNormalizeBase<CoroNormalize> {
public:
  using impl::CoroNormalizeBase<CoroNormalize>::CoroNormalizeBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (!isCoroIntrinsicSymbol(funcOp.getSymName()))
        continue;

      if (!funcOp.isDeclaration()) {
        funcOp.emitError("coro intrinsic symbols must be declarations");
        signalPassFailure();
        return;
      }

      StringAttr visibility = funcOp->getAttrOfType<StringAttr>(
          SymbolTable::getVisibilityAttrName());
      if (!visibility || visibility.getValue() != "private") {
        funcOp->setAttr(SymbolTable::getVisibilityAttrName(),
                        StringAttr::get(module.getContext(), "private"));
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// CoroToWAMI Pass
//===----------------------------------------------------------------------===//

class CoroToWAMI : public impl::CoroToWAMIBase<CoroToWAMI> {
public:
  using impl::CoroToWAMIBase<CoroToWAMI>::CoroToWAMIBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    llvm::StringMap<wasmssa::FuncOp> implFuncs;
    for (wasmssa::FuncOp funcOp : module.getOps<wasmssa::FuncOp>()) {
      if (!funcOp.getSymName().starts_with("coro.impl."))
        continue;
      StringRef kind = funcOp.getSymName().drop_front(strlen("coro.impl."));
      implFuncs[kind] = funcOp;
    }

    if (implFuncs.empty())
      return;

    OpBuilder moduleBuilder(ctx);
    moduleBuilder.setInsertionPointToStart(module.getBody());

    bool passFailed = false;
    int64_t nextHandleId = 1;
    llvm::StringMap<SmallVector<RuntimeSlot, 1>> slotsByKind;
    llvm::DenseMap<Operation *, int64_t> spawnHandleByOp;
    llvm::StringMap<KindRuntimeInfo> kindInfo;
    SmallVector<wasmssa::FuncCallOp> calls;
    SmallVector<Operation *> eraseLater;

    module.walk([&](wasmssa::FuncCallOp callOp) { calls.push_back(callOp); });

    // First phase: discover spawn callsites module-wide and allocate runtime
    // slots backed by mutable globals.
    for (wasmssa::FuncCallOp callOp : calls) {
      if (!callOp || !callOp->getBlock())
        continue;

      FlatSymbolRefAttr callee = callOp.getCalleeAttr();
      if (!callee)
        continue;

      auto intrinsic = parseCoroIntrinsicName(callee.getValue());
      if (!intrinsic || intrinsic->kind != CoroIntrinsicKind::Spawn)
        continue;

      if (callOp.getNumResults() != 1 ||
          !callOp.getResult(0).getType().isInteger(64)) {
        callOp.emitError("coro.spawn.* must return a single i64 handle");
        passFailed = true;
        break;
      }

      auto implIt = implFuncs.find(intrinsic->suffix);
      if (implIt == implFuncs.end()) {
        callOp.emitError("missing implementation symbol @")
            << buildImplSymbolForKind(intrinsic->suffix);
        passFailed = true;
        break;
      }

      wasmssa::FuncOp implFunc = implIt->second;
      FunctionType implType = implFunc.getFunctionType();
      unsigned boundCount = callOp.getNumOperands();
      if (boundCount > implType.getNumInputs()) {
        callOp.emitError(
            "spawn argument count exceeds implementation input arity");
        passFailed = true;
        break;
      }

      FailureOr<FlatSymbolRefAttr> srcContRef = ensureContTypeForBoundPrefix(
          module, callOp, intrinsic->suffix, implType, 0, moduleBuilder);
      if (failed(srcContRef)) {
        passFailed = true;
        break;
      }

      FlatSymbolRefAttr resumeContRef = *srcContRef;
      if (boundCount > 0) {
        FailureOr<FlatSymbolRefAttr> dstContRef =
            ensureContTypeForBoundPrefix(module, callOp, intrinsic->suffix,
                                         implType, boundCount, moduleBuilder);
        if (failed(dstContRef)) {
          passFailed = true;
          break;
        }
        resumeContRef = *dstContRef;
      }

      RuntimeSlot slot;
      slot.handleId = nextHandleId++;
      slot.srcContType = *srcContRef;
      slot.resumeContType = resumeContRef;
      slot.kind = intrinsic->suffix;
      slot.implSym = buildImplSymbolForKind(intrinsic->suffix);
      slot.expectedResumeArgs.append(implType.getInputs().begin() + boundCount,
                                     implType.getInputs().end());
      slot.expectedResumePayloads.append(implType.getResults().begin(),
                                         implType.getResults().end());

      Type contNullableType =
          buildContValueTypeOrEmit(callOp, resumeContRef, /*nullable=*/true);
      if (!contNullableType) {
        passFailed = true;
        break;
      }

      std::string slotStem =
          ("coro_slot_" + sanitizeSymbolPiece(intrinsic->suffix) + "_" +
           Twine(slot.handleId))
              .str();
      FailureOr<FlatSymbolRefAttr> stateGlobal = ensureI32MutableGlobal(
          module, callOp, slotStem + "_state", moduleBuilder);
      if (failed(stateGlobal)) {
        passFailed = true;
        break;
      }
      FailureOr<FlatSymbolRefAttr> contGlobal = ensureNullableContMutableGlobal(
          module, callOp, slotStem + "_cont", contNullableType, moduleBuilder);
      if (failed(contGlobal)) {
        passFailed = true;
        break;
      }
      slot.stateGlobal = *stateGlobal;
      slot.contGlobal = *contGlobal;

      spawnHandleByOp[callOp.getOperation()] = slot.handleId;
      slotsByKind[intrinsic->suffix].push_back(slot);
    }

    if (passFailed) {
      signalPassFailure();
      return;
    }

    // Second phase: materialize per-kind runtime symbols (yield tag + helper).
    for (auto &entry : slotsByKind) {
      StringRef kind = entry.getKey();
      SmallVector<RuntimeSlot, 1> &slots = entry.getValue();
      if (slots.empty())
        continue;

      RuntimeSlot &canonical = slots.front();
      for (RuntimeSlot &slot : slots) {
        if (!typeSequencesEqual(slot.expectedResumeArgs,
                                canonical.expectedResumeArgs) ||
            !typeSequencesEqual(slot.expectedResumePayloads,
                                canonical.expectedResumePayloads) ||
            slot.resumeContType != canonical.resumeContType) {
          module.emitError("inconsistent runtime slot signatures for kind '")
              << kind << "'";
          signalPassFailure();
          return;
        }
      }

      KindRuntimeInfo info;
      info.resumeContType = canonical.resumeContType;
      info.resumeArgTypes.assign(canonical.expectedResumeArgs.begin(),
                                 canonical.expectedResumeArgs.end());
      info.payloadTypes.assign(canonical.expectedResumePayloads.begin(),
                               canonical.expectedResumePayloads.end());

      auto tagType =
          FunctionType::get(ctx, info.payloadTypes, info.resumeArgTypes);
      std::string tagSym = "coro_tag_" + sanitizeSymbolPiece(kind);
      FailureOr<FlatSymbolRefAttr> tagRef =
          ensureWamiTag(module, module, tagSym, tagType, moduleBuilder);
      if (failed(tagRef)) {
        signalPassFailure();
        return;
      }
      info.tagSym = *tagRef;

      FailureOr<FlatSymbolRefAttr> helperRef = ensureCoroResumeHelper(
          module, module, kind, info.resumeContType, info.resumeArgTypes,
          info.payloadTypes, info.tagSym, moduleBuilder);
      if (failed(helperRef)) {
        signalPassFailure();
        return;
      }
      info.helperSym = *helperRef;
      kindInfo[kind] = info;
    }

    auto findSlotByHandle = [&](StringRef kind,
                                int64_t handleId) -> RuntimeSlot * {
      auto it = slotsByKind.find(kind);
      if (it == slotsByKind.end())
        return nullptr;
      for (RuntimeSlot &slot : it->second) {
        if (slot.handleId == handleId)
          return &slot;
      }
      return nullptr;
    };

    // Third phase: rewrite intrinsic callsites.
    for (wasmssa::FuncCallOp callOp : calls) {
      if (!callOp || !callOp->getBlock())
        continue;

      FlatSymbolRefAttr callee = callOp.getCalleeAttr();
      if (!callee)
        continue;

      auto intrinsic = parseCoroIntrinsicName(callee.getValue());
      if (!intrinsic)
        continue;

      OpBuilder b(callOp);
      switch (intrinsic->kind) {
      case CoroIntrinsicKind::Spawn: {
        auto handleIt = spawnHandleByOp.find(callOp.getOperation());
        if (handleIt == spawnHandleByOp.end()) {
          callOp.emitError("internal error: missing runtime slot for spawn");
          passFailed = true;
          break;
        }

        RuntimeSlot *slot =
            findSlotByHandle(intrinsic->suffix, handleIt->second);
        if (!slot) {
          callOp.emitError("internal error: failed to resolve runtime slot");
          passFailed = true;
          break;
        }

        Type funcrefType = parseWamiTypeOrEmit(
            callOp, ("!wami.funcref<@" + slot->implSym + ">"));
        Type srcContType = buildContValueTypeOrEmit(callOp, slot->srcContType,
                                                    /*nullable=*/false);
        Type dstContType = buildContValueTypeOrEmit(
            callOp, slot->resumeContType, /*nullable=*/false);
        if (!funcrefType || !srcContType || !dstContType) {
          passFailed = true;
          break;
        }

        OperationState refFuncState(callOp.getLoc(), "wami.ref.func");
        refFuncState.addAttribute(
            "func", FlatSymbolRefAttr::get(b.getContext(), slot->implSym));
        refFuncState.addTypes(funcrefType);
        Operation *refFuncOp = b.create(refFuncState);

        OperationState contNewState(callOp.getLoc(), "wami.cont.new");
        contNewState.addOperands(refFuncOp->getResult(0));
        contNewState.addAttribute("cont_type", slot->srcContType);
        contNewState.addTypes(srcContType);
        Operation *contValueOp = b.create(contNewState);
        Value contValue = contValueOp->getResult(0);

        if (callOp.getNumOperands() > 0) {
          SmallVector<Value> bindOperands;
          bindOperands.push_back(contValue);
          bindOperands.append(callOp.getOperands().begin(),
                              callOp.getOperands().end());
          OperationState bindState(callOp.getLoc(), "wami.cont.bind");
          bindState.addOperands(bindOperands);
          bindState.addAttribute("src_cont_type", slot->srcContType);
          bindState.addAttribute("dst_cont_type", slot->resumeContType);
          bindState.addTypes(dstContType);
          Operation *bindOp = b.create(bindState);
          contValue = bindOp->getResult(0);
        }

        Value ready = createI32Const(b, callOp.getLoc(), 1);
        if (failed(createWasmssaGlobalSet(
                callOp, b, slot->contGlobal.getValue(), contValue)) ||
            failed(createWasmssaGlobalSet(
                callOp, b, slot->stateGlobal.getValue(), ready))) {
          passFailed = true;
          break;
        }

        Value handleConst = createI64Const(b, callOp.getLoc(), slot->handleId);
        callOp.getResult(0).replaceAllUsesWith(handleConst);
        eraseLater.push_back(callOp);
        continue;
      }
      case CoroIntrinsicKind::Resume: {
        if (callOp.getNumOperands() < 1) {
          callOp.emitError("coro.resume.* requires a handle operand");
          passFailed = true;
          break;
        }
        if (!callOp.getOperand(0).getType().isInteger(64)) {
          callOp.emitError("coro.resume.* handle operand must be i64");
          passFailed = true;
          break;
        }
        if (callOp.getNumResults() < 2 ||
            !callOp.getResult(0).getType().isInteger(64) ||
            !callOp.getResult(1).getType().isInteger(32)) {
          callOp.emitError("coro.resume.* lowered signature must be "
                           "(i64, i32, payload...)");
          passFailed = true;
          break;
        }

        auto slotIt = slotsByKind.find(intrinsic->suffix);
        auto infoIt = kindInfo.find(intrinsic->suffix);
        if (slotIt == slotsByKind.end() || slotIt->second.empty() ||
            infoIt == kindInfo.end()) {
          callOp.emitError("missing runtime state for kind '")
              << intrinsic->suffix << "'";
          passFailed = true;
          break;
        }
        SmallVector<RuntimeSlot, 1> &slots = slotIt->second;
        KindRuntimeInfo &info = infoIt->second;

        SmallVector<Type> resumeArgTypes(callOp.getOperandTypes().begin() + 1,
                                         callOp.getOperandTypes().end());
        if (!typeSequencesEqual(resumeArgTypes, info.resumeArgTypes)) {
          callOp.emitError(
              "resume argument types do not match continuation type");
          passFailed = true;
          break;
        }

        SmallVector<Type> payloadTypes(callOp.getResultTypes().begin() + 2,
                                       callOp.getResultTypes().end());
        if (!typeSequencesEqual(payloadTypes, info.payloadTypes)) {
          callOp.emitError(
              "resume payload result types do not match implementation");
          passFailed = true;
          break;
        }

        Type contNullableType = buildContValueTypeOrEmit(
            callOp, info.resumeContType, /*nullable=*/true);
        if (!contNullableType) {
          passFailed = true;
          break;
        }

        // Fast path for the common single-slot case. Avoid reference-typed
        // select materialization so runtimes without ref-select support can
        // execute lowered coroutines.
        if (slots.size() == 1) {
          RuntimeSlot &slot = slots.front();
          FailureOr<Value> oldCont = createWasmssaGlobalGet(
              callOp, b, slot.contGlobal.getValue(), contNullableType);
          FailureOr<Value> oldState = createWasmssaGlobalGet(
              callOp, b, slot.stateGlobal.getValue(), b.getI32Type());
          if (failed(oldCont) || failed(oldState)) {
            passFailed = true;
            break;
          }

          SmallVector<Value> helperOperands;
          helperOperands.push_back(*oldCont);
          helperOperands.append(callOp.getOperands().begin() + 1,
                                callOp.getOperands().end());

          SmallVector<Type> helperResultTypes;
          helperResultTypes.push_back(b.getI32Type());
          helperResultTypes.push_back(contNullableType);
          helperResultTypes.append(info.payloadTypes.begin(),
                                   info.payloadTypes.end());
          FailureOr<SmallVector<Value>> helperResults =
              createWasmssaCall(callOp, b, info.helperSym.getValue(),
                                helperResultTypes, helperOperands);
          if (failed(helperResults)) {
            passFailed = true;
            break;
          }

          Value doneI32 = (*helperResults)[0];
          Value nextCont = (*helperResults)[1];
          Value ready = createI32Const(b, callOp.getLoc(), 1);
          Value doneState = createI32Const(b, callOp.getLoc(), 2);
          Value resumedState =
              createWamiSelect(b, callOp.getLoc(), doneState, ready, doneI32);

          if (failed(createWasmssaGlobalSet(
                  callOp, b, slot.stateGlobal.getValue(), resumedState)) ||
              failed(createWasmssaGlobalSet(
                  callOp, b, slot.contGlobal.getValue(), nextCont))) {
            passFailed = true;
            break;
          }

          callOp.getResult(0).replaceAllUsesWith(callOp.getOperand(0));
          callOp.getResult(1).replaceAllUsesWith(doneI32);
          for (auto [idx, payload] :
               llvm::enumerate(ArrayRef<Value>(*helperResults).drop_front(2))) {
            callOp.getResult(idx + 2).replaceAllUsesWith(payload);
          }
          eraseLater.push_back(callOp);
          continue;
        }

        FailureOr<Value> nullCont =
            createWamiRefNull(callOp, b, contNullableType);
        if (failed(nullCont)) {
          passFailed = true;
          break;
        }

        SmallVector<Value> matches;
        matches.reserve(slots.size());
        for (const RuntimeSlot &slot : slots) {
          Value handleConst = createI64Const(b, callOp.getLoc(), slot.handleId);
          matches.push_back(createWasmssaEq(b, callOp.getLoc(),
                                            callOp.getOperand(0), handleConst));
        }

        Value selectedCont = *nullCont;
        Value selectedState = createI32Const(b, callOp.getLoc(), 0);
        for (unsigned i = 0; i < slots.size(); ++i) {
          FailureOr<Value> candidateCont = createWasmssaGlobalGet(
              callOp, b, slots[i].contGlobal.getValue(), contNullableType);
          FailureOr<Value> candidateState = createWasmssaGlobalGet(
              callOp, b, slots[i].stateGlobal.getValue(), b.getI32Type());
          if (failed(candidateCont) || failed(candidateState)) {
            passFailed = true;
            break;
          }
          selectedCont = createWamiSelect(b, callOp.getLoc(), *candidateCont,
                                          selectedCont, matches[i]);
          selectedState = createWamiSelect(b, callOp.getLoc(), *candidateState,
                                           selectedState, matches[i]);
        }
        if (passFailed)
          break;

        Value ready = createI32Const(b, callOp.getLoc(), 1);
        Value isReady =
            createWasmssaEq(b, callOp.getLoc(), selectedState, ready);
        Value safeCont = createWamiSelect(b, callOp.getLoc(), selectedCont,
                                          *nullCont, isReady);

        SmallVector<Value> helperOperands;
        helperOperands.push_back(safeCont);
        helperOperands.append(callOp.getOperands().begin() + 1,
                              callOp.getOperands().end());

        SmallVector<Type> helperResultTypes;
        helperResultTypes.push_back(b.getI32Type());
        helperResultTypes.push_back(contNullableType);
        helperResultTypes.append(info.payloadTypes.begin(),
                                 info.payloadTypes.end());
        FailureOr<SmallVector<Value>> helperResults =
            createWasmssaCall(callOp, b, info.helperSym.getValue(),
                              helperResultTypes, helperOperands);
        if (failed(helperResults)) {
          passFailed = true;
          break;
        }
        Value doneI32 = (*helperResults)[0];
        Value nextCont = (*helperResults)[1];

        Value doneState = createI32Const(b, callOp.getLoc(), 2);
        Value resumedState =
            createWamiSelect(b, callOp.getLoc(), doneState, ready, doneI32);

        for (unsigned i = 0; i < slots.size(); ++i) {
          FailureOr<Value> oldState = createWasmssaGlobalGet(
              callOp, b, slots[i].stateGlobal.getValue(), b.getI32Type());
          FailureOr<Value> oldCont = createWasmssaGlobalGet(
              callOp, b, slots[i].contGlobal.getValue(), contNullableType);
          if (failed(oldState) || failed(oldCont)) {
            passFailed = true;
            break;
          }

          Value stateWhenReady = createWamiSelect(
              b, callOp.getLoc(), resumedState, *oldState, isReady);
          Value contWhenReady =
              createWamiSelect(b, callOp.getLoc(), nextCont, *oldCont, isReady);
          Value finalState = createWamiSelect(
              b, callOp.getLoc(), stateWhenReady, *oldState, matches[i]);
          Value finalCont = createWamiSelect(b, callOp.getLoc(), contWhenReady,
                                             *oldCont, matches[i]);

          if (failed(createWasmssaGlobalSet(
                  callOp, b, slots[i].stateGlobal.getValue(), finalState)) ||
              failed(createWasmssaGlobalSet(
                  callOp, b, slots[i].contGlobal.getValue(), finalCont))) {
            passFailed = true;
            break;
          }
        }
        if (passFailed)
          break;

        callOp.getResult(0).replaceAllUsesWith(callOp.getOperand(0));
        callOp.getResult(1).replaceAllUsesWith(doneI32);
        for (auto [idx, payload] :
             llvm::enumerate(ArrayRef<Value>(*helperResults).drop_front(2))) {
          callOp.getResult(idx + 2).replaceAllUsesWith(payload);
        }
        eraseLater.push_back(callOp);
        continue;
      }
      case CoroIntrinsicKind::Yield: {
        auto infoIt = kindInfo.find(intrinsic->suffix);
        if (infoIt == kindInfo.end()) {
          callOp.emitError("missing runtime kind info for coro.yield.")
              << intrinsic->suffix;
          passFailed = true;
          break;
        }
        KindRuntimeInfo &info = infoIt->second;

        if (!typeSequencesEqual(callOp.getOperandTypes(), info.payloadTypes)) {
          callOp.emitError(
              "yield payload types must match resume payload types");
          passFailed = true;
          break;
        }
        if (!typeSequencesEqual(callOp.getResultTypes(), info.resumeArgTypes)) {
          callOp.emitError(
              "yield result types must match resume argument types");
          passFailed = true;
          break;
        }

        OperationState suspendState(callOp.getLoc(), "wami.suspend");
        suspendState.addOperands(callOp.getOperands());
        suspendState.addAttribute("tag", info.tagSym);
        suspendState.addTypes(callOp.getResultTypes());
        Operation *suspendOp = b.create(suspendState);
        callOp.replaceAllUsesWith(suspendOp->getResults());
        eraseLater.push_back(callOp);
        continue;
      }
      case CoroIntrinsicKind::Cancel: {
        if (callOp.getNumOperands() != 1 ||
            !callOp.getOperand(0).getType().isInteger(64) ||
            callOp.getNumResults() != 0) {
          callOp.emitError("coro.cancel.* must have type (i64) -> ()");
          passFailed = true;
          break;
        }

        auto slotIt = slotsByKind.find(intrinsic->suffix);
        auto infoIt = kindInfo.find(intrinsic->suffix);
        if (slotIt == slotsByKind.end() || slotIt->second.empty() ||
            infoIt == kindInfo.end()) {
          callOp.emitError("missing runtime state for kind '")
              << intrinsic->suffix << "'";
          passFailed = true;
          break;
        }
        SmallVector<RuntimeSlot, 1> &slots = slotIt->second;
        KindRuntimeInfo &info = infoIt->second;

        Type contNullableType = buildContValueTypeOrEmit(
            callOp, info.resumeContType, /*nullable=*/true);
        if (!contNullableType) {
          passFailed = true;
          break;
        }
        FailureOr<Value> nullCont =
            createWamiRefNull(callOp, b, contNullableType);
        if (failed(nullCont)) {
          passFailed = true;
          break;
        }

        Value canceled = createI32Const(b, callOp.getLoc(), 3);
        for (RuntimeSlot &slot : slots) {
          Value handleConst = createI64Const(b, callOp.getLoc(), slot.handleId);
          Value match = createWasmssaEq(b, callOp.getLoc(),
                                        callOp.getOperand(0), handleConst);

          FailureOr<Value> oldState = createWasmssaGlobalGet(
              callOp, b, slot.stateGlobal.getValue(), b.getI32Type());
          FailureOr<Value> oldCont = createWasmssaGlobalGet(
              callOp, b, slot.contGlobal.getValue(), contNullableType);
          if (failed(oldState) || failed(oldCont)) {
            passFailed = true;
            break;
          }

          Value newState =
              createWamiSelect(b, callOp.getLoc(), canceled, *oldState, match);
          Value newCont =
              createWamiSelect(b, callOp.getLoc(), *nullCont, *oldCont, match);
          if (failed(createWasmssaGlobalSet(
                  callOp, b, slot.stateGlobal.getValue(), newState)) ||
              failed(createWasmssaGlobalSet(
                  callOp, b, slot.contGlobal.getValue(), newCont))) {
            passFailed = true;
            break;
          }
        }
        if (passFailed)
          break;
        eraseLater.push_back(callOp);
        continue;
      }
      case CoroIntrinsicKind::IsDone: {
        if (callOp.getNumOperands() != 1 ||
            !callOp.getOperand(0).getType().isInteger(64) ||
            callOp.getNumResults() != 1 ||
            !callOp.getResult(0).getType().isInteger(32)) {
          callOp.emitError(
              "coro.is_done.* lowered signature must be (i64) -> i32");
          passFailed = true;
          break;
        }

        auto slotIt = slotsByKind.find(intrinsic->suffix);
        if (slotIt == slotsByKind.end() || slotIt->second.empty()) {
          callOp.emitError("missing runtime state for kind '")
              << intrinsic->suffix << "'";
          passFailed = true;
          break;
        }
        SmallVector<RuntimeSlot, 1> &slots = slotIt->second;

        Value selectedState = createI32Const(b, callOp.getLoc(), 0);
        for (RuntimeSlot &slot : slots) {
          Value handleConst = createI64Const(b, callOp.getLoc(), slot.handleId);
          Value match = createWasmssaEq(b, callOp.getLoc(),
                                        callOp.getOperand(0), handleConst);
          FailureOr<Value> state = createWasmssaGlobalGet(
              callOp, b, slot.stateGlobal.getValue(), b.getI32Type());
          if (failed(state)) {
            passFailed = true;
            break;
          }
          selectedState = createWamiSelect(b, callOp.getLoc(), *state,
                                           selectedState, match);
        }
        if (passFailed)
          break;

        Value ready = createI32Const(b, callOp.getLoc(), 1);
        Value isReady =
            createWasmssaEq(b, callOp.getLoc(), selectedState, ready);
        Value done = createWasmssaEqz(b, callOp.getLoc(), isReady);
        callOp.getResult(0).replaceAllUsesWith(done);
        eraseLater.push_back(callOp);
        continue;
      }
      }
    }

    if (passFailed) {
      signalPassFailure();
      return;
    }

    for (Operation *op : eraseLater) {
      if (op && op->getBlock())
        op->erase();
    }

    llvm::StringSet<> usedCallees;
    module.walk([&](wasmssa::FuncCallOp callOp) {
      if (FlatSymbolRefAttr callee = callOp.getCalleeAttr())
        usedCallees.insert(callee.getValue());
    });

    SmallVector<wasmssa::FuncImportOp> importsToErase;
    for (wasmssa::FuncImportOp importOp :
         module.getOps<wasmssa::FuncImportOp>()) {
      if (!importOp.getSymName().starts_with("coro."))
        continue;
      if (!usedCallees.contains(importOp.getSymName()))
        importsToErase.push_back(importOp);
    }
    for (wasmssa::FuncImportOp importOp : importsToErase)
      importOp.erase();
  }
};

//===----------------------------------------------------------------------===//
// CoroToLLVM Pass
//===----------------------------------------------------------------------===//

class CoroToLLVM : public impl::CoroToLLVMBase<CoroToLLVM> {
public:
  using impl::CoroToLLVMBase<CoroToLLVM>::CoroToLLVMBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    llvm::StringMap<int64_t> kindHandleIds;
    int64_t nextHandleId = 1;
    bool passFailed = false;

    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      SmallVector<func::CallOp> calls;
      funcOp.walk([&](func::CallOp callOp) { calls.push_back(callOp); });

      for (func::CallOp callOp : calls) {
        if (!callOp || !callOp->getBlock())
          continue;

        FlatSymbolRefAttr callee = callOp.getCalleeAttr();
        if (!callee)
          continue;

        auto intrinsic = parseCoroIntrinsicName(callee.getValue());
        if (!intrinsic)
          continue;

        OpBuilder b(callOp);
        switch (intrinsic->kind) {
        case CoroIntrinsicKind::Spawn: {
          if (callOp.getNumResults() != 1 ||
              !callOp.getResult(0).getType().isInteger(64)) {
            callOp.emitError("coro.spawn.* must return a single i64 handle");
            passFailed = true;
            break;
          }

          int64_t handleId = 0;
          auto it = kindHandleIds.find(intrinsic->suffix);
          if (it == kindHandleIds.end()) {
            handleId = nextHandleId++;
            kindHandleIds[intrinsic->suffix] = handleId;
          } else {
            handleId = it->second;
          }
          auto cst = arith::ConstantIntOp::create(b, callOp.getLoc(), handleId,
                                                  /*width=*/64);
          callOp.getResult(0).replaceAllUsesWith(cst.getResult());
          callOp.erase();
          continue;
        }
        case CoroIntrinsicKind::Resume: {
          if (callOp.getNumOperands() < 1) {
            callOp.emitError("coro.resume.* requires a handle operand");
            passFailed = true;
            break;
          }
          if (callOp.getNumResults() < 2 ||
              !callOp.getResult(0).getType().isInteger(64) ||
              !callOp.getResult(1).getType().isInteger(1)) {
            callOp.emitError("coro.resume.* must return (i64, i1, payload...)");
            passFailed = true;
            break;
          }
          SmallVector<Value> args(callOp.getOperands().begin() + 1,
                                  callOp.getOperands().end());
          SmallVector<Type> payloadTypes(callOp.getResultTypes().begin() + 2,
                                         callOp.getResultTypes().end());
          std::string implSym = buildImplSymbolForKind(intrinsic->suffix);
          auto direct = func::CallOp::create(b, callOp.getLoc(), implSym,
                                             payloadTypes, args);
          auto done = arith::ConstantIntOp::create(b, callOp.getLoc(), 1,
                                                   /*width=*/1);
          callOp.getResult(0).replaceAllUsesWith(callOp.getOperand(0));
          callOp.getResult(1).replaceAllUsesWith(done.getResult());
          for (auto [idx, payload] : llvm::enumerate(direct.getResults()))
            callOp.getResult(idx + 2).replaceAllUsesWith(payload);
          callOp.erase();
          continue;
        }
        case CoroIntrinsicKind::Yield: {
          if (callOp.getNumResults() == 0) {
            callOp.erase();
            continue;
          }
          if (callOp.getNumResults() == 1 && callOp.getNumOperands() == 1 &&
              callOp.getResult(0).getType() == callOp.getOperand(0).getType()) {
            callOp.getResult(0).replaceAllUsesWith(callOp.getOperand(0));
            callOp.erase();
            continue;
          }
          callOp.emitError(
              "coro.yield.* direct lowering currently supports only "
              "(T)->T or ()");
          passFailed = true;
          break;
        }
        case CoroIntrinsicKind::IsDone: {
          if (callOp.getNumResults() != 1 ||
              !callOp.getResult(0).getType().isInteger(1)) {
            callOp.emitError("coro.is_done.* must return i1");
            passFailed = true;
            break;
          }
          auto one = arith::ConstantIntOp::create(b, callOp.getLoc(), 1,
                                                  /*width=*/1);
          callOp.getResult(0).replaceAllUsesWith(one.getResult());
          callOp.erase();
          continue;
        }
        case CoroIntrinsicKind::Cancel:
          callOp.erase();
          continue;
        }
      }
      if (passFailed)
        break;
    }

    if (passFailed) {
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> eraseDecls;
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (!isCoroIntrinsicSymbol(funcOp.getSymName()))
        continue;
      if (funcOp.use_empty())
        eraseDecls.push_back(funcOp);
    }
    for (func::FuncOp funcOp : eraseDecls)
      funcOp.erase();
  }
};

} // namespace mlir::wami
