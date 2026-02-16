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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Hashing.h"
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

struct CoroLLVMRuntimeDecls {
  LLVM::LLVMFuncOp mallocFn;
  LLVM::LLVMFuncOp freeFn;
  LLVM::LLVMFuncOp trapFn;
  LLVM::LLVMFuncOp coroIdFn;
  LLVM::LLVMFuncOp coroSizeFn;
  LLVM::LLVMFuncOp coroBeginFn;
  LLVM::LLVMFuncOp coroSaveFn;
  LLVM::LLVMFuncOp coroSuspendFn;
  LLVM::LLVMFuncOp coroEndFn;
  LLVM::LLVMFuncOp coroFreeFn;
  LLVM::LLVMFuncOp coroResumeFn;
  LLVM::LLVMFuncOp coroDestroyFn;
};

struct CoroLLVMKindInfo {
  std::string kind;
  LLVM::LLVMFuncOp spawnDecl;
  LLVM::LLVMFuncOp resumeDecl;
  LLVM::LLVMFuncOp isDoneDecl;
  LLVM::LLVMFuncOp cancelDecl;
  LLVM::LLVMFuncOp yieldDecl;
  LLVM::LLVMFuncOp implFunc;

  SmallVector<Type> spawnArgTypes;
  SmallVector<Type> resumeArgTypes;
  SmallVector<Type> payloadTypes;
  Type implPayloadPackedType;
  Type yieldResumePackedType;
  LLVM::LLVMStructType runtimeType;

  unsigned magicField = 0;
  unsigned doneField = 1;
  unsigned frameField = 2;
  unsigned spawnBase = 3;
  unsigned resumeBase = 3;
  unsigned payloadBase = 3;
  uint64_t magic = 0;
  bool hasYield = false;

  std::string spawnHelperSym;
  std::string resumeHelperSym;
  std::string isDoneHelperSym;
  std::string cancelHelperSym;
};

static Value createLLVMConstI1(OpBuilder &b, Location loc, bool value) {
  return b.create<LLVM::ConstantOp>(loc, b.getI1Type(), b.getBoolAttr(value));
}

static Value createLLVMConstI32(OpBuilder &b, Location loc, int32_t value) {
  return b.create<LLVM::ConstantOp>(loc, b.getI32Type(),
                                    b.getI32IntegerAttr(value));
}

static Value createLLVMConstU64(OpBuilder &b, Location loc, uint64_t value) {
  return b.create<LLVM::ConstantOp>(
      loc, b.getI64Type(),
      b.getIntegerAttr(b.getI64Type(), APInt(/*numBits=*/64, value)));
}

static Value createLLVMNullPtr(OpBuilder &b, Location loc) {
  return b.create<LLVM::ZeroOp>(loc,
                                LLVM::LLVMPointerType::get(b.getContext()));
}

static Value createRuntimeFieldPtr(OpBuilder &b, Location loc, Value rtPtr,
                                   LLVM::LLVMStructType runtimeTy,
                                   unsigned fieldIndex) {
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  return b.create<LLVM::GEPOp>(
      loc, ptrTy, runtimeTy, rtPtr,
      ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});
}

static Value loadRuntimeField(OpBuilder &b, Location loc, Value rtPtr,
                              LLVM::LLVMStructType runtimeTy, Type fieldTy,
                              unsigned fieldIndex) {
  Value fieldPtr = createRuntimeFieldPtr(b, loc, rtPtr, runtimeTy, fieldIndex);
  return b.create<LLVM::LoadOp>(loc, fieldTy, fieldPtr);
}

static void storeRuntimeField(OpBuilder &b, Location loc, Value rtPtr,
                              LLVM::LLVMStructType runtimeTy, unsigned fieldIdx,
                              Value value) {
  Value fieldPtr = createRuntimeFieldPtr(b, loc, rtPtr, runtimeTy, fieldIdx);
  b.create<LLVM::StoreOp>(loc, value, fieldPtr);
}

static Type getPackedLLVMTypeForValues(MLIRContext *ctx,
                                       ArrayRef<Type> values) {
  if (values.empty())
    return LLVM::LLVMVoidType::get(ctx);
  if (values.size() == 1)
    return values.front();
  return LLVM::LLVMStructType::getLiteral(ctx, values);
}

static Value packLLVMValues(OpBuilder &b, Location loc, Type packedType,
                            ArrayRef<Value> values) {
  if (values.empty())
    return Value();
  if (values.size() == 1)
    return values.front();

  Value packed = b.create<LLVM::UndefOp>(loc, packedType);
  for (auto [idx, v] : llvm::enumerate(values))
    packed = b.create<LLVM::InsertValueOp>(loc, packed, v, idx);
  return packed;
}

static SmallVector<Value> unpackLLVMValues(OpBuilder &b, Location loc,
                                           Value packed,
                                           ArrayRef<Type> unpackedTypes) {
  SmallVector<Value> out;
  if (unpackedTypes.empty())
    return out;
  if (unpackedTypes.size() == 1) {
    out.push_back(packed);
    return out;
  }
  out.reserve(unpackedTypes.size());
  for (auto [idx, ty] : llvm::enumerate(unpackedTypes))
    out.push_back(b.create<LLVM::ExtractValueOp>(loc, ty, packed, idx));
  return out;
}

static Value buildResumeTupleValue(OpBuilder &b, Location loc, Type tupleTy,
                                   Value handle, Value done,
                                   ArrayRef<Value> payloads) {
  auto structTy = dyn_cast<LLVM::LLVMStructType>(tupleTy);
  if (!structTy)
    return Value();

  Value tuple = b.create<LLVM::UndefOp>(loc, tupleTy);
  tuple = b.create<LLVM::InsertValueOp>(loc, tuple, handle, /*position=*/0);
  tuple = b.create<LLVM::InsertValueOp>(loc, tuple, done, /*position=*/1);
  for (auto [idx, payload] : llvm::enumerate(payloads))
    tuple = b.create<LLVM::InsertValueOp>(loc, tuple, payload, idx + 2);
  return tuple;
}

static FailureOr<LLVM::LLVMFuncOp>
ensureLLVMFunctionDecl(ModuleOp module, Operation *diagOp, StringRef symName,
                       LLVM::LLVMFunctionType type, OpBuilder &moduleBuilder,
                       bool isExternal, bool makePrivate = false) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(symName)) {
    if (existing.getFunctionType() != type)
      return diagOp->emitError("symbol @")
             << symName << " exists with mismatched LLVM function type";
    return existing;
  }

  auto fn =
      moduleBuilder.create<LLVM::LLVMFuncOp>(diagOp->getLoc(), symName, type);
  if (isExternal)
    fn.setLinkage(LLVM::Linkage::External);
  if (makePrivate)
    fn.setPrivate();
  return fn;
}

static FailureOr<CoroLLVMRuntimeDecls>
ensureCoroLLVMRuntimeDecls(ModuleOp module, Operation *diagOp,
                           OpBuilder &moduleBuilder) {
  MLIRContext *ctx = module.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto tokTy = LLVM::LLVMTokenType::get(ctx);
  auto i1Ty = IntegerType::get(ctx, 1);
  auto i8Ty = IntegerType::get(ctx, 8);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  CoroLLVMRuntimeDecls decls;

  auto mallocTy = LLVM::LLVMFunctionType::get(ptrTy, {i32Ty}, false);
  auto freeTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy}, false);
  auto trapTy = LLVM::LLVMFunctionType::get(voidTy, {}, false);
  auto coroIdTy =
      LLVM::LLVMFunctionType::get(tokTy, {i32Ty, ptrTy, ptrTy, ptrTy}, false);
  auto coroSizeTy = LLVM::LLVMFunctionType::get(i32Ty, {}, false);
  auto coroBeginTy = LLVM::LLVMFunctionType::get(ptrTy, {tokTy, ptrTy}, false);
  auto coroSaveTy = LLVM::LLVMFunctionType::get(tokTy, {ptrTy}, false);
  auto coroSuspendTy = LLVM::LLVMFunctionType::get(i8Ty, {tokTy, i1Ty}, false);
  auto coroEndTy =
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i1Ty, tokTy}, false);
  auto coroFreeTy = LLVM::LLVMFunctionType::get(ptrTy, {tokTy, ptrTy}, false);
  auto coroResumeTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy}, false);
  auto coroDestroyTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy}, false);

  auto mallocFn = ensureLLVMFunctionDecl(
      module, diagOp, "coro.rt.llvm.alloc", mallocTy, moduleBuilder,
      /*isExternal=*/false, /*makePrivate=*/true);
  auto freeFn = ensureLLVMFunctionDecl(
      module, diagOp, "coro.rt.llvm.free", freeTy, moduleBuilder,
      /*isExternal=*/false, /*makePrivate=*/true);
  auto trapFn =
      ensureLLVMFunctionDecl(module, diagOp, "llvm.trap", trapTy, moduleBuilder,
                             /*isExternal=*/true);
  auto coroIdFn =
      ensureLLVMFunctionDecl(module, diagOp, "llvm.coro.id", coroIdTy,
                             moduleBuilder, /*isExternal=*/true);
  auto coroSizeFn = ensureLLVMFunctionDecl(module, diagOp, "llvm.coro.size.i32",
                                           coroSizeTy, moduleBuilder,
                                           /*isExternal=*/true);
  auto coroBeginFn = ensureLLVMFunctionDecl(module, diagOp, "llvm.coro.begin",
                                            coroBeginTy, moduleBuilder,
                                            /*isExternal=*/true);
  auto coroSaveFn =
      ensureLLVMFunctionDecl(module, diagOp, "llvm.coro.save", coroSaveTy,
                             moduleBuilder, /*isExternal=*/true);
  auto coroSuspendFn = ensureLLVMFunctionDecl(
      module, diagOp, "llvm.coro.suspend", coroSuspendTy, moduleBuilder,
      /*isExternal=*/true);
  auto coroEndFn =
      ensureLLVMFunctionDecl(module, diagOp, "llvm.coro.end", coroEndTy,
                             moduleBuilder, /*isExternal=*/true);
  auto coroFreeFn =
      ensureLLVMFunctionDecl(module, diagOp, "llvm.coro.free", coroFreeTy,
                             moduleBuilder, /*isExternal=*/true);
  auto coroResumeFn = ensureLLVMFunctionDecl(module, diagOp, "llvm.coro.resume",
                                             coroResumeTy, moduleBuilder,
                                             /*isExternal=*/true);
  auto coroDestroyFn = ensureLLVMFunctionDecl(
      module, diagOp, "llvm.coro.destroy", coroDestroyTy, moduleBuilder,
      /*isExternal=*/true);

  if (failed(mallocFn) || failed(freeFn) || failed(trapFn) ||
      failed(coroIdFn) || failed(coroSizeFn) || failed(coroBeginFn) ||
      failed(coroSaveFn) || failed(coroSuspendFn) || failed(coroEndFn) ||
      failed(coroFreeFn) || failed(coroResumeFn) || failed(coroDestroyFn))
    return failure();

  // Internal bump allocator state used to avoid host malloc/free imports.
  constexpr const char *heapGlobalSym = "coro.rt.llvm.heap_ptr";
  LLVM::GlobalOp heapGlobal =
      module.lookupSymbol<LLVM::GlobalOp>(heapGlobalSym);
  if (!heapGlobal) {
    heapGlobal = moduleBuilder.create<LLVM::GlobalOp>(
        diagOp->getLoc(), i32Ty,
        /*isConstant=*/false, LLVM::Linkage::Internal, heapGlobalSym,
        moduleBuilder.getI32IntegerAttr(70000), /*alignment=*/0,
        /*addr_space=*/0);
  } else if (heapGlobal.getGlobalType() != i32Ty || heapGlobal.getConstant()) {
    diagOp->emitError("existing @")
        << heapGlobalSym << " has incompatible type/constness";
    return failure();
  }

  if ((*mallocFn).empty()) {
    OpBuilder allocBuilder(ctx);
    Block *entry = (*mallocFn).addEntryBlock(allocBuilder);
    allocBuilder.setInsertionPointToStart(entry);
    Value heapAddr = allocBuilder.create<LLVM::AddressOfOp>(
        diagOp->getLoc(), ptrTy, heapGlobal.getSymName());
    Value cur =
        allocBuilder.create<LLVM::LoadOp>(diagOp->getLoc(), i32Ty, heapAddr);
    Value c7 = createLLVMConstI32(allocBuilder, diagOp->getLoc(), 7);
    Value cNeg8 = createLLVMConstI32(allocBuilder, diagOp->getLoc(), -8);
    Value curAligned = allocBuilder.create<LLVM::AndOp>(
        diagOp->getLoc(),
        allocBuilder.create<LLVM::AddOp>(diagOp->getLoc(), cur, c7), cNeg8);
    Value size = entry->getArgument(0);
    Value sizeAligned = allocBuilder.create<LLVM::AndOp>(
        diagOp->getLoc(),
        allocBuilder.create<LLVM::AddOp>(diagOp->getLoc(), size, c7), cNeg8);
    Value next = allocBuilder.create<LLVM::AddOp>(diagOp->getLoc(), curAligned,
                                                  sizeAligned);
    allocBuilder.create<LLVM::StoreOp>(diagOp->getLoc(), next, heapAddr);
    Value ptr = allocBuilder.create<LLVM::IntToPtrOp>(diagOp->getLoc(), ptrTy,
                                                      curAligned);
    allocBuilder.create<LLVM::ReturnOp>(diagOp->getLoc(), ptr);
  }

  if ((*freeFn).empty()) {
    OpBuilder freeBuilder(ctx);
    Block *entry = (*freeFn).addEntryBlock(freeBuilder);
    freeBuilder.setInsertionPointToStart(entry);
    freeBuilder.create<LLVM::ReturnOp>(diagOp->getLoc(), ValueRange());
  }

  decls.mallocFn = *mallocFn;
  decls.freeFn = *freeFn;
  decls.trapFn = *trapFn;
  decls.coroIdFn = *coroIdFn;
  decls.coroSizeFn = *coroSizeFn;
  decls.coroBeginFn = *coroBeginFn;
  decls.coroSaveFn = *coroSaveFn;
  decls.coroSuspendFn = *coroSuspendFn;
  decls.coroEndFn = *coroEndFn;
  decls.coroFreeFn = *coroFreeFn;
  decls.coroResumeFn = *coroResumeFn;
  decls.coroDestroyFn = *coroDestroyFn;
  return decls;
}

static LogicalResult
decodeResumeSignature(Operation *diagOp, LLVM::LLVMFuncOp resumeDecl,
                      SmallVectorImpl<Type> &resumeArgTypes,
                      SmallVectorImpl<Type> &payloadTypes) {
  LLVM::LLVMFunctionType ty = resumeDecl.getFunctionType();
  ArrayRef<Type> params = ty.getParams();
  if (params.empty() || !params.front().isInteger(64))
    return diagOp->emitError(
        "coro.resume.* must have leading i64 handle parameter");
  resumeArgTypes.assign(params.begin() + 1, params.end());

  auto retStruct = dyn_cast<LLVM::LLVMStructType>(ty.getReturnType());
  if (!retStruct || retStruct.getBody().size() < 2)
    return diagOp->emitError(
        "coro.resume.* must return LLVM struct (i64, i1, payload...)");

  ArrayRef<Type> elems = retStruct.getBody();
  if (!elems[0].isInteger(64) || !elems[1].isInteger(1))
    return diagOp->emitError(
        "coro.resume.* result struct must start with (i64, i1)");

  payloadTypes.assign(elems.begin() + 2, elems.end());
  return success();
}

static void appendTrapBlock(LLVM::LLVMFuncOp helperFunc, Block *trapBlock,
                            const CoroLLVMRuntimeDecls &decls, Location loc) {
  OpBuilder tb = OpBuilder::atBlockBegin(trapBlock);
  tb.create<LLVM::CallOp>(loc, TypeRange(), SymbolRefAttr::get(decls.trapFn),
                          ValueRange());
  tb.create<LLVM::UnreachableOp>(loc);
}

class CoroToLLVM : public impl::CoroToLLVMBase<CoroToLLVM> {
public:
  using impl::CoroToLLVMBase<CoroToLLVM>::CoroToLLVMBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    if (module.getOps<LLVM::LLVMFuncOp>().empty()) {
      bool hasFuncCoro = false;
      for (func::FuncOp f : module.getOps<func::FuncOp>()) {
        if (isCoroIntrinsicSymbol(f.getSymName()) ||
            f.getSymName().starts_with("coro.impl.")) {
          hasFuncCoro = true;
          break;
        }
      }
      if (hasFuncCoro) {
        module.emitError(
            "coro-to-llvm expects LLVM dialect input; run standard "
            "func/cf/scf/arithmetic-to-LLVM lowering first");
        signalPassFailure();
      }
      return;
    }

    llvm::StringMap<CoroLLVMKindInfo> infosByKind;
    for (LLVM::LLVMFuncOp fn : module.getOps<LLVM::LLVMFuncOp>()) {
      StringRef sym = fn.getSymName();
      if (sym.starts_with("coro.impl.")) {
        StringRef kind = sym.drop_front(strlen("coro.impl."));
        infosByKind[kind].kind = kind.str();
        infosByKind[kind].implFunc = fn;
        continue;
      }

      auto intrinsic = parseCoroIntrinsicName(sym);
      if (!intrinsic)
        continue;
      CoroLLVMKindInfo &info = infosByKind[intrinsic->suffix];
      info.kind = intrinsic->suffix;
      switch (intrinsic->kind) {
      case CoroIntrinsicKind::Spawn:
        info.spawnDecl = fn;
        break;
      case CoroIntrinsicKind::Resume:
        info.resumeDecl = fn;
        break;
      case CoroIntrinsicKind::Yield:
        info.yieldDecl = fn;
        break;
      case CoroIntrinsicKind::IsDone:
        info.isDoneDecl = fn;
        break;
      case CoroIntrinsicKind::Cancel:
        info.cancelDecl = fn;
        break;
      }
    }

    if (infosByKind.empty())
      return;

    OpBuilder moduleBuilder(ctx);
    moduleBuilder.setInsertionPointToStart(module.getBody());

    FailureOr<CoroLLVMRuntimeDecls> runtimeDecls =
        ensureCoroLLVMRuntimeDecls(module, module, moduleBuilder);
    if (failed(runtimeDecls)) {
      signalPassFailure();
      return;
    }

    SmallVector<CoroLLVMKindInfo *> orderedKinds;
    orderedKinds.reserve(infosByKind.size());
    for (auto &it : infosByKind)
      orderedKinds.push_back(&it.second);

    llvm::sort(orderedKinds,
               [](const CoroLLVMKindInfo *a, const CoroLLVMKindInfo *b) {
                 return a->kind < b->kind;
               });

    bool passFailed = false;
    for (CoroLLVMKindInfo *info : orderedKinds) {
      if (!info->spawnDecl || !info->resumeDecl || !info->implFunc) {
        module.emitError(
            "coro-to-llvm requires @coro.spawn.<kind>, "
            "@coro.resume.<kind>, and @coro.impl.<kind> for kind '")
            << info->kind << "'";
        passFailed = true;
        break;
      }

      LLVM::LLVMFunctionType spawnTy = info->spawnDecl.getFunctionType();
      if (!spawnTy.getReturnType().isInteger(64)) {
        info->spawnDecl.emitError("coro.spawn.* must return i64");
        passFailed = true;
        break;
      }
      info->spawnArgTypes.assign(spawnTy.getParams().begin(),
                                 spawnTy.getParams().end());

      if (failed(decodeResumeSignature(info->resumeDecl, info->resumeDecl,
                                       info->resumeArgTypes,
                                       info->payloadTypes))) {
        passFailed = true;
        break;
      }

      info->implPayloadPackedType =
          info->implFunc.getFunctionType().getReturnType();
      info->yieldResumePackedType =
          getPackedLLVMTypeForValues(ctx, info->resumeArgTypes);

      Type expectedImplReturn =
          getPackedLLVMTypeForValues(ctx, info->payloadTypes);
      if (expectedImplReturn != info->implPayloadPackedType) {
        info->implFunc.emitError(
            "impl return type does not match payload type packing for kind '")
            << info->kind << "'";
        passFailed = true;
        break;
      }

      LLVM::LLVMFunctionType implTy = info->implFunc.getFunctionType();
      SmallVector<Type> expectedImplParams;
      expectedImplParams.append(info->spawnArgTypes.begin(),
                                info->spawnArgTypes.end());
      expectedImplParams.append(info->resumeArgTypes.begin(),
                                info->resumeArgTypes.end());
      if (!typeSequencesEqual(implTy.getParams(), expectedImplParams)) {
        info->implFunc.emitError("impl parameter types must equal spawn args + "
                                 "resume args for kind '")
            << info->kind << "'";
        passFailed = true;
        break;
      }

      if (info->yieldDecl) {
        LLVM::LLVMFunctionType yieldTy = info->yieldDecl.getFunctionType();
        ArrayRef<Type> yieldParams = yieldTy.getParams();
        if (!typeSequencesEqual(yieldParams, info->payloadTypes)) {
          info->yieldDecl.emitError(
              "coro.yield.* operand types must match payload types");
          passFailed = true;
          break;
        }
        if (yieldTy.getReturnType() != info->yieldResumePackedType) {
          info->yieldDecl.emitError("coro.yield.* result type must match "
                                    "packed resume argument type");
          passFailed = true;
          break;
        }
      }

      // Runtime object layout:
      // [0]=magic(i64), [1]=done(i1), [2]=frame(ptr),
      // [spawn args...], [resume args...], [payload...]
      SmallVector<Type> fields;
      fields.push_back(IntegerType::get(ctx, 64));
      fields.push_back(IntegerType::get(ctx, 1));
      fields.push_back(ptrTy);
      info->spawnBase = fields.size();
      fields.append(info->spawnArgTypes.begin(), info->spawnArgTypes.end());
      info->resumeBase = fields.size();
      fields.append(info->resumeArgTypes.begin(), info->resumeArgTypes.end());
      info->payloadBase = fields.size();
      fields.append(info->payloadTypes.begin(), info->payloadTypes.end());
      info->runtimeType = LLVM::LLVMStructType::getLiteral(ctx, fields);

      info->magic = static_cast<uint64_t>(llvm::hash_value(info->kind));
      if (info->magic == 0)
        info->magic = 1;

      std::string k = sanitizeSymbolPiece(info->kind);
      info->spawnHelperSym = ("coro.rt.llvm.spawn." + k);
      info->resumeHelperSym = ("coro.rt.llvm.resume." + k);
      info->isDoneHelperSym = ("coro.rt.llvm.is_done." + k);
      info->cancelHelperSym = ("coro.rt.llvm.cancel." + k);
    }

    if (passFailed) {
      signalPassFailure();
      return;
    }

    // 1) Rebuild implementation functions with runtime pointer argument and
    //    lower yield calls to llvm.coro suspend/resume logic.
    for (CoroLLVMKindInfo *info : orderedKinds) {
      LLVM::LLVMFuncOp implFn = info->implFunc;
      if (implFn.use_empty() == false) {
        implFn.emitError("direct calls to @coro.impl.* are not supported in "
                         "coro-to-llvm; use coro intrinsics only");
        passFailed = true;
        break;
      }

      if (implFn.empty()) {
        implFn.emitError("coro implementation must have a body");
        passFailed = true;
        break;
      }

      LLVM::LLVMFunctionType oldTy = implFn.getFunctionType();
      SmallVector<Type> newParams(oldTy.getParams().begin(),
                                  oldTy.getParams().end());
      newParams.push_back(ptrTy);
      auto newTy = LLVM::LLVMFunctionType::get(oldTy.getReturnType(), newParams,
                                               oldTy.isVarArg());
      implFn.setFunctionType(newTy);
      Block &entry = implFn.getBody().front();
      BlockArgument rtArg = entry.addArgument(ptrTy, implFn.getLoc());
      SmallVector<LLVM::ReturnOp> originalReturns;
      implFn.walk(
          [&](LLVM::ReturnOp retOp) { originalReturns.push_back(retOp); });

      SmallVector<LLVM::CallOp> yields;
      implFn.walk([&](LLVM::CallOp callOp) {
        FlatSymbolRefAttr callee = callOp.getCalleeAttr();
        if (!callee)
          return;
        auto intrinsic = parseCoroIntrinsicName(callee.getValue());
        if (!intrinsic || intrinsic->kind != CoroIntrinsicKind::Yield)
          return;
        if (intrinsic->suffix != info->kind) {
          callOp.emitError("yield kind does not match impl kind");
          passFailed = true;
          return;
        }
        yields.push_back(callOp);
      });
      if (passFailed)
        break;

      info->hasYield = !yields.empty();

      Value coroId;
      Value coroHdl;
      Block *commonSuspendBlock = nullptr;
      if (info->hasYield) {
        implFn->setAttr(
            "passthrough",
            ArrayAttr::get(ctx, {StringAttr::get(ctx, "presplitcoroutine")}));

        OpBuilder eb = OpBuilder::atBlockBegin(&entry);
        Value zeroI32 = createLLVMConstI32(eb, implFn.getLoc(), 0);
        Value nullPtr = createLLVMNullPtr(eb, implFn.getLoc());
        coroId =
            eb.create<LLVM::CallOp>(
                  implFn.getLoc(), TypeRange{LLVM::LLVMTokenType::get(ctx)},
                  SymbolRefAttr::get(runtimeDecls->coroIdFn),
                  ValueRange{zeroI32, nullPtr, nullPtr, nullPtr})
                .getResult();
        Value frameSizeI32 =
            eb.create<LLVM::CallOp>(
                  implFn.getLoc(), TypeRange{eb.getI32Type()},
                  SymbolRefAttr::get(runtimeDecls->coroSizeFn), ValueRange())
                .getResult();
        Value frameMem =
            eb.create<LLVM::CallOp>(implFn.getLoc(), TypeRange{ptrTy},
                                    SymbolRefAttr::get(runtimeDecls->mallocFn),
                                    frameSizeI32)
                .getResult();
        coroHdl = eb.create<LLVM::CallOp>(
                        implFn.getLoc(), TypeRange{ptrTy},
                        SymbolRefAttr::get(runtimeDecls->coroBeginFn),
                        ValueRange{coroId, frameMem})
                      .getResult();
        storeRuntimeField(eb, implFn.getLoc(), rtArg, info->runtimeType,
                          info->frameField, coroHdl);

        // Use a single suspend-return block per coroutine impl so LLVM sees
        // exactly one fallthrough coro.end in the function.
        Region &r = implFn.getBody();
        commonSuspendBlock = new Block();
        r.push_back(commonSuspendBlock);
        OpBuilder sb = OpBuilder::atBlockBegin(commonSuspendBlock);
        Value noneToken = sb.create<LLVM::NoneTokenOp>(implFn.getLoc());
        Value unwind = createLLVMConstI1(sb, implFn.getLoc(), false);
        sb.create<LLVM::CallOp>(implFn.getLoc(), TypeRange(),
                                SymbolRefAttr::get(runtimeDecls->coroEndFn),
                                ValueRange{coroHdl, unwind, noneToken});
        SmallVector<Value> rets;
        if (!isa<LLVM::LLVMVoidType>(info->implPayloadPackedType))
          rets.push_back(sb.create<LLVM::UndefOp>(implFn.getLoc(),
                                                  info->implPayloadPackedType));
        sb.create<LLVM::ReturnOp>(implFn.getLoc(), rets);
      }

      // Rewrite each yield to real suspend/resume machinery.
      for (LLVM::CallOp yieldCall : yields) {
        if (!yieldCall || !yieldCall->getBlock())
          continue;

        Block *preBlock = yieldCall->getBlock();
        auto nextIt = std::next(Block::iterator(yieldCall));
        Block *contBlock = preBlock->splitBlock(nextIt);

        // Thread resumed value into continuation block.
        if (yieldCall.getNumResults() == 1)
          contBlock->addArgument(yieldCall.getResult().getType(),
                                 yieldCall.getLoc());

        if (yieldCall.getNumResults() == 1) {
          Value oldRes = yieldCall.getResult();
          Value contArg = contBlock->getArgument(0);
          oldRes.replaceAllUsesWith(contArg);
        }

        // Create resume/cleanup blocks.
        Region &r = implFn.getBody();
        Block *resumeBlock = new Block();
        Block *cleanupBlock = new Block();
        r.push_back(resumeBlock);
        r.push_back(cleanupBlock);

        OpBuilder pb(yieldCall);
        for (auto [idx, payload] : llvm::enumerate(yieldCall.getOperands())) {
          storeRuntimeField(pb, yieldCall.getLoc(), rtArg, info->runtimeType,
                            info->payloadBase + idx, payload);
        }
        storeRuntimeField(pb, yieldCall.getLoc(), rtArg, info->runtimeType,
                          info->doneField,
                          createLLVMConstI1(pb, yieldCall.getLoc(), false));

        Value saveTok =
            pb.create<LLVM::CallOp>(
                  yieldCall.getLoc(), TypeRange{LLVM::LLVMTokenType::get(ctx)},
                  SymbolRefAttr::get(runtimeDecls->coroSaveFn),
                  ValueRange{coroHdl})
                .getResult();
        Value isFinal = createLLVMConstI1(pb, yieldCall.getLoc(), false);
        Value suspendCode =
            pb.create<LLVM::CallOp>(
                  yieldCall.getLoc(), TypeRange{pb.getI8Type()},
                  SymbolRefAttr::get(runtimeDecls->coroSuspendFn),
                  ValueRange{saveTok, isFinal})
                .getResult();
        Value suspendCodeI32 = pb.create<LLVM::SExtOp>(
            yieldCall.getLoc(), pb.getI32Type(), suspendCode);

        SmallVector<int32_t, 2> caseVals = {0, 1};
        SmallVector<Block *, 2> caseDests = {resumeBlock, cleanupBlock};
        pb.create<LLVM::SwitchOp>(
            yieldCall.getLoc(), suspendCodeI32,
            /*defaultDestination=*/commonSuspendBlock,
            /*defaultOperands=*/ValueRange(),
            /*caseValues=*/caseVals,
            /*caseDestinations=*/caseDests,
            /*caseOperands=*/ArrayRef<ValueRange>({ValueRange(), ValueRange()}),
            /*branchWeights=*/ArrayRef<int32_t>());
        yieldCall.erase();

        // Resume block: load resume args and continue.
        OpBuilder rb = OpBuilder::atBlockBegin(resumeBlock);
        SmallVector<Value> resumeValues;
        for (auto [idx, ty] : llvm::enumerate(info->resumeArgTypes)) {
          resumeValues.push_back(loadRuntimeField(rb, implFn.getLoc(), rtArg,
                                                  info->runtimeType, ty,
                                                  info->resumeBase + idx));
        }
        SmallVector<Value> brArgs;
        if (!resumeValues.empty()) {
          Value packed = packLLVMValues(
              rb, implFn.getLoc(), info->yieldResumePackedType, resumeValues);
          brArgs.push_back(packed);
        }
        rb.create<LLVM::BrOp>(implFn.getLoc(), brArgs, contBlock);

        // Cleanup block: free frame (destroy path), set state, then go suspend.
        OpBuilder cb = OpBuilder::atBlockBegin(cleanupBlock);
        Value frameMem = cb.create<LLVM::CallOp>(
                               implFn.getLoc(), TypeRange{ptrTy},
                               SymbolRefAttr::get(runtimeDecls->coroFreeFn),
                               ValueRange{coroId, coroHdl})
                             .getResult();
        cb.create<LLVM::CallOp>(implFn.getLoc(), TypeRange(),
                                SymbolRefAttr::get(runtimeDecls->freeFn),
                                frameMem);
        storeRuntimeField(cb, implFn.getLoc(), rtArg, info->runtimeType,
                          info->frameField,
                          createLLVMNullPtr(cb, implFn.getLoc()));
        storeRuntimeField(cb, implFn.getLoc(), rtArg, info->runtimeType,
                          info->doneField,
                          createLLVMConstI1(cb, implFn.getLoc(), true));
        cb.create<LLVM::BrOp>(implFn.getLoc(), ValueRange(),
                              commonSuspendBlock);
      }

      // Rewrite original returns to publish done/payload.
      for (LLVM::ReturnOp retOp : originalReturns) {
        if (!retOp || !retOp->getBlock())
          continue;

        OpBuilder rb(retOp);
        SmallVector<Value> payloadValues;
        if (!info->payloadTypes.empty()) {
          if (retOp.getNumOperands() != 1) {
            retOp.emitError("impl return must carry packed payload value");
            passFailed = true;
            break;
          }
          payloadValues = unpackLLVMValues(
              rb, retOp.getLoc(), retOp.getOperand(0), info->payloadTypes);
        }
        for (auto [idx, payload] : llvm::enumerate(payloadValues)) {
          storeRuntimeField(rb, retOp.getLoc(), rtArg, info->runtimeType,
                            info->payloadBase + idx, payload);
        }
        storeRuntimeField(rb, retOp.getLoc(), rtArg, info->runtimeType,
                          info->doneField,
                          createLLVMConstI1(rb, retOp.getLoc(), true));
        if (!info->hasYield) {
          // Non-suspending impls complete immediately and never own a coro
          // frame.
          storeRuntimeField(rb, retOp.getLoc(), rtArg, info->runtimeType,
                            info->frameField,
                            createLLVMNullPtr(rb, retOp.getLoc()));
          continue;
        }

        // For suspending coroutines, lower source-level return to LLVM final
        // suspend form. A direct return from a resumed coroutine is not valid
        // and gets lowered to unreachable by coro-split.
        Region &r = implFn.getBody();
        Block *trapBlock = new Block();
        Block *cleanupBlock = new Block();
        r.push_back(trapBlock);
        r.push_back(cleanupBlock);

        Value noneToken = rb.create<LLVM::NoneTokenOp>(retOp.getLoc());
        Value isFinal = createLLVMConstI1(rb, retOp.getLoc(), true);
        Value suspendCode =
            rb.create<LLVM::CallOp>(
                  retOp.getLoc(), TypeRange{rb.getI8Type()},
                  SymbolRefAttr::get(runtimeDecls->coroSuspendFn),
                  ValueRange{noneToken, isFinal})
                .getResult();
        Value suspendCodeI32 = rb.create<LLVM::SExtOp>(
            retOp.getLoc(), rb.getI32Type(), suspendCode);
        SmallVector<int32_t, 2> caseVals = {0, 1};
        SmallVector<Block *, 2> caseDests = {trapBlock, cleanupBlock};
        rb.create<LLVM::SwitchOp>(
            retOp.getLoc(), suspendCodeI32,
            /*defaultDestination=*/commonSuspendBlock,
            /*defaultOperands=*/ValueRange(),
            /*caseValues=*/caseVals,
            /*caseDestinations=*/caseDests,
            /*caseOperands=*/ArrayRef<ValueRange>({ValueRange(), ValueRange()}),
            /*branchWeights=*/ArrayRef<int32_t>());
        retOp.erase();

        appendTrapBlock(implFn, trapBlock, *runtimeDecls, implFn.getLoc());

        OpBuilder cb = OpBuilder::atBlockBegin(cleanupBlock);
        Value frameMem = cb.create<LLVM::CallOp>(
                               implFn.getLoc(), TypeRange{ptrTy},
                               SymbolRefAttr::get(runtimeDecls->coroFreeFn),
                               ValueRange{coroId, coroHdl})
                             .getResult();
        cb.create<LLVM::CallOp>(implFn.getLoc(), TypeRange(),
                                SymbolRefAttr::get(runtimeDecls->freeFn),
                                frameMem);
        storeRuntimeField(cb, implFn.getLoc(), rtArg, info->runtimeType,
                          info->frameField,
                          createLLVMNullPtr(cb, implFn.getLoc()));
        cb.create<LLVM::BrOp>(implFn.getLoc(), ValueRange(),
                              commonSuspendBlock);
      }
      if (passFailed)
        break;

      // Create per-kind runtime helper functions.
      auto spawnHelperTy = info->spawnDecl.getFunctionType();
      auto resumeHelperTy = info->resumeDecl.getFunctionType();
      auto doneHelperTy = info->isDoneDecl ? info->isDoneDecl.getFunctionType()
                                           : LLVM::LLVMFunctionType();
      auto cancelHelperTy = info->cancelDecl
                                ? info->cancelDecl.getFunctionType()
                                : LLVM::LLVMFunctionType();

      auto spawnHelper = ensureLLVMFunctionDecl(
          module, info->spawnDecl, info->spawnHelperSym, spawnHelperTy,
          moduleBuilder, /*isExternal=*/false, /*makePrivate=*/true);
      auto resumeHelper = ensureLLVMFunctionDecl(
          module, info->resumeDecl, info->resumeHelperSym, resumeHelperTy,
          moduleBuilder, /*isExternal=*/false, /*makePrivate=*/true);
      if (failed(spawnHelper) || failed(resumeHelper)) {
        passFailed = true;
        break;
      }

      std::optional<LLVM::LLVMFuncOp> doneHelper;
      std::optional<LLVM::LLVMFuncOp> cancelHelper;
      if (info->isDoneDecl) {
        FailureOr<LLVM::LLVMFuncOp> helper = ensureLLVMFunctionDecl(
            module, info->isDoneDecl, info->isDoneHelperSym, doneHelperTy,
            moduleBuilder, /*isExternal=*/false, /*makePrivate=*/true);
        if (failed(helper)) {
          passFailed = true;
          break;
        }
        doneHelper = *helper;
      }
      if (info->cancelDecl) {
        FailureOr<LLVM::LLVMFuncOp> helper = ensureLLVMFunctionDecl(
            module, info->cancelDecl, info->cancelHelperSym, cancelHelperTy,
            moduleBuilder, /*isExternal=*/false, /*makePrivate=*/true);
        if (failed(helper)) {
          passFailed = true;
          break;
        }
        cancelHelper = *helper;
      }

      // Build spawn helper body.
      if (spawnHelper->empty()) {
        OpBuilder sb(ctx);
        Block *entryBlock = spawnHelper->addEntryBlock(sb);
        sb.setInsertionPointToStart(entryBlock);

        Value nullPtr = createLLVMNullPtr(sb, spawnHelper->getLoc());
        Value sizePtr = sb.create<LLVM::GEPOp>(spawnHelper->getLoc(), ptrTy,
                                               info->runtimeType, nullPtr,
                                               ArrayRef<LLVM::GEPArg>{1});
        Value sizeI32 = sb.create<LLVM::PtrToIntOp>(spawnHelper->getLoc(),
                                                    sb.getI32Type(), sizePtr);
        Value rtPtr = sb.create<LLVM::CallOp>(
                            spawnHelper->getLoc(), TypeRange{ptrTy},
                            SymbolRefAttr::get(runtimeDecls->mallocFn), sizeI32)
                          .getResult();

        storeRuntimeField(
            sb, spawnHelper->getLoc(), rtPtr, info->runtimeType,
            info->magicField,
            createLLVMConstU64(sb, spawnHelper->getLoc(), info->magic));
        storeRuntimeField(sb, spawnHelper->getLoc(), rtPtr, info->runtimeType,
                          info->doneField,
                          createLLVMConstI1(sb, spawnHelper->getLoc(), false));
        storeRuntimeField(sb, spawnHelper->getLoc(), rtPtr, info->runtimeType,
                          info->frameField,
                          createLLVMNullPtr(sb, spawnHelper->getLoc()));

        for (auto [idx, arg] : llvm::enumerate(entryBlock->getArguments())) {
          storeRuntimeField(sb, spawnHelper->getLoc(), rtPtr, info->runtimeType,
                            info->spawnBase + idx, arg);
        }

        Value handle = sb.create<LLVM::PtrToIntOp>(spawnHelper->getLoc(),
                                                   sb.getI64Type(), rtPtr);
        sb.create<LLVM::ReturnOp>(spawnHelper->getLoc(), handle);
      }

      // Build resume helper body.
      if (resumeHelper->empty()) {
        OpBuilder rb(ctx);
        Block *entryBlock = resumeHelper->addEntryBlock(rb);
        Region &r = resumeHelper->getBody();
        auto *trapBlock = new Block();
        auto *checkDoneBlock = new Block();
        auto *runBlock = new Block();
        auto *startBlock = new Block();
        auto *firstStartBlock = new Block();
        auto *resumeExistingBlock = new Block();
        auto *collectBlock = new Block();
        r.push_back(trapBlock);
        r.push_back(checkDoneBlock);
        r.push_back(runBlock);
        r.push_back(startBlock);
        r.push_back(firstStartBlock);
        r.push_back(resumeExistingBlock);
        r.push_back(collectBlock);

        // entry: decode handle + validate magic
        rb.setInsertionPointToStart(entryBlock);
        Value handle = entryBlock->getArgument(0);
        Value rtPtr =
            rb.create<LLVM::IntToPtrOp>(resumeHelper->getLoc(), ptrTy, handle);
        Value nullPtr = createLLVMNullPtr(rb, resumeHelper->getLoc());
        Value isNull = rb.create<LLVM::ICmpOp>(
            resumeHelper->getLoc(), LLVM::ICmpPredicate::eq, rtPtr, nullPtr);
        rb.create<LLVM::CondBrOp>(resumeHelper->getLoc(), isNull, trapBlock,
                                  ValueRange(), checkDoneBlock, ValueRange());

        // check magic / done
        OpBuilder cb = OpBuilder::atBlockBegin(checkDoneBlock);
        Value magic = loadRuntimeField(cb, resumeHelper->getLoc(), rtPtr,
                                       info->runtimeType, cb.getI64Type(),
                                       info->magicField);
        Value magicOk = cb.create<LLVM::ICmpOp>(
            resumeHelper->getLoc(), LLVM::ICmpPredicate::eq, magic,
            createLLVMConstU64(cb, resumeHelper->getLoc(), info->magic));
        cb.create<LLVM::CondBrOp>(resumeHelper->getLoc(), magicOk, runBlock,
                                  ValueRange(), trapBlock, ValueRange());

        OpBuilder runb = OpBuilder::atBlockBegin(runBlock);
        Value doneNow = loadRuntimeField(runb, resumeHelper->getLoc(), rtPtr,
                                         info->runtimeType, runb.getI1Type(),
                                         info->doneField);
        Block *doneFastBlock = new Block();
        r.push_back(doneFastBlock);
        runb.create<LLVM::CondBrOp>(resumeHelper->getLoc(), doneNow,
                                    doneFastBlock, ValueRange(), startBlock,
                                    ValueRange());

        // done fast path: return current payload, done=true
        OpBuilder doneb = OpBuilder::atBlockBegin(doneFastBlock);
        SmallVector<Value> donePayloads;
        for (auto [idx, ty] : llvm::enumerate(info->payloadTypes)) {
          donePayloads.push_back(loadRuntimeField(doneb, resumeHelper->getLoc(),
                                                  rtPtr, info->runtimeType, ty,
                                                  info->payloadBase + idx));
        }
        Value doneTuple = buildResumeTupleValue(doneb, resumeHelper->getLoc(),
                                                resumeHelperTy.getReturnType(),
                                                handle, doneNow, donePayloads);
        doneb.create<LLVM::ReturnOp>(resumeHelper->getLoc(), doneTuple);

        // start block: store new resume args, dispatch first-start vs resume.
        OpBuilder startb = OpBuilder::atBlockBegin(startBlock);
        for (unsigned i = 0; i < info->resumeArgTypes.size(); ++i) {
          Value arg = entryBlock->getArgument(i + 1);
          storeRuntimeField(startb, resumeHelper->getLoc(), rtPtr,
                            info->runtimeType, info->resumeBase + i, arg);
        }
        Value frame =
            loadRuntimeField(startb, resumeHelper->getLoc(), rtPtr,
                             info->runtimeType, ptrTy, info->frameField);
        Value frameNull = startb.create<LLVM::ICmpOp>(
            resumeHelper->getLoc(), LLVM::ICmpPredicate::eq, frame, nullPtr);
        startb.create<LLVM::CondBrOp>(resumeHelper->getLoc(), frameNull,
                                      firstStartBlock, ValueRange(),
                                      resumeExistingBlock, ValueRange());

        // first start path
        OpBuilder firstb = OpBuilder::atBlockBegin(firstStartBlock);
        SmallVector<Value> callArgs;
        callArgs.reserve(info->spawnArgTypes.size() +
                         info->resumeArgTypes.size() + 1);
        for (auto [idx, ty] : llvm::enumerate(info->spawnArgTypes)) {
          callArgs.push_back(loadRuntimeField(firstb, resumeHelper->getLoc(),
                                              rtPtr, info->runtimeType, ty,
                                              info->spawnBase + idx));
        }
        for (unsigned i = 0; i < info->resumeArgTypes.size(); ++i)
          callArgs.push_back(entryBlock->getArgument(i + 1));
        callArgs.push_back(rtPtr);
        firstb.create<LLVM::CallOp>(
            resumeHelper->getLoc(), TypeRange{info->implPayloadPackedType},
            SymbolRefAttr::get(info->implFunc), callArgs);
        firstb.create<LLVM::BrOp>(resumeHelper->getLoc(), ValueRange(),
                                  collectBlock);

        // resume-existing path (frame != null)
        OpBuilder rib = OpBuilder::atBlockBegin(resumeExistingBlock);
        Value frameForResume =
            loadRuntimeField(rib, resumeHelper->getLoc(), rtPtr,
                             info->runtimeType, ptrTy, info->frameField);
        rib.create<LLVM::CallOp>(resumeHelper->getLoc(), TypeRange(),
                                 SymbolRefAttr::get(runtimeDecls->coroResumeFn),
                                 frameForResume);
        rib.create<LLVM::BrOp>(resumeHelper->getLoc(), ValueRange(),
                               collectBlock);

        OpBuilder collb = OpBuilder::atBlockBegin(collectBlock);
        Value doneFinal = loadRuntimeField(collb, resumeHelper->getLoc(), rtPtr,
                                           info->runtimeType, collb.getI1Type(),
                                           info->doneField);
        SmallVector<Value> payloadVals;
        for (auto [idx, ty] : llvm::enumerate(info->payloadTypes)) {
          payloadVals.push_back(loadRuntimeField(collb, resumeHelper->getLoc(),
                                                 rtPtr, info->runtimeType, ty,
                                                 info->payloadBase + idx));
        }
        Value tuple = buildResumeTupleValue(collb, resumeHelper->getLoc(),
                                            resumeHelperTy.getReturnType(),
                                            handle, doneFinal, payloadVals);
        collb.create<LLVM::ReturnOp>(resumeHelper->getLoc(), tuple);

        appendTrapBlock(*resumeHelper, trapBlock, *runtimeDecls,
                        resumeHelper->getLoc());
      }

      // Build is_done helper body.
      if (doneHelper && doneHelper->empty()) {
        OpBuilder db(ctx);
        Block *entryBlock = doneHelper->addEntryBlock(db);
        Region &r = doneHelper->getBody();
        auto *trapBlock = new Block();
        auto *okBlock = new Block();
        r.push_back(trapBlock);
        r.push_back(okBlock);

        db.setInsertionPointToStart(entryBlock);
        Value handle = entryBlock->getArgument(0);
        Value rtPtr =
            db.create<LLVM::IntToPtrOp>(doneHelper->getLoc(), ptrTy, handle);
        Value nullPtr = createLLVMNullPtr(db, doneHelper->getLoc());
        Value isNull = db.create<LLVM::ICmpOp>(
            doneHelper->getLoc(), LLVM::ICmpPredicate::eq, rtPtr, nullPtr);
        db.create<LLVM::CondBrOp>(doneHelper->getLoc(), isNull, trapBlock,
                                  ValueRange(), okBlock, ValueRange());

        OpBuilder okb = OpBuilder::atBlockBegin(okBlock);
        Value magic = loadRuntimeField(okb, doneHelper->getLoc(), rtPtr,
                                       info->runtimeType, okb.getI64Type(),
                                       info->magicField);
        Value magicOk = okb.create<LLVM::ICmpOp>(
            doneHelper->getLoc(), LLVM::ICmpPredicate::eq, magic,
            createLLVMConstU64(okb, doneHelper->getLoc(), info->magic));
        Block *retBlock = new Block();
        r.push_back(retBlock);
        okb.create<LLVM::CondBrOp>(doneHelper->getLoc(), magicOk, retBlock,
                                   ValueRange(), trapBlock, ValueRange());

        OpBuilder retb = OpBuilder::atBlockBegin(retBlock);
        Value done = loadRuntimeField(retb, doneHelper->getLoc(), rtPtr,
                                      info->runtimeType, retb.getI1Type(),
                                      info->doneField);
        retb.create<LLVM::ReturnOp>(doneHelper->getLoc(), done);

        appendTrapBlock(*doneHelper, trapBlock, *runtimeDecls,
                        doneHelper->getLoc());
      }

      // Build cancel helper body.
      if (cancelHelper && cancelHelper->empty()) {
        OpBuilder cb(ctx);
        Block *entryBlock = cancelHelper->addEntryBlock(cb);
        Region &r = cancelHelper->getBody();
        auto *trapBlock = new Block();
        auto *okBlock = new Block();
        auto *freeBlock = new Block();
        auto *retBlock = new Block();
        r.push_back(trapBlock);
        r.push_back(okBlock);
        r.push_back(freeBlock);
        r.push_back(retBlock);

        cb.setInsertionPointToStart(entryBlock);
        Value handle = entryBlock->getArgument(0);
        Value rtPtr =
            cb.create<LLVM::IntToPtrOp>(cancelHelper->getLoc(), ptrTy, handle);
        Value nullPtr = createLLVMNullPtr(cb, cancelHelper->getLoc());
        Value isNull = cb.create<LLVM::ICmpOp>(
            cancelHelper->getLoc(), LLVM::ICmpPredicate::eq, rtPtr, nullPtr);
        cb.create<LLVM::CondBrOp>(cancelHelper->getLoc(), isNull, trapBlock,
                                  ValueRange(), okBlock, ValueRange());

        OpBuilder okb = OpBuilder::atBlockBegin(okBlock);
        Value magic = loadRuntimeField(okb, cancelHelper->getLoc(), rtPtr,
                                       info->runtimeType, okb.getI64Type(),
                                       info->magicField);
        Value magicOk = okb.create<LLVM::ICmpOp>(
            cancelHelper->getLoc(), LLVM::ICmpPredicate::eq, magic,
            createLLVMConstU64(okb, cancelHelper->getLoc(), info->magic));
        okb.create<LLVM::CondBrOp>(cancelHelper->getLoc(), magicOk, freeBlock,
                                   ValueRange(), trapBlock, ValueRange());

        OpBuilder freeb = OpBuilder::atBlockBegin(freeBlock);
        Value frame =
            loadRuntimeField(freeb, cancelHelper->getLoc(), rtPtr,
                             info->runtimeType, ptrTy, info->frameField);
        Value frameNull = freeb.create<LLVM::ICmpOp>(
            cancelHelper->getLoc(), LLVM::ICmpPredicate::eq, frame, nullPtr);
        Block *destroyBlock = new Block();
        r.push_back(destroyBlock);
        freeb.create<LLVM::CondBrOp>(cancelHelper->getLoc(), frameNull,
                                     retBlock, ValueRange(), destroyBlock,
                                     ValueRange());

        OpBuilder destrb = OpBuilder::atBlockBegin(destroyBlock);
        destrb.create<LLVM::CallOp>(
            cancelHelper->getLoc(), TypeRange(),
            SymbolRefAttr::get(runtimeDecls->coroDestroyFn), frame);
        destrb.create<LLVM::BrOp>(cancelHelper->getLoc(), ValueRange(),
                                  retBlock);

        OpBuilder retb = OpBuilder::atBlockBegin(retBlock);
        retb.create<LLVM::CallOp>(cancelHelper->getLoc(), TypeRange(),
                                  SymbolRefAttr::get(runtimeDecls->freeFn),
                                  rtPtr);
        retb.create<LLVM::ReturnOp>(cancelHelper->getLoc(), ValueRange());

        appendTrapBlock(*cancelHelper, trapBlock, *runtimeDecls,
                        cancelHelper->getLoc());
      }
    }

    if (passFailed) {
      signalPassFailure();
      return;
    }

    // 2) Rewrite intrinsic callsites to helper functions.
    SmallVector<LLVM::CallOp> calls;
    module.walk([&](LLVM::CallOp callOp) { calls.push_back(callOp); });
    for (LLVM::CallOp callOp : calls) {
      if (!callOp || !callOp->getBlock())
        continue;
      FlatSymbolRefAttr callee = callOp.getCalleeAttr();
      if (!callee)
        continue;

      auto intrinsic = parseCoroIntrinsicName(callee.getValue());
      if (!intrinsic)
        continue;
      auto it = infosByKind.find(intrinsic->suffix);
      if (it == infosByKind.end())
        continue;
      CoroLLVMKindInfo &info = it->second;

      StringRef helperSym;
      switch (intrinsic->kind) {
      case CoroIntrinsicKind::Spawn:
        helperSym = info.spawnHelperSym;
        break;
      case CoroIntrinsicKind::Resume:
        helperSym = info.resumeHelperSym;
        break;
      case CoroIntrinsicKind::IsDone:
        if (!info.isDoneDecl)
          continue;
        helperSym = info.isDoneHelperSym;
        break;
      case CoroIntrinsicKind::Cancel:
        if (!info.cancelDecl)
          continue;
        helperSym = info.cancelHelperSym;
        break;
      case CoroIntrinsicKind::Yield:
        continue;
      }

      OpBuilder b(callOp);
      auto repl = b.create<LLVM::CallOp>(
          callOp.getLoc(), callOp.getResultTypes(),
          SymbolRefAttr::get(ctx, helperSym), callOp.getOperands());
      if (callOp.getNumResults() == 1)
        callOp.getResult().replaceAllUsesWith(repl.getResult());
      callOp.erase();
    }

    // 3) Drop now-unused coro intrinsic declarations.
    SmallVector<LLVM::LLVMFuncOp> eraseDecls;
    for (LLVM::LLVMFuncOp fn : module.getOps<LLVM::LLVMFuncOp>()) {
      if (!isCoroIntrinsicSymbol(fn.getSymName()))
        continue;
      if (fn.use_empty())
        eraseDecls.push_back(fn);
    }
    for (LLVM::LLVMFuncOp fn : eraseDecls)
      fn.erase();
  }
};

} // namespace mlir::wami
