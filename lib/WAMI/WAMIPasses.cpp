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
  SmallVector<Type> resumeResultTypes;
};

struct HandleBinding {
  Value handleValue;
  Value contValue;
  FlatSymbolRefAttr contType;
  SmallVector<Type> expectedResumeArgs;
  SmallVector<Type> expectedResumeResults;
  std::string kind;
  Operation *spawnOp = nullptr;
  bool consumed = false;
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
        if (!usage.resumeArgTypes.empty() &&
            !typeSequencesEqual(resumeArgTypes, usage.resumeArgTypes)) {
          callOp.emitError(
              "inconsistent coro.resume.* argument types for kind '")
              << intrinsic->suffix << "'";
          failedVerify = true;
          return;
        }
        if (!usage.resumeResultTypes.empty() &&
            !typeSequencesEqual(callOp.getResultTypes(),
                                usage.resumeResultTypes)) {
          callOp.emitError("inconsistent coro.resume.* result types for kind '")
              << intrinsic->suffix << "'";
          failedVerify = true;
          return;
        }

        usage.resumeArgTypes.assign(resumeArgTypes.begin(),
                                    resumeArgTypes.end());
        usage.resumeResultTypes.assign(callOp.getResultTypes().begin(),
                                       callOp.getResultTypes().end());
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
      if (!usage.resumeResultTypes.empty() &&
          !typeSequencesEqual(implType.getResults(), usage.resumeResultTypes)) {
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

    for (wasmssa::FuncOp funcOp : module.getOps<wasmssa::FuncOp>()) {
      llvm::MapVector<Value, HandleBinding> handleBindings;
      SmallVector<wasmssa::FuncCallOp> calls;
      SmallVector<Operation *> eraseLater;

      funcOp.walk([&](wasmssa::FuncCallOp callOp) { calls.push_back(callOp); });

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

          FailureOr<FlatSymbolRefAttr> srcContRef =
              ensureContTypeForBoundPrefix(module, callOp, intrinsic->suffix,
                                           implType, 0, moduleBuilder);
          if (failed(srcContRef)) {
            passFailed = true;
            break;
          }

          std::string implSym = buildImplSymbolForKind(intrinsic->suffix);
          Type funcrefType = parseWamiTypeOrEmit(
              callOp, ("!wami.funcref<@" + implSym + ">").c_str());
          Type srcContType = parseWamiTypeOrEmit(
              callOp,
              ("!wami.cont<@" + srcContRef->getValue().str() + ">").c_str());
          if (!funcrefType || !srcContType) {
            passFailed = true;
            break;
          }

          OperationState refFuncState(callOp.getLoc(), "wami.ref.func");
          refFuncState.addAttribute("func",
                                    FlatSymbolRefAttr::get(ctx, implSym));
          refFuncState.addTypes(funcrefType);
          Operation *refFuncOp = b.create(refFuncState);

          OperationState contNewState(callOp.getLoc(), "wami.cont.new");
          contNewState.addOperands(refFuncOp->getResult(0));
          contNewState.addAttribute("cont_type", *srcContRef);
          contNewState.addTypes(srcContType);
          Operation *contValueOp = b.create(contNewState);

          Value contValue = contValueOp->getResult(0);
          FlatSymbolRefAttr boundContRef = *srcContRef;
          SmallVector<Type> expectedResumeArgs;
          expectedResumeArgs.append(implType.getInputs().begin() + boundCount,
                                    implType.getInputs().end());

          if (boundCount > 0) {
            FailureOr<FlatSymbolRefAttr> dstContRef =
                ensureContTypeForBoundPrefix(module, callOp, intrinsic->suffix,
                                             implType, boundCount,
                                             moduleBuilder);
            if (failed(dstContRef)) {
              passFailed = true;
              break;
            }

            Type dstContType = parseWamiTypeOrEmit(
                callOp,
                ("!wami.cont<@" + dstContRef->getValue().str() + ">").c_str());
            if (!dstContType) {
              passFailed = true;
              break;
            }

            SmallVector<Value> bindOperands;
            bindOperands.push_back(contValue);
            bindOperands.append(callOp.getOperands().begin(),
                                callOp.getOperands().end());

            OperationState bindState(callOp.getLoc(), "wami.cont.bind");
            bindState.addOperands(bindOperands);
            bindState.addAttribute("src_cont_type", *srcContRef);
            bindState.addAttribute("dst_cont_type", *dstContRef);
            bindState.addTypes(dstContType);
            Operation *bindOp = b.create(bindState);
            contValue = bindOp->getResult(0);
            boundContRef = *dstContRef;
          }

          SmallVector<Type> expectedResumeResults(implType.getResults().begin(),
                                                  implType.getResults().end());
          handleBindings[callOp.getResult(0)] =
              HandleBinding{callOp.getResult(0),
                            contValue,
                            boundContRef,
                            expectedResumeArgs,
                            expectedResumeResults,
                            intrinsic->suffix,
                            callOp,
                            false};
          continue;
        }
        case CoroIntrinsicKind::Resume: {
          if (callOp.getNumOperands() < 1) {
            callOp.emitError("coro.resume.* requires a handle operand");
            passFailed = true;
            break;
          }

          auto bindingIt = handleBindings.find(callOp.getOperand(0));
          if (bindingIt == handleBindings.end()) {
            callOp.emitError("resume handle must come directly from "
                             "coro.spawn.* in the same "
                             "function");
            passFailed = true;
            break;
          }

          HandleBinding &binding = bindingIt->second;
          if (binding.consumed) {
            callOp.emitError(
                "multiple coro.resume.* calls for one handle are not supported "
                "by coro-to-wami yet");
            passFailed = true;
            break;
          }

          SmallVector<Type> resumeArgTypes(callOp.getOperandTypes().begin() + 1,
                                           callOp.getOperandTypes().end());
          if (!typeSequencesEqual(resumeArgTypes, binding.expectedResumeArgs)) {
            callOp.emitError(
                "resume argument types do not match continuation expectation");
            passFailed = true;
            break;
          }
          if (!typeSequencesEqual(callOp.getResultTypes(),
                                  binding.expectedResumeResults)) {
            callOp.emitError(
                "resume result types do not match continuation result types");
            passFailed = true;
            break;
          }

          SmallVector<Value> resumeOperands;
          resumeOperands.push_back(binding.contValue);
          resumeOperands.append(callOp.getOperands().begin() + 1,
                                callOp.getOperands().end());

          OperationState resumeState(callOp.getLoc(), "wami.resume");
          resumeState.addOperands(resumeOperands);
          resumeState.addAttribute("cont_type", binding.contType);
          resumeState.addAttribute("handlers", ArrayAttr::get(ctx, {}));
          resumeState.addTypes(callOp.getResultTypes());
          Operation *resumeOp = b.create(resumeState);

          callOp.replaceAllUsesWith(resumeOp->getResults());
          eraseLater.push_back(callOp);
          binding.consumed = true;
          continue;
        }
        case CoroIntrinsicKind::Yield: {
          auto tagType = FunctionType::get(ctx, callOp.getOperandTypes(),
                                           callOp.getResultTypes());
          std::string tagSym =
              "coro_tag_" + sanitizeSymbolPiece(intrinsic->suffix);
          FailureOr<FlatSymbolRefAttr> tagRef =
              ensureWamiTag(module, callOp, tagSym, tagType, moduleBuilder);
          if (failed(tagRef)) {
            passFailed = true;
            break;
          }

          OperationState suspendState(callOp.getLoc(), "wami.suspend");
          suspendState.addOperands(callOp.getOperands());
          suspendState.addAttribute("tag", *tagRef);
          suspendState.addTypes(callOp.getResultTypes());
          Operation *suspendOp = b.create(suspendState);
          callOp.replaceAllUsesWith(suspendOp->getResults());
          eraseLater.push_back(callOp);
          continue;
        }
        case CoroIntrinsicKind::Cancel:
          eraseLater.push_back(callOp);
          continue;
        case CoroIntrinsicKind::IsDone:
          callOp.emitError(
              "coro.is_done.* lowering to WAMI is not implemented yet");
          passFailed = true;
          break;
        }
      }

      if (passFailed)
        break;

      for (auto &entry : handleBindings) {
        HandleBinding &binding = entry.second;
        bool hasLiveUse = false;
        for (OpOperand &use : binding.handleValue.getUses()) {
          if (!llvm::is_contained(eraseLater, use.getOwner())) {
            hasLiveUse = true;
            break;
          }
        }
        if (hasLiveUse) {
          binding.spawnOp->emitError(
              "spawn handle escapes coro-to-wami supported pattern");
          passFailed = true;
          break;
        }
        eraseLater.push_back(binding.spawnOp);
      }

      if (passFailed)
        break;

      for (Operation *op : eraseLater) {
        if (op && op->getBlock())
          op->erase();
      }
    }

    if (passFailed) {
      signalPassFailure();
      return;
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
          SmallVector<Value> args(callOp.getOperands().begin() + 1,
                                  callOp.getOperands().end());
          std::string implSym = buildImplSymbolForKind(intrinsic->suffix);
          auto direct = func::CallOp::create(b, callOp.getLoc(), implSym,
                                             callOp.getResultTypes(), args);
          callOp.replaceAllUsesWith(direct.getResults());
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
