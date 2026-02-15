//===- WAMIOps.cpp - WAMI dialect operations --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements operations for the WAMI dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/WAMIOps.h"
#include "WAMI/WAMIAttrs.h"
#include "WAMI/WAMIDialect.h"
#include "WAMI/WAMITypes.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#define GET_OP_CLASSES
#include "WAMI/WAMIOps.cpp.inc"

using namespace mlir;
using namespace mlir::wami;

namespace {

static LogicalResult verifyTypeSequence(StringRef what, TypeRange actual,
                                        TypeRange expected, Operation *op) {
  if (actual.size() != expected.size()) {
    return op->emitError(what)
           << " arity mismatch: expected " << expected.size() << " but got "
           << actual.size();
  }

  for (auto [idx, pair] : llvm::enumerate(llvm::zip(actual, expected))) {
    Type actualType = std::get<0>(pair);
    Type expectedType = std::get<1>(pair);
    if (actualType != expectedType) {
      return op->emitError(what)
             << " type mismatch at index " << idx << ": expected "
             << expectedType << " but got " << actualType;
    }
  }

  return success();
}

static FailureOr<FunctionType>
resolveTypeFuncSymbol(Operation *op, FlatSymbolRefAttr ref, StringRef context) {
  auto typeFunc = SymbolTable::lookupNearestSymbolFrom<TypeFuncOp>(op, ref);
  if (!typeFunc) {
    op->emitError(context) << ": unknown wami.type.func symbol " << ref;
    return failure();
  }

  auto typeAttr = typeFunc->getAttrOfType<TypeAttr>("type");
  if (!typeAttr) {
    op->emitError(context) << ": wami.type.func " << ref
                           << " missing 'type' attribute";
    return failure();
  }

  auto fnType = dyn_cast<FunctionType>(typeAttr.getValue());
  if (!fnType) {
    op->emitError(context) << ": expected function type for " << ref;
    return failure();
  }

  return fnType;
}

static FailureOr<FunctionType>
resolveContSignature(Operation *op, FlatSymbolRefAttr ref, StringRef context) {
  auto contType = SymbolTable::lookupNearestSymbolFrom<TypeContOp>(op, ref);
  if (!contType) {
    op->emitError(context) << ": unknown wami.type.cont symbol " << ref;
    return failure();
  }

  auto funcTypeRef = contType->getAttrOfType<FlatSymbolRefAttr>("func_type");
  if (!funcTypeRef) {
    op->emitError(context) << ": wami.type.cont " << ref
                           << " missing 'func_type' attribute";
    return failure();
  }

  return resolveTypeFuncSymbol(op, funcTypeRef, context);
}

static FailureOr<FunctionType>
resolveTagSignature(Operation *op, FlatSymbolRefAttr ref, StringRef context) {
  auto tagOp = SymbolTable::lookupNearestSymbolFrom<TagOp>(op, ref);
  if (!tagOp) {
    op->emitError(context) << ": unknown wami.tag symbol " << ref;
    return failure();
  }

  auto typeAttr = tagOp->getAttrOfType<TypeAttr>("type");
  if (!typeAttr) {
    op->emitError(context) << ": wami.tag " << ref
                           << " missing 'type' attribute";
    return failure();
  }

  auto fnType = dyn_cast<FunctionType>(typeAttr.getValue());
  if (!fnType) {
    op->emitError(context) << ": expected function type for tag " << ref;
    return failure();
  }
  return fnType;
}

static FailureOr<FunctionType>
resolveFuncRefSignature(Operation *op, FuncRefType funcRef, StringRef context) {
  FlatSymbolRefAttr funcRefSym = funcRef.getFuncName();
  Operation *funcSym = SymbolTable::lookupNearestSymbolFrom(op, funcRefSym);
  if (!funcSym) {
    op->emitError(context) << ": unknown function symbol " << funcRefSym;
    return failure();
  }

  if (auto funcOp = dyn_cast<wasmssa::FuncOp>(funcSym))
    return funcOp.getFunctionType();
  if (auto importOp = dyn_cast<wasmssa::FuncImportOp>(funcSym))
    return importOp.getType();

  op->emitError(context) << ": symbol " << funcRefSym
                         << " is not a wasmssa.func or wasmssa.import_func";
  return failure();
}

static LogicalResult verifyContValueType(Operation *op, Type contValueType,
                                         FlatSymbolRefAttr expectedSym,
                                         StringRef context,
                                         bool requireNonNull = false) {
  auto contTy = dyn_cast<ContType>(contValueType);
  if (!contTy) {
    return op->emitError(context)
           << ": expected operand of !wami.cont<...> type, got "
           << contValueType;
  }

  if (requireNonNull && contTy.getNullable())
    return op->emitError(context) << " requires non-null continuation";

  if (contTy.getTypeName() != expectedSym) {
    return op->emitError(context) << ": continuation value type " << contTy
                                  << " does not match symbol " << expectedSym;
  }
  return success();
}

static unsigned countEnclosingStructuredLabels(Operation *op) {
  unsigned depth = 0;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<wasmssa::BlockOp, wasmssa::LoopOp, wasmssa::IfOp>(parent))
      ++depth;
  }
  return depth;
}

static LogicalResult verifyResumeHandlers(Operation *op, ArrayAttr handlers,
                                          StringRef context) {
  llvm::StringSet<> seenTags;
  unsigned labelDepth = countEnclosingStructuredLabels(op);

  for (Attribute handlerAttr : handlers) {
    if (auto onLabel = dyn_cast<OnLabelHandlerAttr>(handlerAttr)) {
      FlatSymbolRefAttr tagRef = onLabel.getTag();
      if (!seenTags.insert(tagRef.getValue()).second)
        return op->emitError("duplicate handler tag ") << tagRef;

      int64_t level = onLabel.getLevel();
      if (level < 0)
        return op->emitError("on_label level must be non-negative");
      if (static_cast<uint64_t>(level) >= labelDepth) {
        return op->emitError("on_label level ")
               << level << " exceeds enclosing structured label depth "
               << labelDepth;
      }

      if (failed(resolveTagSignature(op, tagRef, context)))
        return failure();
      continue;
    }

    if (auto onSwitch = dyn_cast<OnSwitchHandlerAttr>(handlerAttr)) {
      FlatSymbolRefAttr tagRef = onSwitch.getTag();
      if (!seenTags.insert(tagRef.getValue()).second)
        return op->emitError("duplicate handler tag ") << tagRef;
      FailureOr<FunctionType> tagSig = resolveTagSignature(op, tagRef, context);
      if (failed(tagSig))
        return failure();
      if (!tagSig->getInputs().empty())
        return op->emitError("on_switch handler tag must have empty inputs");
      continue;
    }

    return op->emitError(
        "handlers must contain #wami.on_label or #wami.on_switch attributes");
  }

  return success();
}

} // namespace

LogicalResult TypeContOp::verify() {
  auto funcTypeRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("func_type");
  if (!funcTypeRef)
    return emitOpError("missing 'func_type' attribute");
  return succeeded(resolveTypeFuncSymbol(*this, funcTypeRef, "type.cont"))
             ? success()
             : failure();
}

LogicalResult RefNullOp::verify() {
  Type resultType = getResult().getType();
  if (!isa<FuncRefType, ContType>(resultType)) {
    return emitOpError("ref.null result must be a WAMI reference type, got ")
           << resultType;
  }

  if (auto contTy = dyn_cast<ContType>(resultType);
      contTy && !contTy.getNullable()) {
    return emitOpError(
        "ref.null for continuation requires nullable continuation type");
  }

  return success();
}

LogicalResult ContNewOp::verify() {
  auto contTypeRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("cont_type");
  if (!contTypeRef)
    return emitOpError("missing 'cont_type' attribute");

  FailureOr<FunctionType> contSig =
      resolveContSignature(*this, contTypeRef, "cont.new");
  if (failed(contSig))
    return failure();

  auto funcRefTy = dyn_cast<FuncRefType>(getFunc().getType());
  if (!funcRefTy)
    return emitOpError("operand must be !wami.funcref<...>");

  FailureOr<FunctionType> funcSig =
      resolveFuncRefSignature(*this, funcRefTy, "cont.new");
  if (failed(funcSig))
    return failure();

  if (*funcSig != *contSig)
    return emitOpError("funcref signature does not match continuation type");

  auto resultContTy = dyn_cast<ContType>(getResult().getType());
  if (!resultContTy)
    return emitOpError("result must be !wami.cont<...>");
  if (resultContTy.getNullable())
    return emitOpError("cont.new result must be non-null continuation");

  if (resultContTy.getTypeName() != contTypeRef) {
    return emitOpError("result continuation type symbol ")
           << resultContTy.getTypeName()
           << " does not match cont_type attribute " << contTypeRef;
  }

  return success();
}

LogicalResult ContBindOp::verify() {
  auto srcRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("src_cont_type");
  auto dstRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("dst_cont_type");
  if (!srcRef || !dstRef)
    return emitOpError("missing src_cont_type or dst_cont_type attribute");

  FailureOr<FunctionType> srcSig =
      resolveContSignature(*this, srcRef, "cont.bind source");
  FailureOr<FunctionType> dstSig =
      resolveContSignature(*this, dstRef, "cont.bind destination");
  if (failed(srcSig) || failed(dstSig))
    return failure();

  ValueRange operands = (*this)->getOperands();
  if (operands.empty())
    return emitOpError("expected continuation operand");

  if (failed(verifyContValueType(*this, operands.front().getType(), srcRef,
                                 "cont.bind", /*requireNonNull=*/true)))
    return failure();

  auto resultContTy = dyn_cast<ContType>(getResult().getType());
  if (!resultContTy)
    return emitOpError("result must be !wami.cont<...>");
  if (resultContTy.getNullable())
    return emitOpError("cont.bind result must be non-null continuation");
  if (resultContTy.getTypeName() != dstRef) {
    return emitOpError("result continuation symbol ")
           << resultContTy.getTypeName() << " does not match dst_cont_type "
           << dstRef;
  }

  ValueRange boundArgs = operands.drop_front();
  ArrayRef<Type> srcInputs = srcSig->getInputs();
  ArrayRef<Type> dstInputs = dstSig->getInputs();

  if (srcInputs.size() != boundArgs.size() + dstInputs.size()) {
    return emitOpError("source continuation input arity must equal "
                       "bound_args + destination input arity");
  }

  for (auto [idx, arg] : llvm::enumerate(boundArgs)) {
    Type expectedType = srcInputs[idx];
    if (arg.getType() != expectedType) {
      return emitOpError("bound argument type mismatch at index ")
             << idx << ": expected " << expectedType << " but got "
             << arg.getType();
    }
  }

  for (auto [idx, dstTy] : llvm::enumerate(dstInputs)) {
    Type expectedType = srcInputs[boundArgs.size() + idx];
    if (dstTy != expectedType) {
      return emitOpError("destination continuation input type mismatch at "
                         "index ")
             << idx << ": expected " << expectedType << " but got " << dstTy;
    }
  }

  if (failed(verifyTypeSequence("continuation result types",
                                dstSig->getResults(), srcSig->getResults(),
                                *this)))
    return failure();

  return success();
}

LogicalResult SuspendOp::verify() {
  auto tagRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("tag");
  if (!tagRef)
    return emitOpError("missing 'tag' attribute");

  FailureOr<FunctionType> tagType =
      resolveTagSignature(*this, tagRef, "suspend");
  if (failed(tagType))
    return failure();

  ValueRange payload = (*this)->getOperands();
  if (failed(verifyTypeSequence("suspend payload types", payload.getTypes(),
                                tagType->getInputs(), *this)))
    return failure();

  return verifyTypeSequence("suspend result types", getResultTypes(),
                            tagType->getResults(), *this);
}

LogicalResult ResumeOp::verify() {
  auto contTypeRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("cont_type");
  auto handlerArray = (*this)->getAttrOfType<ArrayAttr>("handlers");
  if (!contTypeRef || !handlerArray)
    return emitOpError("missing cont_type or handlers attribute");

  FailureOr<FunctionType> contSig =
      resolveContSignature(*this, contTypeRef, "resume");
  if (failed(contSig))
    return failure();

  ValueRange operands = (*this)->getOperands();
  if (operands.empty())
    return emitOpError("expected continuation operand");

  if (failed(verifyContValueType(*this, operands.front().getType(), contTypeRef,
                                 "resume", /*requireNonNull=*/false)))
    return failure();

  ValueRange args = operands.drop_front();
  if (failed(verifyTypeSequence("resume argument types", args.getTypes(),
                                contSig->getInputs(), *this)))
    return failure();

  if (failed(verifyTypeSequence("resume result types", getResultTypes(),
                                contSig->getResults(), *this)))
    return failure();

  return verifyResumeHandlers(*this, handlerArray, "resume handler");
}

LogicalResult ResumeThrowOp::verify() {
  auto contTypeRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("cont_type");
  auto handlerArray = (*this)->getAttrOfType<ArrayAttr>("handlers");
  if (!contTypeRef || !handlerArray)
    return emitOpError("missing cont_type or handlers attribute");

  FailureOr<FunctionType> contSig =
      resolveContSignature(*this, contTypeRef, "resume_throw");
  if (failed(contSig))
    return failure();

  ValueRange operands = (*this)->getOperands();
  if (operands.empty())
    return emitOpError("expected continuation operand");

  if (failed(verifyContValueType(*this, operands.front().getType(), contTypeRef,
                                 "resume_throw", /*requireNonNull=*/false)))
    return failure();

  ValueRange args = operands.drop_front();
  if (failed(verifyTypeSequence("resume_throw argument types", args.getTypes(),
                                contSig->getInputs(), *this)))
    return failure();

  if ((*this)->getNumResults() != 0)
    return emitOpError("resume_throw must not produce normal results");

  return verifyResumeHandlers(*this, handlerArray, "resume_throw handler");
}

LogicalResult BarrierOp::verify() {
  Region &body = (*this)->getRegion(0);
  if (body.empty())
    return emitOpError("barrier body must have one block");

  Block &block = body.front();
  Operation *terminator = block.getTerminator();
  if (!terminator)
    return emitOpError("barrier body must terminate with wami.barrier.yield");

  auto yieldOp = dyn_cast<BarrierYieldOp>(terminator);
  if (!yieldOp)
    return emitOpError("barrier body must terminate with wami.barrier.yield");

  return verifyTypeSequence("barrier yield types",
                            yieldOp.getValues().getTypes(), getResultTypes(),
                            *this);
}
