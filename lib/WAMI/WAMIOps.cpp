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

static LogicalResult verifyContValueType(Operation *op, Type contValueType,
                                         FlatSymbolRefAttr expectedSym,
                                         StringRef context) {
  auto contTy = dyn_cast<ContType>(contValueType);
  if (!contTy) {
    return op->emitError(context)
           << ": expected operand of !wami.cont<...> type, got "
           << contValueType;
  }

  if (contTy.getTypeName() != expectedSym) {
    return op->emitError(context) << ": continuation value type " << contTy
                                  << " does not match symbol " << expectedSym;
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
  return success();
}

LogicalResult ContNewOp::verify() {
  auto contTypeRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("cont_type");
  if (!contTypeRef)
    return emitOpError("missing 'cont_type' attribute");

  if (failed(resolveContSignature(*this, contTypeRef, "cont.new")))
    return failure();

  auto resultContTy = dyn_cast<ContType>(getResult().getType());
  if (!resultContTy)
    return emitOpError("result must be !wami.cont<...>");

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
                                 "cont.bind")))
    return failure();

  auto resultContTy = dyn_cast<ContType>(getResult().getType());
  if (!resultContTy)
    return emitOpError("result must be !wami.cont<...>");
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
  auto tagArray = (*this)->getAttrOfType<ArrayAttr>("handler_tags");
  if (!contTypeRef || !tagArray)
    return emitOpError("missing cont_type or handler_tags attribute");

  FailureOr<FunctionType> contSig =
      resolveContSignature(*this, contTypeRef, "resume");
  if (failed(contSig))
    return failure();

  ValueRange operands = (*this)->getOperands();
  if (operands.empty())
    return emitOpError("expected continuation operand");

  if (failed(verifyContValueType(*this, operands.front().getType(), contTypeRef,
                                 "resume")))
    return failure();

  ValueRange args = operands.drop_front();
  if (failed(verifyTypeSequence("resume argument types", args.getTypes(),
                                contSig->getInputs(), *this)))
    return failure();

  if (failed(verifyTypeSequence("resume result types", getResultTypes(),
                                contSig->getResults(), *this)))
    return failure();

  llvm::StringSet<> seenTags;
  SmallVector<FlatSymbolRefAttr> handlerTags;
  handlerTags.reserve(tagArray.size());
  for (Attribute attr : tagArray) {
    auto tagRef = dyn_cast<FlatSymbolRefAttr>(attr);
    if (!tagRef)
      return emitOpError("handler_tags must contain symbol references");
    if (!seenTags.insert(tagRef.getValue()).second)
      return emitOpError("duplicate handler tag ") << tagRef;
    handlerTags.push_back(tagRef);
  }

  Region &handlers = (*this)->getRegion(0);
  if (handlers.getBlocks().size() != handlerTags.size()) {
    return emitOpError("handler region block count must match handler_tags "
                       "count");
  }

  for (auto [idx, block] : llvm::enumerate(handlers.getBlocks())) {
    FailureOr<FunctionType> tagSig =
        resolveTagSignature(*this, handlerTags[idx], "resume handler");
    if (failed(tagSig))
      return failure();

    if (failed(verifyTypeSequence("handler block argument types",
                                  block.getArgumentTypes(), tagSig->getInputs(),
                                  *this)))
      return failure();

    Operation *terminator = block.getTerminator();
    if (!terminator)
      return emitOpError("handler block must end with wami.handler.yield");

    auto yieldOp = dyn_cast<HandlerYieldOp>(terminator);
    if (!yieldOp)
      return emitOpError(
          "handler block must terminate with wami.handler.yield");

    if (failed(verifyTypeSequence("handler yield types",
                                  yieldOp.getValues().getTypes(),
                                  contSig->getResults(), *this)))
      return failure();
  }

  return success();
}

LogicalResult ResumeThrowOp::verify() {
  auto contTypeRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("cont_type");
  auto tagArray = (*this)->getAttrOfType<ArrayAttr>("handler_tags");
  if (!contTypeRef || !tagArray)
    return emitOpError("missing cont_type or handler_tags attribute");

  FailureOr<FunctionType> contSig =
      resolveContSignature(*this, contTypeRef, "resume_throw");
  if (failed(contSig))
    return failure();

  ValueRange operands = (*this)->getOperands();
  if (operands.empty())
    return emitOpError("expected continuation operand");

  if (failed(verifyContValueType(*this, operands.front().getType(), contTypeRef,
                                 "resume_throw")))
    return failure();

  ValueRange args = operands.drop_front();
  if (failed(verifyTypeSequence("resume_throw argument types", args.getTypes(),
                                contSig->getInputs(), *this)))
    return failure();

  if ((*this)->getNumResults() != 0)
    return emitOpError("resume_throw must not produce normal results");

  llvm::StringSet<> seenTags;
  SmallVector<FlatSymbolRefAttr> handlerTags;
  handlerTags.reserve(tagArray.size());
  for (Attribute attr : tagArray) {
    auto tagRef = dyn_cast<FlatSymbolRefAttr>(attr);
    if (!tagRef)
      return emitOpError("handler_tags must contain symbol references");
    if (!seenTags.insert(tagRef.getValue()).second)
      return emitOpError("duplicate handler tag ") << tagRef;
    handlerTags.push_back(tagRef);
  }

  Region &handlers = (*this)->getRegion(0);
  if (handlers.getBlocks().size() != handlerTags.size()) {
    return emitOpError("handler region block count must match handler_tags "
                       "count");
  }

  for (auto [idx, block] : llvm::enumerate(handlers.getBlocks())) {
    FailureOr<FunctionType> tagSig =
        resolveTagSignature(*this, handlerTags[idx], "resume_throw handler");
    if (failed(tagSig))
      return failure();

    if (failed(verifyTypeSequence("handler block argument types",
                                  block.getArgumentTypes(), tagSig->getInputs(),
                                  *this)))
      return failure();

    Operation *terminator = block.getTerminator();
    if (!terminator)
      return emitOpError("handler block must have a terminator");
    if (isa<HandlerYieldOp>(terminator)) {
      return emitOpError(
          "resume_throw handlers must not terminate with wami.handler.yield");
    }
  }

  return success();
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
