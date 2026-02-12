//===- WasmStackOps.cpp - WasmStack dialect operations ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wasmstack/WasmStackOps.h"
#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::wasmstack;

// Include generated interface implementations
#include "wasmstack/WasmStackInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "wasmstack/WasmStackOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse: @name : (argTypes) -> (resultTypes) [export("name")] { body }
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function type
  FunctionType fnType;
  if (parser.parseColonType(fnType))
    return failure();

  result.addAttribute(FuncOp::getFuncTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  // Parse optional export attribute
  if (succeeded(parser.parseOptionalKeyword("export"))) {
    StringAttr exportName;
    if (parser.parseLParen() || parser.parseAttribute(exportName) ||
        parser.parseRParen())
      return failure();
    result.addAttribute("export_name", exportName);
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the body region
  auto *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  // Ensure single block
  if (body->empty())
    body->emplaceBlock();

  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << " : " << getFuncType();

  if (auto exportName = getExportNameAttr()) {
    p << " export(" << exportName << ")";
  }

  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getFuncTypeAttrName(), getSymNameAttrName(), "export_name"});

  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

LogicalResult FuncOp::verify() {
  // Verify the function body is not empty (has at least one block)
  if (getBody().empty())
    return emitOpError("expected non-empty function body");

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

/// Helper to check if a type is a valid WebAssembly type.
static bool isValidWasmType(Type type) {
  // Value types: i32, i64, f32, f64
  if (type.isSignlessInteger(32) || type.isSignlessInteger(64))
    return true;
  if (isa<Float32Type, Float64Type>(type))
    return true;
  // Reference types: funcref, contref, externref
  if (isa<FuncRefType, ContRefType, ExternRefType>(type))
    return true;
  return false;
}

LogicalResult GlobalOp::verify() {
  Type globalType = getType();
  if (!isValidWasmType(globalType)) {
    return emitOpError("global type must be a valid WebAssembly type "
                       "(i32, i64, f32, f64, or reference type), but got ")
           << globalType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BlockOp
//===----------------------------------------------------------------------===//

LogicalResult BlockOp::verify() {
  // Verify the body is not empty
  if (getBody().empty())
    return emitOpError("expected non-empty block body");
  return success();
}

StringRef BlockOp::getLabelName() { return getLabel(); }

// isLoop() has default implementation (false) in interface

TypeRange BlockOp::getBranchResultTypes() {
  // TODO(Phase 4): Implement properly when stack verification is added.
  // Challenge: TypeArrayAttr stores ArrayAttr of TypeAttr, but TypeRange
  // needs a view over Type values. Converting requires caching or
  // restructuring how result types are stored.
  // For now, return empty - stack verification not yet implemented.
  return TypeRange();
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

LogicalResult LoopOp::verify() {
  // Verify the body is not empty
  if (getBody().empty())
    return emitOpError("expected non-empty loop body");
  return success();
}

StringRef LoopOp::getLabelName() { return getLabel(); }

bool LoopOp::isLoop() { return true; }

TypeRange LoopOp::getBranchResultTypes() {
  // Loop branches restart from beginning, so no values passed
  return TypeRange();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

StringRef IfOp::getLabelName() {
  if (auto label = getLabel())
    return *label;
  return {};
}

TypeRange IfOp::getBranchResultTypes() {
  // TODO(Phase 4): Implement properly when stack verification is added.
  // See BlockOp::getBranchResultTypes() for details.
  return TypeRange();
}

//===----------------------------------------------------------------------===//
// Branch Target Verification (shared by BrOp, BrIfOp)
//===----------------------------------------------------------------------===//

/// Verifies that a branch target label exists in an ancestor control frame.
/// Returns success if found, failure with appropriate error otherwise.
static LogicalResult verifyBranchTarget(Operation *op, StringRef targetName) {
  Operation *parent = op->getParentOp();

  while (parent) {
    if (auto block = dyn_cast<BlockOp>(parent)) {
      if (block.getLabel() == targetName)
        return success();
    } else if (auto loop = dyn_cast<LoopOp>(parent)) {
      if (loop.getLabel() == targetName)
        return success();
    } else if (auto ifOp = dyn_cast<IfOp>(parent)) {
      auto label = ifOp.getLabel();
      if (label.has_value() && label.value() == targetName)
        return success();
    }
    if (isa<FuncOp>(parent)) {
      // Reached function boundary without finding target
      break;
    }
    parent = parent->getParentOp();
  }

  return op->emitOpError("branch target '")
         << targetName << "' not found in enclosing block/loop/if";
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

LogicalResult BrOp::verify() {
  return verifyBranchTarget(getOperation(), getTargetAttr().getValue());
}

//===----------------------------------------------------------------------===//
// BrIfOp
//===----------------------------------------------------------------------===//

LogicalResult BrIfOp::verify() {
  return verifyBranchTarget(getOperation(), getTargetAttr().getValue());
}

//===----------------------------------------------------------------------===//
// ResumeOp
//===----------------------------------------------------------------------===//

ParseResult ResumeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse continuation type reference
  FlatSymbolRefAttr contType;
  if (parser.parseAttribute(contType))
    return failure();
  result.addAttribute("cont_type", contType);

  // Parse optional handlers: (tag -> label, ...)
  SmallVector<Attribute> handlers;
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      FlatSymbolRefAttr tag;
      FlatSymbolRefAttr label;
      if (parser.parseAttribute(tag) || parser.parseArrow() ||
          parser.parseAttribute(label))
        return failure();

      SmallVector<Attribute> pair = {tag, label};
      handlers.push_back(ArrayAttr::get(parser.getContext(), pair));
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return failure();
  }
  result.addAttribute("handlers",
                      ArrayAttr::get(parser.getContext(), handlers));

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void ResumeOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getContTypeAttr().getValue());

  auto handlers = getHandlers();
  if (!handlers.empty()) {
    p << " (";
    llvm::interleaveComma(handlers, p, [&](Attribute handler) {
      auto pair = cast<ArrayAttr>(handler);
      p.printSymbolName(cast<FlatSymbolRefAttr>(pair[0]).getValue());
      p << " -> ";
      p.printSymbolName(cast<FlatSymbolRefAttr>(pair[1]).getValue());
    });
    p << ")";
  }

  p.printOptionalAttrDict((*this)->getAttrs(), {"cont_type", "handlers"});
}

LogicalResult ResumeOp::verify() {
  // Verify handler format
  for (auto handler : getHandlers()) {
    auto pair = dyn_cast<ArrayAttr>(handler);
    if (!pair || pair.size() != 2)
      return emitOpError("invalid handler format");
    if (!isa<FlatSymbolRefAttr>(pair[0]) || !isa<FlatSymbolRefAttr>(pair[1]))
      return emitOpError("handler must be (tag -> label) pair");
  }
  return success();
}
