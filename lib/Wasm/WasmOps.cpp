//===- WasmOps.cpp - Wasm dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/FunctionImplementation.h"

#include "Wasm/WasmDialect.h"
#include "Wasm/WasmOps.h"
#include "Wasm/WasmTypes.h"

#define GET_OP_CLASSES
#include "Wasm/WasmOps.cpp.inc"

namespace mlir::wasm {

llvm::LogicalResult ConstantOp::verify() {
  // TODO: Value must be either of i32, i64, f32, or f64 attribute.
  if (!llvm::isa<IntegerAttr, FloatAttr>(getValue())) {
    return emitOpError(
        "value must be either of i32, i64, f32, or f64 attribute");
  }
  return success();
}

void WasmFuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, mlir::FunctionType type,
                       llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

ParseResult WasmFuncOp::parse(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void WasmFuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult
TempGetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: check if the symbol is a global variable
  return success();
}

void BlockOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    llvm::StringRef name,
                    llvm::ArrayRef<mlir::Attribute> return_types) {
  state.addAttribute("name", builder.getStringAttr(name));
  state.addAttribute("return_types", builder.getArrayAttr(return_types));
  state.addRegion();
}

void IfElseOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     llvm::ArrayRef<mlir::Attribute> return_types) {
  state.addAttribute("return_types", builder.getArrayAttr(return_types));
  state.addRegion();
  state.addRegion();
}

void LoopOpDeprecated::initialize(OpBuilder &builder) {
  Region &body = getBody();
  auto *entryBlock = builder.createBlock(&body);
  auto *mainBlock = builder.createBlock(&body);

  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(entryBlock);
  builder.create<BranchOpDeprecated>(getLoc(), mainBlock);
  builder.restoreInsertionPoint(ip);
}
Block *LoopOpDeprecated::getEntryBlock() { return &getRegion().front(); }

Block *LoopOpDeprecated::getMainBlock() {
  Region &region = getRegion();
  auto it = region.begin();
  ++it; // Skip the entry block.
  return &*it;
}

llvm::LogicalResult
LoopOpDeprecated::inlineRegionToMainBlock(Region &sourceRegion,
                                          PatternRewriter &rewriter) {
  // Ensure the source region has exactly one block for simplicity.
  if (sourceRegion.getBlocks().size() != 1) {
    return emitOpError(
        "inlineRegionToLoopHeader: source region must have exactly one block");
  }

  Block &sourceBlock = sourceRegion.front();

  // Get the loop header block where we will inline the operations.
  Block *mainBlock = getMainBlock();
  if (!mainBlock) {
    return emitOpError("inlineRegionToLoopHeader: loop header block not found");
  }

  // Prepare a mapping from source values to target values.
  IRMapping mapping;

  // If the source block has arguments, map them to the loop header's block
  // arguments.
  if (!sourceBlock.getArguments().empty()) {
    // For simplicity, assume the number of arguments matches.
    if (sourceBlock.getArguments().size() != mainBlock->getArguments().size()) {
      return emitOpError(
          "inlineRegionToLoopHeader: block argument count mismatch");
    }

    for (auto it :
         llvm::zip(sourceBlock.getArguments(), mainBlock->getArguments())) {
      Value sourceArg = std::get<0>(it);
      Value targetArg = std::get<1>(it);
      mapping.map(sourceArg, targetArg);
    }
  }

  // Clone each operation from the source block into the loop header block.
  for (Operation &op : sourceBlock) {
    // Clone the operation with the current mapping.
    Operation *clonedOp = rewriter.clone(op, mapping);
    if (!clonedOp) {
      return emitOpError("inlineRegionToLoopHeader: failed to clone operation");
    }

    // If the cloned operation has results, map them.
    if (!clonedOp->getResults().empty()) {
      for (auto it : llvm::zip(op.getResults(), clonedOp->getResults())) {
        Value sourceResult = std::get<0>(it);
        Value targetResult = std::get<1>(it);
        mapping.map(sourceResult, targetResult);
      }
    }
  }

  return success();
}

} // namespace mlir::wasm
