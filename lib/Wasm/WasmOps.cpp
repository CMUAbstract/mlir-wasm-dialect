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

void TempLocalOp::build(OpBuilder &builder, OperationState &state,
                        mlir::Type inner) {
  auto context = inner.getContext();
  auto localType = mlir::wasm::LocalType::get(context, inner);
  state.addTypes(localType);
  state.addAttribute("type", mlir::TypeAttr::get(inner));
}

ParseResult parseLocalOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand local;
  Type localInnerType;

  if (parser.parseOperand(local) || parser.parseColonType(localInnerType))
    return failure();

  auto localType = LocalType::get(parser.getContext(), localInnerType);
  if (parser.resolveOperand(local, localType, result.operands))
    return failure();

  return success();
}

ParseResult TempLocalGetOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLocalOp(parser, result);
}

void TempLocalGetOp::print(OpAsmPrinter &p) {
  p << " " << getLocal();
  p << " : " << getLocal().getType().getInner();
}

ParseResult TempLocalSetOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLocalOp(parser, result);
}

void TempLocalSetOp::print(OpAsmPrinter &p) {
  p << " " << getLocal();
  p << " : " << getLocal().getType().getInner();
}

void BlockLoopOp::build(OpBuilder &builder, OperationState &state) {
  state.addRegion();
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

void TempGlobalOp::build(OpBuilder &builder, OperationState &state,
                         bool isMutable, mlir::Type inner) {
  auto context = inner.getContext();
  auto globalType = mlir::wasm::GlobalType::get(context, inner);
  state.addTypes(globalType);
  state.addAttribute("is_mutable", builder.getBoolAttr(isMutable));
  state.addAttribute("type", mlir::TypeAttr::get(inner));
}

ParseResult parseGlobalOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand global;
  Type globalInnerType;

  if (parser.parseOperand(global) || parser.parseColonType(globalInnerType))
    return failure();

  auto globalType = GlobalType::get(parser.getContext(), globalInnerType);
  if (parser.resolveOperand(global, globalType, result.operands))
    return failure();

  return success();
}

ParseResult TempGlobalGetOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  return parseGlobalOp(parser, result);
}

void TempGlobalGetOp::print(OpAsmPrinter &p) {
  p << " " << getGlobal();
  p << " : " << getGlobal().getType().getInner();
}

ParseResult TempGlobalSetOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  return parseGlobalOp(parser, result);
}

void TempGlobalSetOp::print(OpAsmPrinter &p) {
  p << " " << getGlobal();
  p << " : " << getGlobal().getType().getInner();
}

ParseResult TempGlobalIndexOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseGlobalOp(parser, result);
}

void TempGlobalIndexOp::print(OpAsmPrinter &p) {
  p << " " << getGlobal();
  p << " : " << getGlobal().getType().getInner();
}

} // namespace mlir::wasm
