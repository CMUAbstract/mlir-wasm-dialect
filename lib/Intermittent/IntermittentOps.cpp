//===- IntermittentOps.cpp - Intermittent dialect ops ---------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Intermittent/IntermittentOps.h"
#include "Intermittent/IntermittentDialect.h"
#include "Intermittent/IntermittentTypes.h"

#define GET_OP_CLASSES
#include "Intermittent/IntermittentOps.cpp.inc"

namespace mlir::intermittent {

ParseResult IdempotentTaskOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Parse the symbol name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  return success();
}

void IdempotentTaskOp::print(OpAsmPrinter &p) {
  p << " " << '@' << getSymName();
  p << ' ';
  p.printRegion(getBody());
}

ParseResult NonVolatileNewOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  if (parser.parseLParen() || parser.parseRParen() || parser.parseColon())
    return failure();

  NonVolatileType nonVolatileType;
  if (parser.parseCustomTypeWithFallback(nonVolatileType))
    return failure();
  result.addTypes({nonVolatileType});
  auto elementType = nonVolatileType.getElementType();

  result.addAttribute("inner", TypeAttr::get(elementType));
  return success();
}

void NonVolatileNewOp::print(::mlir::OpAsmPrinter &p) {
  p << "() : ";
  NonVolatileType type = NonVolatileType::get(getContext(), getInner());
  p.printType(type);
}

} // namespace mlir::intermittent
