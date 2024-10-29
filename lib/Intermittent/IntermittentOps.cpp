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
  StringAttr symNameAttr;
  if (parser.parseSymbolName(symNameAttr, SymbolTable::getSymbolAttrName(),
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
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{SymbolTable::getSymbolAttrName()});
  p << ' ';
  p.printRegion(getBody());
}

} // namespace mlir::intermittent
