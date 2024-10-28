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

// void NonVolatileNewOp::print(OpAsmPrinter &p) { p << " : " << getType(); }
//
// ParseResult NonVolatileNewOp::parse(OpAsmParser &parser,
//                                     OperationState &result) {
//   Type type;
//   if (parser.parseColonType(type))
//     return failure();
//   result.addTypes(type);
//   return success();
// }

// void NonVolatileLoadOp::print(OpAsmPrinter &p) {
//   p << ' ' << getVar() << " : " << getVar().getType();
// }

// ParseResult NonVolatileLoadOp::parse(OpAsmParser &parser,
//                                      OperationState &result) {
//   OpAsmParser::OperandType varOperand;
//   Type varType;
//
//   // Parse the operand and its type
//   if (parser.parseOperand(varOperand) || parser.parseColonType(varType))
//     return failure();
//
//   // Resolve the operand
//   if (parser.resolveOperand(varOperand, varType, result.operands))
//     return failure();
//
//   // Ensure the operand is of NonVolatileType
//   auto nonVolatileType = varType.dyn_cast<NonVolatileType>();
//   if (!nonVolatileType) {
//     return parser.emitError(parser.getNameLoc(),
//                             "expected operand of type NonVolatileType");
//   }
//
//   // The result type is the element type of the NonVolatileType
//   result.addTypes(nonVolatileType.getInnerType());
//
//   return success();
// }

// void NonVolatileStoreOp::print(OpAsmPrinter &p) {
//   p << ' ' << getVar() << ", " << getValue() << " : " << getVar().getType()
//     << ", " << getValue().getType();
// }
//
// ParseResult NonVolatileStoreOp::parse(OpAsmParser &parser,
//                                       OperationState &result) {
//   OpAsmParser::OperandType varOperand, valueOperand;
//   Type type;
//   auto context = parser.getBuilder().getContext();
//
//   // Parse operands and their types
//   if (parser.parseOperand(varOperand) || parser.parseComma() ||
//       parser.parseOperand(valueOperand) || parser.parseColonType(varType) ||
//       parser.parseComma() || parser.parseType(type))
//     return failure();
//
//   // Resolve operands
//   if (parser.resolveOperand(varOperand, NonVolatileType::get(context, type),
//                             result.operands) ||
//       parser.resolveOperand(valueOperand, type, result.operands))
//     return failure();
//
//   return success();
// }

} // namespace mlir::intermittent
