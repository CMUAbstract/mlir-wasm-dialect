//===- SsaWasmTypes.cpp - SsaWasm dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SsaWasm/SsaWasmTypes.h"

#include "SsaWasm/SsaWasmDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::ssawasm;

#define GET_TYPEDEF_CLASSES
#include "SsaWasm/SsaWasmTypes.cpp.inc"

void SsaWasmDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SsaWasm/SsaWasmTypes.cpp.inc"
      >();
}

void SsaWasmIntegerType::print(mlir::AsmPrinter &printer) const {
  printer << "i" << getBitWidth();
}

mlir::LogicalResult SsaWasmIntegerType::parse(mlir::AsmParser &parser) {
  // Read the type name (e.g., "f32" or "f64")
  StringRef typeName;
  if (parser.parseKeyword(&typeName))
    return failure();

  // Expect the type name to start with 'i'
  if (!typeName.startswith("i"))
    return parser.emitError(parser.getNameLoc(),
                            "expected type name starting with 'i'");

  // Extract the numeric portion after 'f'
  auto bitWidthStr = typeName.drop_front();
  unsigned bitWidth;
  if (bitWidthStr.getAsInteger(10, bitWidth))
    return parser.emitError(parser.getNameLoc(), "failed to parse bit width");

  // Set the bitWidth parameter
  setBitWidth(bitWidth);
  return success();
}

void SsaWasmFloatType::print(mlir::AsmPrinter &printer) const {
  printer << "f" << getBitWidth();
}

mlir::LogicalResult SsaWasmFloatType::parse(mlir::AsmParser &parser) {
  // Read the type name (e.g., "f32" or "f64")
  StringRef typeName;
  if (parser.parseKeyword(&typeName))
    return failure();

  // Expect the type name to start with 'f'
  if (!typeName.startswith("f"))
    return parser.emitError(parser.getNameLoc(),
                            "expected type name starting with 'f'");

  // Extract the numeric portion after 'f'
  auto bitWidthStr = typeName.drop_front();
  unsigned bitWidth;
  if (bitWidthStr.getAsInteger(10, bitWidth))
    return parser.emitError(parser.getNameLoc(), "failed to parse bit width");

  // Set the bitWidth parameter
  setBitWidth(bitWidth);
  return success();
}