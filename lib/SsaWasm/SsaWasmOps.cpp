//===- SsaWasmOps.cpp - SsaWasm dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "SsaWasm/SsaWasmDialect.h"
#include "SsaWasm/SsaWasmOps.h"
#include "SsaWasm/SsaWasmTypes.h"

#define GET_OP_CLASSES

namespace mlir::ssawasm {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns true if 'attr' is an integer constant that fits in either i32 or
/// i64.
static bool isInt32Or64(Attribute attr) {
  if (!attr)
    return false;
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return false;
  unsigned bitWidth = intAttr.getValue().getBitWidth();
  return (bitWidth == 32 || bitWidth == 64);
}

/// Returns true if 'attr' is a float constant of type f32 or f64.
static bool isF32OrF64(Attribute attr) {
  if (!attr)
    return false;
  auto floatAttr = dyn_cast<FloatAttr>(attr);
  if (!floatAttr)
    return false;
  Type t = floatAttr.getType();
  return (t.isF32() || t.isF64());
}

/// Adds two integer constants (32-bit or 64-bit).
static Attribute foldAddInteger(IntegerAttr lhs, IntegerAttr rhs,
                                MLIRContext *ctx) {
  APInt result = lhs.getValue() + rhs.getValue();
  // Use the same type as lhs or rhs (they match for your ops).
  return IntegerAttr::get(lhs.getType(), result);
}

/// Sub two integer constants (32-bit or 64-bit).
static Attribute foldSubInteger(IntegerAttr lhs, IntegerAttr rhs,
                                MLIRContext *ctx) {
  APInt result = lhs.getValue() - rhs.getValue();
  return IntegerAttr::get(lhs.getType(), result);
}

/// Mul two integer constants (32-bit or 64-bit).
static Attribute foldMulInteger(IntegerAttr lhs, IntegerAttr rhs,
                                MLIRContext *ctx) {
  APInt result = lhs.getValue() * rhs.getValue();
  return IntegerAttr::get(lhs.getType(), result);
}

/// Returns min of two integer constants (handles both i32, i64).
static Attribute foldMinInteger(IntegerAttr lhs, IntegerAttr rhs) {
  const APInt &l = lhs.getValue();
  const APInt &r = rhs.getValue();
  // For your "min" or "max" you have to decide if it's signed or unsigned.
  // If it's unsigned min, use 'ult', else if signed, use 'slt'.
  // For demonstration, let's pick "min" as a *signed* min:
  APInt result = l.slt(r) ? l : r;
  return IntegerAttr::get(lhs.getType(), result);
}

/// Returns max of two integer constants.
static Attribute foldMaxInteger(IntegerAttr lhs, IntegerAttr rhs) {
  const APInt &l = lhs.getValue();
  const APInt &r = rhs.getValue();
  // For demonstration, let's do signed max:
  APInt result = l.sgt(r) ? l : r;
  return IntegerAttr::get(lhs.getType(), result);
}

/// Fold float add, sub, mul, min, max, etc.
static Attribute foldAddFloat(FloatAttr lhs, FloatAttr rhs) {
  APFloat val = lhs.getValue();
  val.add(rhs.getValue(), APFloat::rmNearestTiesToEven);
  return FloatAttr::get(lhs.getType(), val);
}

static Attribute foldSubFloat(FloatAttr lhs, FloatAttr rhs) {
  APFloat val = lhs.getValue();
  val.subtract(rhs.getValue(), APFloat::rmNearestTiesToEven);
  return FloatAttr::get(lhs.getType(), val);
}

static Attribute foldMulFloat(FloatAttr lhs, FloatAttr rhs) {
  APFloat val = lhs.getValue();
  val.multiply(rhs.getValue(), APFloat::rmNearestTiesToEven);
  return FloatAttr::get(lhs.getType(), val);
}

/// For min or max, you can choose IEEE `minimum` / `maximum` or do a simpler
/// check. This is a simplified version using LLVM's APFloat helpers:
static Attribute foldMinFloat(FloatAttr lhs, FloatAttr rhs) {
  APFloat val = lhs.getValue();
  APFloat::cmpResult c = val.compare(rhs.getValue());
  // For demonstration: "minimum" in the IEEE sense:
  if (c == APFloat::cmpGreaterThan)
    val = rhs.getValue();
  return FloatAttr::get(lhs.getType(), val);
}

static Attribute foldMaxFloat(FloatAttr lhs, FloatAttr rhs) {
  APFloat val = lhs.getValue();
  APFloat::cmpResult c = val.compare(rhs.getValue());
  // For demonstration: "maximum" in the IEEE sense:
  if (c == APFloat::cmpLessThan)
    val = rhs.getValue();
  return FloatAttr::get(lhs.getType(), val);
}

/// If both attributes are constants of the same type, apply `combineFn` to
/// them. Return null if folding is impossible.
static Attribute
tryFoldBinaryOp(Attribute lhs, Attribute rhs,
                function_ref<Attribute(Attribute, Attribute)> combineFn) {
  if (!lhs || !rhs)
    return {};
  // The type must match for your dialect (assuming typed results).
  if (lhs.getTypeID() != rhs.getTypeID())
    return {};

  // Attempt integer fold:
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs))
      return combineFn(lhsInt, rhsInt);
  }

  // Attempt float fold:
  if (auto lhsF = dyn_cast<FloatAttr>(lhs)) {
    if (auto rhsF = dyn_cast<FloatAttr>(rhs))
      return combineFn(lhsF, rhsF);
  }

  // If none matched, folding not possible
  return {};
}

//===----------------------------------------------------------------------===//
// Folding for Each Op
//===----------------------------------------------------------------------===//

/// Example folder for `SsaWasm_ConstantOp`.
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  // The `ConstantLike` trait will usually fold to the `value` attribute
  // directly. For a custom dialect, it’s typical to just do:
  return getValue();
}

/// SsaWasm_AddOp folding:
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // add(x, 0) -> x
  // For integer 0 or float +0.0
  if (Attribute rhs = adaptor.getRhs()) {
    // For integer zero:
    if (isInt32Or64(rhs)) {
      auto iAttr = cast<IntegerAttr>(rhs);
      if (iAttr.getValue() == 0)
        return getLhs();
    }
    // For float zero:
    else if (isF32OrF64(rhs)) {
      auto fAttr = cast<FloatAttr>(rhs);
      if (fAttr.getValue().isZero())
        return getLhs();
    }
  }

  // If both operands constant, do a direct add:
  Attribute folded = tryFoldBinaryOp(
      adaptor.getLhs(), adaptor.getRhs(),
      [&](Attribute a, Attribute b) -> Attribute {
        if (isa<IntegerAttr>(a))
          return foldAddInteger(cast<IntegerAttr>(a), cast<IntegerAttr>(b),
                                getContext());
        else
          return foldAddFloat(cast<FloatAttr>(a), cast<FloatAttr>(b));
      });
  if (folded)
    return folded;

  // Couldn’t fold
  return {};
}

/// SsaWasm_SubOp folding:
OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  // sub(x, 0) -> x
  if (Attribute rhs = adaptor.getRhs()) {
    if (isInt32Or64(rhs)) {
      if (cast<IntegerAttr>(rhs).getValue().isZero())
        return getLhs();
    } else if (isF32OrF64(rhs)) {
      if (cast<FloatAttr>(rhs).getValue().isZero())
        return getLhs();
    }
  }

  // If both constant, do a direct sub
  Attribute folded = tryFoldBinaryOp(
      adaptor.getLhs(), adaptor.getRhs(),
      [&](Attribute a, Attribute b) -> Attribute {
        if (auto ia = dyn_cast<IntegerAttr>(a))
          return foldSubInteger(ia, cast<IntegerAttr>(b), getContext());
        else
          return foldSubFloat(cast<FloatAttr>(a), cast<FloatAttr>(b));
      });
  if (folded)
    return folded;

  return {};
}

/// SsaWasm_MulOp folding:
OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  // mul(x, 0) -> 0
  // mul(x, 1) -> x
  if (Attribute rhs = adaptor.getRhs()) {
    // Check if it is integer 0 or 1:
    if (auto iAttr = dyn_cast<IntegerAttr>(rhs)) {
      const APInt &val = iAttr.getValue();
      if (val == 0)
        return rhs; // 0
      if (val == 1)
        return getLhs();
    }
    // Check if it is float 0.0 or 1.0:
    if (auto fAttr = dyn_cast<FloatAttr>(rhs)) {
      if (fAttr.getValue().isZero())
        return rhs;
      if (fAttr.getValue().compare(APFloat(1.0)) == APFloat::cmpEqual)
        return getLhs();
    }
  }

  // If both constant, do a direct mul
  if (Attribute folded = tryFoldBinaryOp(
          adaptor.getLhs(), adaptor.getRhs(),
          [&](Attribute a, Attribute b) -> Attribute {
            if (auto ia = dyn_cast<IntegerAttr>(a))
              return foldMulInteger(ia, cast<IntegerAttr>(b), getContext());
            else
              return foldMulFloat(cast<FloatAttr>(a), cast<FloatAttr>(b));
          }))
    return folded;

  return {};
}

/// SsaWasm_MinOp folding:
OpFoldResult MinOp::fold(FoldAdaptor adaptor) {
  // If both constant, do a direct min
  if (Attribute folded = tryFoldBinaryOp(
          adaptor.getLhs(), adaptor.getRhs(),
          [&](Attribute a, Attribute b) -> Attribute {
            // For demonstration, assume “min” is a *signed* min for int:
            if (auto ia = dyn_cast<IntegerAttr>(a))
              return foldMinInteger(ia, cast<IntegerAttr>(b));
            else
              return foldMinFloat(cast<FloatAttr>(a), cast<FloatAttr>(b));
          }))
    return folded;

  return {};
}

/// SsaWasm_MaxOp folding:
OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  // If both constant, do a direct max
  if (Attribute folded = tryFoldBinaryOp(
          adaptor.getLhs(), adaptor.getRhs(),
          [&](Attribute a, Attribute b) -> Attribute {
            // For demonstration, assume “max” is a *signed* max for int:
            if (auto ia = dyn_cast<IntegerAttr>(a))
              return foldMaxInteger(ia, cast<IntegerAttr>(b));
            else
              return foldMaxFloat(cast<FloatAttr>(a), cast<FloatAttr>(b));
          }))
    return folded;

  return {};
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
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

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

void LoadOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getAddr());
  p << " : ";
  p.printType(getType());
}

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand addr;
  if (parser.parseOperand(addr))
    return failure();

  Type addrType = WasmIntegerType::get(parser.getContext(), 32);
  if (parser.resolveOperand(addr, addrType, result.operands))
    return failure();

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();
  result.addTypes(resultType);

  return success();
}

void StoreOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getAddr());
  p << ", ";
  p.printOperand(getValue());
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand addr;
  OpAsmParser::UnresolvedOperand value;
  if (parser.parseOperand(addr) || parser.parseComma() ||
      parser.parseOperand(value))
    return failure();

  Type addrType = WasmIntegerType::get(parser.getContext(), 32);
  if (parser.resolveOperand(addr, addrType, result.operands))
    return failure();

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();
  result.addTypes(resultType);

  return success();
}

// BlockLoopOp
std::tuple<Block *, Block *, Block *>
BlockLoopOp::initialize(OpBuilder &builder) {
  Block *entryBlock = builder.createBlock(&getRegion());
  Block *loopStartLabel = builder.createBlock(&getRegion());
  Block *blockEndLabel = builder.createBlock(&getRegion());

  builder.setInsertionPointToEnd(entryBlock);
  builder.create<TempBranchOp>(getLoc(), loopStartLabel);

  builder.setInsertionPointToEnd(blockEndLabel);
  builder.create<BlockLoopTerminatorOp>(getLoc());

  return std::make_tuple(entryBlock, loopStartLabel, blockEndLabel);
}

Block *BlockLoopOp::getEntryBlock() { return &getRegion().front(); }
Block *BlockLoopOp::getExitBlock() {
  // find block with BlockLoopTerminatorOp
  for (Block &block : getRegion()) {
    for (Operation &op : block) {
      if (isa<BlockLoopTerminatorOp>(op)) {
        return &block;
      }
    }
  }
  return nullptr;
}

bool BlockLoopBranchOp::isBranchingToBegin() {
  auto blockLoopOp = getParentOp<BlockLoopOp>();
  return getDest() == blockLoopOp.getEntryBlock();
}

bool BlockLoopCondBranchOp::isBranchingToBegin() {
  auto blockLoopOp = getParentOp<BlockLoopOp>();
  return getDest() == blockLoopOp.getEntryBlock();
}

// for SsaWasm::DataOp
// copied from mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp

static void printGlobalMemrefOpTypeAndInitialValue(OpAsmPrinter &p, DataOp op,
                                                   TypeAttr type,
                                                   Attribute initialValue) {
  p << type;
  if (!op.isExternal()) {
    p << " = ";
    if (op.isUninitialized())
      p << "uninitialized";
    else
      p.printAttributeWithoutType(initialValue);
  }
}

static ParseResult
parseGlobalMemrefOpTypeAndInitialValue(OpAsmParser &parser, TypeAttr &typeAttr,
                                       Attribute &initialValue) {
  Type type;
  if (parser.parseType(type))
    return failure();

  auto memrefType = llvm::dyn_cast<MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape())
    return parser.emitError(parser.getNameLoc())
           << "type should be static shaped memref, but got " << type;
  typeAttr = TypeAttr::get(type);

  if (parser.parseOptionalEqual())
    return success();

  if (succeeded(parser.parseOptionalKeyword("uninitialized"))) {
    initialValue = UnitAttr::get(parser.getContext());
    return success();
  }

  Type tensorType = memref::getTensorTypeFromMemRefType(memrefType);
  if (parser.parseAttribute(initialValue, tensorType))
    return failure();
  if (!llvm::isa<ElementsAttr>(initialValue))
    return parser.emitError(parser.getNameLoc())
           << "initial value should be a unit or elements attribute";
  return success();
}

} // namespace mlir::ssawasm

#include "SsaWasm/SsaWasmOps.cpp.inc"