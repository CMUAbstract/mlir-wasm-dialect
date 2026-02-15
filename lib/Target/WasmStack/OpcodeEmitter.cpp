//===- OpcodeEmitter.cpp - WasmStack op to binary opcode emitter -*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target/WasmStack/OpcodeEmitter.h"
#include "Target/WasmStack/WasmBinaryConstants.h"
#include "wasmstack/WasmStackOps.h"

using namespace mlir;
using namespace mlir::wasmstack;
namespace wc = mlir::wasmstack::wasm;

//===----------------------------------------------------------------------===//
// Label resolution
//===----------------------------------------------------------------------===//

std::optional<uint32_t>
OpcodeEmitter::resolveLabelDepth(llvm::StringRef label) const {
  uint32_t depth = 0;
  for (auto it = labelStack.rbegin(), end = labelStack.rend(); it != end;
       ++it, ++depth) {
    if (it->first == label)
      return depth;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Block type encoding
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitBlockType(Operation *op, ArrayAttr paramTypes,
                                  ArrayAttr resultTypes) {
  bool noParams = !paramTypes || paramTypes.empty();
  size_t numResults = resultTypes ? resultTypes.size() : 0;

  if (noParams && numResults == 0) {
    // void block type
    writer.writeByte(wc::BlockTypeVoid);
    return true;
  } else if (noParams && numResults == 1) {
    // Single result type - encode as valtype byte
    Type resultType = cast<TypeAttr>(resultTypes[0]).getValue();
    if (!writer.writeValType(resultType, &indexSpace))
      return false;
    return true;
  } else {
    // Multi-value: encode as type index (signed LEB128)
    IndexSpace::FuncSig sig;
    if (paramTypes) {
      for (Attribute a : paramTypes)
        sig.params.push_back(cast<TypeAttr>(a).getValue());
    }
    if (resultTypes) {
      for (Attribute a : resultTypes)
        sig.results.push_back(cast<TypeAttr>(a).getValue());
    }
    auto typeIdx = indexSpace.tryGetTypeIndex(sig);
    if (!typeIdx) {
      op->emitOpError("block signature type was not indexed for binary "
                      "encoding");
      return false;
    }
    writer.writeSLEB128(static_cast<int32_t>(*typeIdx));
    return true;
  }
}

//===----------------------------------------------------------------------===//
// Function body emission
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitFunctionBody(Operation *funcOp) {
  auto func = cast<FuncOp>(funcOp);
  Block &body = func.getBody().front();

  for (Operation &op : body) {
    if (!emitOperation(&op))
      return false;
  }

  // Emit implicit end for function body
  writer.writeByte(wc::Opcode::End);
  return true;
}

//===----------------------------------------------------------------------===//
// Operation dispatch
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitOperation(Operation *op) {
  // Try each category
  if (emitConstOp(op))
    return true;
  if (emitLocalOp(op))
    return true;
  if (emitGlobalOp(op))
    return true;
  if (emitArithmeticOp(op))
    return true;
  if (emitCompareOp(op))
    return true;
  if (emitConversionOp(op))
    return true;
  if (emitControlFlowOp(op))
    return true;
  if (emitMemoryOp(op))
    return true;
  if (emitStackSwitchingOp(op))
    return true;
  if (emitCallOp(op))
    return true;
  if (emitMiscOp(op))
    return true;

  // Skip local declarations (handled in code section preamble)
  if (isa<LocalOp>(op))
    return true;

  op->emitError("unsupported operation in binary emitter: ") << op->getName();
  return false;
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitConstOp(Operation *op) {
  if (auto constOp = dyn_cast<I32ConstOp>(op)) {
    writer.writeByte(wc::Opcode::I32Const);
    writer.writeSLEB128(static_cast<int32_t>(constOp.getValue()));
    return true;
  }
  if (auto constOp = dyn_cast<I64ConstOp>(op)) {
    writer.writeByte(wc::Opcode::I64Const);
    writer.writeSLEB128(constOp.getValue());
    return true;
  }
  if (auto constOp = dyn_cast<F32ConstOp>(op)) {
    writer.writeByte(wc::Opcode::F32Const);
    writer.writeF32(constOp.getValue().convertToFloat());
    return true;
  }
  if (auto constOp = dyn_cast<F64ConstOp>(op)) {
    writer.writeByte(wc::Opcode::F64Const);
    writer.writeF64(constOp.getValue().convertToDouble());
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Local variable operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitLocalOp(Operation *op) {
  if (auto localGet = dyn_cast<LocalGetOp>(op)) {
    writer.writeByte(wc::Opcode::LocalGet);
    writer.writeULEB128(localGet.getIndex());
    return true;
  }
  if (auto localSet = dyn_cast<LocalSetOp>(op)) {
    writer.writeByte(wc::Opcode::LocalSet);
    writer.writeULEB128(localSet.getIndex());
    return true;
  }
  if (auto localTee = dyn_cast<LocalTeeOp>(op)) {
    writer.writeByte(wc::Opcode::LocalTee);
    writer.writeULEB128(localTee.getIndex());
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Global variable operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitGlobalOp(Operation *op) {
  if (auto globalGet = dyn_cast<GlobalGetOp>(op)) {
    writer.writeByte(wc::Opcode::GlobalGet);
    auto idx = indexSpace.tryGetGlobalIndex(globalGet.getGlobal());
    if (!idx) {
      globalGet.emitOpError("unresolved global symbol '")
          << globalGet.getGlobal() << "' in binary emitter";
      return false;
    }
    if (tracker) {
      auto symIdx = indexSpace.tryGetSymbolIndex(globalGet.getGlobal());
      if (!symIdx) {
        globalGet.emitOpError("missing relocation symbol for global '")
            << globalGet.getGlobal() << "'";
        return false;
      }
      tracker->addCodeRelocation(
          static_cast<uint8_t>(wc::RelocType::R_WASM_GLOBAL_INDEX_LEB),
          sectionOffset + writer.offset(), *symIdx);
      writer.writeFixedULEB128(*idx);
    } else {
      writer.writeULEB128(*idx);
    }
    return true;
  }
  if (auto globalSet = dyn_cast<GlobalSetOp>(op)) {
    writer.writeByte(wc::Opcode::GlobalSet);
    auto idx = indexSpace.tryGetGlobalIndex(globalSet.getGlobal());
    if (!idx) {
      globalSet.emitOpError("unresolved global symbol '")
          << globalSet.getGlobal() << "' in binary emitter";
      return false;
    }
    if (tracker) {
      auto symIdx = indexSpace.tryGetSymbolIndex(globalSet.getGlobal());
      if (!symIdx) {
        globalSet.emitOpError("missing relocation symbol for global '")
            << globalSet.getGlobal() << "'";
        return false;
      }
      tracker->addCodeRelocation(
          static_cast<uint8_t>(wc::RelocType::R_WASM_GLOBAL_INDEX_LEB),
          sectionOffset + writer.offset(), *symIdx);
      writer.writeFixedULEB128(*idx);
    } else {
      writer.writeULEB128(*idx);
    }
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Type-dispatched opcode helper
//===----------------------------------------------------------------------===//

std::optional<uint8_t> OpcodeEmitter::getTypedOpcode(llvm::StringRef opName,
                                                     mlir::Type type) const {
  bool isI32 = type.isInteger(32);
  bool isI64 = type.isInteger(64);
  bool isF32 = type.isF32();
  bool isF64 = type.isF64();

  // Binary arithmetic
  if (opName == "add") {
    if (isI32)
      return wc::Opcode::I32Add;
    if (isI64)
      return wc::Opcode::I64Add;
    if (isF32)
      return wc::Opcode::F32Add;
    if (isF64)
      return wc::Opcode::F64Add;
  }
  if (opName == "sub") {
    if (isI32)
      return wc::Opcode::I32Sub;
    if (isI64)
      return wc::Opcode::I64Sub;
    if (isF32)
      return wc::Opcode::F32Sub;
    if (isF64)
      return wc::Opcode::F64Sub;
  }
  if (opName == "mul") {
    if (isI32)
      return wc::Opcode::I32Mul;
    if (isI64)
      return wc::Opcode::I64Mul;
    if (isF32)
      return wc::Opcode::F32Mul;
    if (isF64)
      return wc::Opcode::F64Mul;
  }
  if (opName == "div_s") {
    if (isI32)
      return wc::Opcode::I32DivS;
    if (isI64)
      return wc::Opcode::I64DivS;
  }
  if (opName == "div_u") {
    if (isI32)
      return wc::Opcode::I32DivU;
    if (isI64)
      return wc::Opcode::I64DivU;
  }
  if (opName == "div") {
    if (isF32)
      return wc::Opcode::F32Div;
    if (isF64)
      return wc::Opcode::F64Div;
  }
  if (opName == "rem_s") {
    if (isI32)
      return wc::Opcode::I32RemS;
    if (isI64)
      return wc::Opcode::I64RemS;
  }
  if (opName == "rem_u") {
    if (isI32)
      return wc::Opcode::I32RemU;
    if (isI64)
      return wc::Opcode::I64RemU;
  }
  // Bitwise
  if (opName == "and") {
    if (isI32)
      return wc::Opcode::I32And;
    if (isI64)
      return wc::Opcode::I64And;
  }
  if (opName == "or") {
    if (isI32)
      return wc::Opcode::I32Or;
    if (isI64)
      return wc::Opcode::I64Or;
  }
  if (opName == "xor") {
    if (isI32)
      return wc::Opcode::I32Xor;
    if (isI64)
      return wc::Opcode::I64Xor;
  }
  if (opName == "shl") {
    if (isI32)
      return wc::Opcode::I32Shl;
    if (isI64)
      return wc::Opcode::I64Shl;
  }
  if (opName == "shr_s") {
    if (isI32)
      return wc::Opcode::I32ShrS;
    if (isI64)
      return wc::Opcode::I64ShrS;
  }
  if (opName == "shr_u") {
    if (isI32)
      return wc::Opcode::I32ShrU;
    if (isI64)
      return wc::Opcode::I64ShrU;
  }
  if (opName == "rotl") {
    if (isI32)
      return wc::Opcode::I32Rotl;
    if (isI64)
      return wc::Opcode::I64Rotl;
  }
  if (opName == "rotr") {
    if (isI32)
      return wc::Opcode::I32Rotr;
    if (isI64)
      return wc::Opcode::I64Rotr;
  }
  // Unary integer
  if (opName == "clz") {
    if (isI32)
      return wc::Opcode::I32Clz;
    if (isI64)
      return wc::Opcode::I64Clz;
  }
  if (opName == "ctz") {
    if (isI32)
      return wc::Opcode::I32Ctz;
    if (isI64)
      return wc::Opcode::I64Ctz;
  }
  if (opName == "popcnt") {
    if (isI32)
      return wc::Opcode::I32Popcnt;
    if (isI64)
      return wc::Opcode::I64Popcnt;
  }
  // Unary float
  if (opName == "abs") {
    if (isF32)
      return wc::Opcode::F32Abs;
    if (isF64)
      return wc::Opcode::F64Abs;
  }
  if (opName == "neg") {
    if (isF32)
      return wc::Opcode::F32Neg;
    if (isF64)
      return wc::Opcode::F64Neg;
  }
  if (opName == "ceil") {
    if (isF32)
      return wc::Opcode::F32Ceil;
    if (isF64)
      return wc::Opcode::F64Ceil;
  }
  if (opName == "floor") {
    if (isF32)
      return wc::Opcode::F32Floor;
    if (isF64)
      return wc::Opcode::F64Floor;
  }
  if (opName == "trunc") {
    if (isF32)
      return wc::Opcode::F32Trunc;
    if (isF64)
      return wc::Opcode::F64Trunc;
  }
  if (opName == "nearest") {
    if (isF32)
      return wc::Opcode::F32Nearest;
    if (isF64)
      return wc::Opcode::F64Nearest;
  }
  if (opName == "sqrt") {
    if (isF32)
      return wc::Opcode::F32Sqrt;
    if (isF64)
      return wc::Opcode::F64Sqrt;
  }
  // Float binary
  if (opName == "min") {
    if (isF32)
      return wc::Opcode::F32Min;
    if (isF64)
      return wc::Opcode::F64Min;
  }
  if (opName == "max") {
    if (isF32)
      return wc::Opcode::F32Max;
    if (isF64)
      return wc::Opcode::F64Max;
  }
  if (opName == "copysign") {
    if (isF32)
      return wc::Opcode::F32Copysign;
    if (isF64)
      return wc::Opcode::F64Copysign;
  }
  // Comparison
  if (opName == "eqz") {
    if (isI32)
      return wc::Opcode::I32Eqz;
    if (isI64)
      return wc::Opcode::I64Eqz;
  }
  if (opName == "eq") {
    if (isI32)
      return wc::Opcode::I32Eq;
    if (isI64)
      return wc::Opcode::I64Eq;
    if (isF32)
      return wc::Opcode::F32Eq;
    if (isF64)
      return wc::Opcode::F64Eq;
  }
  if (opName == "ne") {
    if (isI32)
      return wc::Opcode::I32Ne;
    if (isI64)
      return wc::Opcode::I64Ne;
    if (isF32)
      return wc::Opcode::F32Ne;
    if (isF64)
      return wc::Opcode::F64Ne;
  }
  if (opName == "lt_s") {
    if (isI32)
      return wc::Opcode::I32LtS;
    if (isI64)
      return wc::Opcode::I64LtS;
  }
  if (opName == "lt_u") {
    if (isI32)
      return wc::Opcode::I32LtU;
    if (isI64)
      return wc::Opcode::I64LtU;
  }
  if (opName == "gt_s") {
    if (isI32)
      return wc::Opcode::I32GtS;
    if (isI64)
      return wc::Opcode::I64GtS;
  }
  if (opName == "gt_u") {
    if (isI32)
      return wc::Opcode::I32GtU;
    if (isI64)
      return wc::Opcode::I64GtU;
  }
  if (opName == "le_s") {
    if (isI32)
      return wc::Opcode::I32LeS;
    if (isI64)
      return wc::Opcode::I64LeS;
  }
  if (opName == "le_u") {
    if (isI32)
      return wc::Opcode::I32LeU;
    if (isI64)
      return wc::Opcode::I64LeU;
  }
  if (opName == "ge_s") {
    if (isI32)
      return wc::Opcode::I32GeS;
    if (isI64)
      return wc::Opcode::I64GeS;
  }
  if (opName == "ge_u") {
    if (isI32)
      return wc::Opcode::I32GeU;
    if (isI64)
      return wc::Opcode::I64GeU;
  }
  // Float comparison (no signed/unsigned distinction)
  if (opName == "lt") {
    if (isF32)
      return wc::Opcode::F32Lt;
    if (isF64)
      return wc::Opcode::F64Lt;
  }
  if (opName == "gt") {
    if (isF32)
      return wc::Opcode::F32Gt;
    if (isF64)
      return wc::Opcode::F64Gt;
  }
  if (opName == "le") {
    if (isF32)
      return wc::Opcode::F32Le;
    if (isF64)
      return wc::Opcode::F64Le;
  }
  if (opName == "ge") {
    if (isF32)
      return wc::Opcode::F32Ge;
    if (isF64)
      return wc::Opcode::F64Ge;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitArithmeticOp(Operation *op) {
  // Extract the operation mnemonic after "wasmstack."
  llvm::StringRef fullName = op->getName().getStringRef();
  if (!fullName.starts_with("wasmstack."))
    return false;
  llvm::StringRef mnemonic = fullName.drop_front(strlen("wasmstack."));

  // Binary ops (have a type attribute)
  auto typeAttr = op->getAttrOfType<TypeAttr>("type");
  if (!typeAttr)
    return false;
  Type type = typeAttr.getValue();

  // Check if this is a known arithmetic/bitwise/unary op
  static const llvm::StringRef binaryOps[] = {
      "add",  "sub", "mul", "div_s", "div_u",   "rem_s", "rem_u",
      "and",  "or",  "xor", "shl",   "shr_s",   "shr_u", "rotl",
      "rotr", "div", "min", "max",   "copysign"};
  static const llvm::StringRef unaryOps[] = {
      "clz",  "ctz",   "popcnt", "abs",     "neg",
      "ceil", "floor", "trunc",  "nearest", "sqrt"};

  for (auto name : binaryOps) {
    if (mnemonic == name) {
      auto opcode = getTypedOpcode(name, type);
      if (!opcode) {
        op->emitOpError("unsupported typed opcode for '")
            << name << "' with type " << type;
        return false;
      }
      writer.writeByte(*opcode);
      return true;
    }
  }
  for (auto name : unaryOps) {
    if (mnemonic == name) {
      auto opcode = getTypedOpcode(name, type);
      if (!opcode) {
        op->emitOpError("unsupported typed opcode for '")
            << name << "' with type " << type;
        return false;
      }
      writer.writeByte(*opcode);
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitCompareOp(Operation *op) {
  llvm::StringRef fullName = op->getName().getStringRef();
  if (!fullName.starts_with("wasmstack."))
    return false;
  llvm::StringRef mnemonic = fullName.drop_front(strlen("wasmstack."));

  auto typeAttr = op->getAttrOfType<TypeAttr>("type");
  if (!typeAttr)
    return false;
  Type type = typeAttr.getValue();

  static const llvm::StringRef compareOps[] = {
      "eqz",  "eq",   "ne",   "lt_s", "lt_u", "gt_s", "gt_u", "le_s",
      "le_u", "ge_s", "ge_u", "lt",   "gt",   "le",   "ge"};

  for (auto name : compareOps) {
    if (mnemonic == name) {
      auto opcode = getTypedOpcode(name, type);
      if (!opcode) {
        op->emitOpError("unsupported typed opcode for '")
            << name << "' with type " << type;
        return false;
      }
      writer.writeByte(*opcode);
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Conversion operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitConversionOp(Operation *op) {
  // Conversion ops have specific fixed mnemonics
  if (isa<I32WrapI64Op>(op)) {
    writer.writeByte(wc::Opcode::I32WrapI64);
    return true;
  }
  if (isa<I64ExtendI32SOp>(op)) {
    writer.writeByte(wc::Opcode::I64ExtendI32S);
    return true;
  }
  if (isa<I64ExtendI32UOp>(op)) {
    writer.writeByte(wc::Opcode::I64ExtendI32U);
    return true;
  }
  if (isa<I32TruncF32SOp>(op)) {
    writer.writeByte(wc::Opcode::I32TruncF32S);
    return true;
  }
  if (isa<I32TruncF32UOp>(op)) {
    writer.writeByte(wc::Opcode::I32TruncF32U);
    return true;
  }
  if (isa<I32TruncF64SOp>(op)) {
    writer.writeByte(wc::Opcode::I32TruncF64S);
    return true;
  }
  if (isa<I32TruncF64UOp>(op)) {
    writer.writeByte(wc::Opcode::I32TruncF64U);
    return true;
  }
  if (isa<I64TruncF32SOp>(op)) {
    writer.writeByte(wc::Opcode::I64TruncF32S);
    return true;
  }
  if (isa<I64TruncF32UOp>(op)) {
    writer.writeByte(wc::Opcode::I64TruncF32U);
    return true;
  }
  if (isa<I64TruncF64SOp>(op)) {
    writer.writeByte(wc::Opcode::I64TruncF64S);
    return true;
  }
  if (isa<I64TruncF64UOp>(op)) {
    writer.writeByte(wc::Opcode::I64TruncF64U);
    return true;
  }
  if (isa<F32ConvertI32SOp>(op)) {
    writer.writeByte(wc::Opcode::F32ConvertI32S);
    return true;
  }
  if (isa<F32ConvertI32UOp>(op)) {
    writer.writeByte(wc::Opcode::F32ConvertI32U);
    return true;
  }
  if (isa<F32ConvertI64SOp>(op)) {
    writer.writeByte(wc::Opcode::F32ConvertI64S);
    return true;
  }
  if (isa<F32ConvertI64UOp>(op)) {
    writer.writeByte(wc::Opcode::F32ConvertI64U);
    return true;
  }
  if (isa<F64ConvertI32SOp>(op)) {
    writer.writeByte(wc::Opcode::F64ConvertI32S);
    return true;
  }
  if (isa<F64ConvertI32UOp>(op)) {
    writer.writeByte(wc::Opcode::F64ConvertI32U);
    return true;
  }
  if (isa<F64ConvertI64SOp>(op)) {
    writer.writeByte(wc::Opcode::F64ConvertI64S);
    return true;
  }
  if (isa<F64ConvertI64UOp>(op)) {
    writer.writeByte(wc::Opcode::F64ConvertI64U);
    return true;
  }
  if (isa<F32DemoteF64Op>(op)) {
    writer.writeByte(wc::Opcode::F32DemoteF64);
    return true;
  }
  if (isa<F64PromoteF32Op>(op)) {
    writer.writeByte(wc::Opcode::F64PromoteF32);
    return true;
  }
  if (isa<I32ReinterpretF32Op>(op)) {
    writer.writeByte(wc::Opcode::I32ReinterpretF32);
    return true;
  }
  if (isa<I64ReinterpretF64Op>(op)) {
    writer.writeByte(wc::Opcode::I64ReinterpretF64);
    return true;
  }
  if (isa<F32ReinterpretI32Op>(op)) {
    writer.writeByte(wc::Opcode::F32ReinterpretI32);
    return true;
  }
  if (isa<F64ReinterpretI64Op>(op)) {
    writer.writeByte(wc::Opcode::F64ReinterpretI64);
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitControlFlowOp(Operation *op) {
  if (auto blockOp = dyn_cast<BlockOp>(op)) {
    writer.writeByte(wc::Opcode::Block);
    if (!emitBlockType(op, blockOp.getParamTypes(), blockOp.getResultTypes())) {
      blockOp.emitOpError("failed to encode block result type");
      return false;
    }

    // Push label and emit body
    ScopedLabel label(*this, blockOp.getLabel(), /*isLoop=*/false);
    for (Operation &innerOp : blockOp.getBody().front()) {
      if (!emitOperation(&innerOp))
        return false;
    }
    writer.writeByte(wc::Opcode::End);
    return true;
  }

  if (auto loopOp = dyn_cast<LoopOp>(op)) {
    writer.writeByte(wc::Opcode::Loop);
    if (!emitBlockType(op, loopOp.getParamTypes(), loopOp.getResultTypes())) {
      loopOp.emitOpError("failed to encode loop result type");
      return false;
    }

    // Push label and emit body
    ScopedLabel label(*this, loopOp.getLabel(), /*isLoop=*/true);
    for (Operation &innerOp : loopOp.getBody().front()) {
      if (!emitOperation(&innerOp))
        return false;
    }
    writer.writeByte(wc::Opcode::End);
    return true;
  }

  if (auto ifOp = dyn_cast<IfOp>(op)) {
    writer.writeByte(wc::Opcode::If);
    if (!emitBlockType(op, ifOp.getParamTypes(), ifOp.getResultTypes())) {
      ifOp.emitOpError("failed to encode if result type");
      return false;
    }

    // Emit then body
    for (Operation &innerOp : ifOp.getThenBody().front()) {
      if (!emitOperation(&innerOp))
        return false;
    }

    // Emit else body if present
    if (!ifOp.getElseBody().empty()) {
      writer.writeByte(wc::Opcode::Else);
      for (Operation &innerOp : ifOp.getElseBody().front()) {
        if (!emitOperation(&innerOp))
          return false;
      }
    }

    writer.writeByte(wc::Opcode::End);
    return true;
  }

  if (auto brOp = dyn_cast<BrOp>(op)) {
    writer.writeByte(wc::Opcode::Br);
    auto depth = resolveLabelDepth(brOp.getTarget());
    if (!depth) {
      brOp.emitOpError("branch target '")
          << brOp.getTarget() << "' not found in active label stack";
      return false;
    }
    writer.writeULEB128(*depth);
    return true;
  }

  if (auto brIfOp = dyn_cast<BrIfOp>(op)) {
    writer.writeByte(wc::Opcode::BrIf);
    auto depth = resolveLabelDepth(brIfOp.getTarget());
    if (!depth) {
      brIfOp.emitOpError("branch target '")
          << brIfOp.getTarget() << "' not found in active label stack";
      return false;
    }
    writer.writeULEB128(*depth);
    return true;
  }

  if (auto brTableOp = dyn_cast<BrTableOp>(op)) {
    writer.writeByte(wc::Opcode::BrTable);
    ArrayAttr targets = brTableOp.getTargets();
    writer.writeULEB128(targets.size());
    for (Attribute target : targets) {
      auto ref = cast<FlatSymbolRefAttr>(target);
      auto depth = resolveLabelDepth(ref.getValue());
      if (!depth) {
        brTableOp.emitOpError("branch table target '")
            << ref.getValue() << "' not found in active label stack";
        return false;
      }
      writer.writeULEB128(*depth);
    }
    auto defaultDepth = resolveLabelDepth(brTableOp.getDefaultTarget());
    if (!defaultDepth) {
      brTableOp.emitOpError("default branch target '")
          << brTableOp.getDefaultTarget()
          << "' not found in active label stack";
      return false;
    }
    writer.writeULEB128(*defaultDepth);
    return true;
  }

  if (isa<ReturnOp>(op)) {
    writer.writeByte(wc::Opcode::Return);
    return true;
  }

  if (isa<UnreachableOp>(op)) {
    writer.writeByte(wc::Opcode::Unreachable);
    return true;
  }

  if (isa<NopOp>(op)) {
    writer.writeByte(wc::Opcode::Nop);
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Memory operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitMemoryOp(Operation *op) {
  // Helper to emit memarg: log2(align) + offset as uleb128
  auto emitMemarg = [&](uint32_t offset, uint32_t align) {
    // align is in bytes; wasm encodes log2(align)
    uint32_t log2Align = 0;
    uint32_t a = align;
    while (a > 1) {
      log2Align++;
      a >>= 1;
    }
    writer.writeULEB128(log2Align);
    writer.writeULEB128(offset);
  };

  // Loads
  if (auto loadOp = dyn_cast<I32LoadOp>(op)) {
    writer.writeByte(wc::Opcode::I32Load);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I64LoadOp>(op)) {
    writer.writeByte(wc::Opcode::I64Load);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<F32LoadOp>(op)) {
    writer.writeByte(wc::Opcode::F32Load);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<F64LoadOp>(op)) {
    writer.writeByte(wc::Opcode::F64Load);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  // Sub-width loads
  if (auto loadOp = dyn_cast<I32Load8SOp>(op)) {
    writer.writeByte(wc::Opcode::I32Load8S);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I32Load8UOp>(op)) {
    writer.writeByte(wc::Opcode::I32Load8U);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I32Load16SOp>(op)) {
    writer.writeByte(wc::Opcode::I32Load16S);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I32Load16UOp>(op)) {
    writer.writeByte(wc::Opcode::I32Load16U);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I64Load8SOp>(op)) {
    writer.writeByte(wc::Opcode::I64Load8S);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I64Load8UOp>(op)) {
    writer.writeByte(wc::Opcode::I64Load8U);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I64Load16SOp>(op)) {
    writer.writeByte(wc::Opcode::I64Load16S);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I64Load16UOp>(op)) {
    writer.writeByte(wc::Opcode::I64Load16U);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I64Load32SOp>(op)) {
    writer.writeByte(wc::Opcode::I64Load32S);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }
  if (auto loadOp = dyn_cast<I64Load32UOp>(op)) {
    writer.writeByte(wc::Opcode::I64Load32U);
    emitMemarg(loadOp.getOffset(), loadOp.getAlign());
    return true;
  }

  // Stores
  if (auto storeOp = dyn_cast<I32StoreOp>(op)) {
    writer.writeByte(wc::Opcode::I32Store);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }
  if (auto storeOp = dyn_cast<I64StoreOp>(op)) {
    writer.writeByte(wc::Opcode::I64Store);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }
  if (auto storeOp = dyn_cast<F32StoreOp>(op)) {
    writer.writeByte(wc::Opcode::F32Store);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }
  if (auto storeOp = dyn_cast<F64StoreOp>(op)) {
    writer.writeByte(wc::Opcode::F64Store);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }
  // Sub-width stores
  if (auto storeOp = dyn_cast<I32Store8Op>(op)) {
    writer.writeByte(wc::Opcode::I32Store8);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }
  if (auto storeOp = dyn_cast<I32Store16Op>(op)) {
    writer.writeByte(wc::Opcode::I32Store16);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }
  if (auto storeOp = dyn_cast<I64Store8Op>(op)) {
    writer.writeByte(wc::Opcode::I64Store8);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }
  if (auto storeOp = dyn_cast<I64Store16Op>(op)) {
    writer.writeByte(wc::Opcode::I64Store16);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }
  if (auto storeOp = dyn_cast<I64Store32Op>(op)) {
    writer.writeByte(wc::Opcode::I64Store32);
    emitMemarg(storeOp.getOffset(), storeOp.getAlign());
    return true;
  }

  // Memory size/grow
  if (isa<MemorySizeOp>(op)) {
    writer.writeByte(wc::Opcode::MemorySize);
    writer.writeByte(0x00); // memory index 0
    return true;
  }
  if (isa<MemoryGrowOp>(op)) {
    writer.writeByte(wc::Opcode::MemoryGrow);
    writer.writeByte(0x00); // memory index 0
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Call operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitCallOp(Operation *op) {
  if (auto callOp = dyn_cast<CallOp>(op)) {
    auto funcIdx = indexSpace.tryGetFuncIndex(callOp.getCallee());
    if (!funcIdx) {
      callOp.emitOpError("unresolved function symbol '")
          << callOp.getCallee()
          << "' (no wasmstack.func or wasmstack.import_func)";
      return false;
    }

    writer.writeByte(wc::Opcode::Call);
    if (tracker) {
      auto symIdx = indexSpace.tryGetSymbolIndex(callOp.getCallee());
      if (!symIdx) {
        callOp.emitOpError("missing relocation symbol for callee '")
            << callOp.getCallee() << "'";
        return false;
      }
      tracker->addCodeRelocation(
          static_cast<uint8_t>(wc::RelocType::R_WASM_FUNCTION_INDEX_LEB),
          sectionOffset + writer.offset(), *symIdx);
      writer.writeFixedULEB128(*funcIdx);
    } else {
      writer.writeULEB128(*funcIdx);
    }
    return true;
  }
  if (auto callIndirectOp = dyn_cast<CallIndirectOp>(op)) {
    writer.writeByte(wc::Opcode::CallIndirect);
    // Type index for the signature
    FunctionType funcType = callIndirectOp.getCalleeType();
    IndexSpace::FuncSig sig;
    for (Type t : funcType.getInputs())
      sig.params.push_back(t);
    for (Type t : funcType.getResults())
      sig.results.push_back(t);
    auto typeIdx = indexSpace.tryGetTypeIndex(sig);
    if (!typeIdx) {
      callIndirectOp.emitOpError("callee signature not found in type section");
      return false;
    }
    if (tracker) {
      tracker->addCodeRelocation(
          static_cast<uint8_t>(wc::RelocType::R_WASM_TYPE_INDEX_LEB),
          sectionOffset + writer.offset(), *typeIdx);
      writer.writeFixedULEB128(*typeIdx);
    } else {
      writer.writeULEB128(*typeIdx);
    }
    writer.writeByte(0x00); // table index 0
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Stack switching and reference operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitStackSwitchingOp(Operation *op) {
  if (auto refFuncOp = dyn_cast<RefFuncOp>(op)) {
    auto funcIdx =
        indexSpace.tryGetFuncIndex(refFuncOp.getFuncAttr().getValue());
    if (!funcIdx) {
      refFuncOp.emitOpError("unresolved function symbol '")
          << refFuncOp.getFuncAttr().getValue()
          << "' (no wasmstack.func or wasmstack.import_func)";
      return false;
    }

    writer.writeByte(wc::Opcode::RefFunc);
    if (tracker) {
      auto symIdx =
          indexSpace.tryGetSymbolIndex(refFuncOp.getFuncAttr().getValue());
      if (!symIdx) {
        refFuncOp.emitOpError("missing relocation symbol for function '")
            << refFuncOp.getFuncAttr().getValue() << "'";
        return false;
      }
      tracker->addCodeRelocation(
          static_cast<uint8_t>(wc::RelocType::R_WASM_FUNCTION_INDEX_LEB),
          sectionOffset + writer.offset(), *symIdx);
      writer.writeFixedULEB128(*funcIdx);
    } else {
      writer.writeULEB128(*funcIdx);
    }
    return true;
  }

  if (auto refNullOp = dyn_cast<RefNullOp>(op)) {
    writer.writeByte(wc::Opcode::RefNull);
    if (!writer.writeHeapType(refNullOp.getType(), &indexSpace)) {
      refNullOp.emitOpError(
          "unsupported ref.null heap type for binary encoding: ")
          << refNullOp.getType();
      return false;
    }
    return true;
  }

  if (auto contNewOp = dyn_cast<ContNewOp>(op)) {
    writer.writeByte(wc::Opcode::ContNew);
    auto contIdx =
        indexSpace.tryGetContTypeIndex(contNewOp.getContTypeAttr().getValue());
    if (!contIdx) {
      contNewOp.emitOpError("unknown wasmstack.type.cont symbol ")
          << contNewOp.getContTypeAttr();
      return false;
    }
    writer.writeULEB128(*contIdx);
    return true;
  }

  if (auto contBindOp = dyn_cast<ContBindOp>(op)) {
    writer.writeByte(wc::Opcode::ContBind);
    auto srcContIdx = indexSpace.tryGetContTypeIndex(
        contBindOp.getSrcContTypeAttr().getValue());
    if (!srcContIdx) {
      contBindOp.emitOpError("unknown source wasmstack.type.cont symbol ")
          << contBindOp.getSrcContTypeAttr();
      return false;
    }
    auto dstContIdx = indexSpace.tryGetContTypeIndex(
        contBindOp.getDstContTypeAttr().getValue());
    if (!dstContIdx) {
      contBindOp.emitOpError("unknown destination wasmstack.type.cont symbol ")
          << contBindOp.getDstContTypeAttr();
      return false;
    }
    writer.writeULEB128(*srcContIdx);
    writer.writeULEB128(*dstContIdx);
    return true;
  }

  if (auto suspendOp = dyn_cast<SuspendOp>(op)) {
    writer.writeByte(wc::Opcode::Suspend);
    auto tagIdx = indexSpace.tryGetTagIndex(suspendOp.getTagAttr().getValue());
    if (!tagIdx) {
      suspendOp.emitOpError("unknown wasmstack.tag symbol ")
          << suspendOp.getTagAttr();
      return false;
    }
    writer.writeULEB128(*tagIdx);
    return true;
  }

  auto emitHandlers = [&](Operation *resumeLikeOp, ArrayAttr handlers) -> bool {
    writer.writeULEB128(handlers.size());
    for (Attribute attr : handlers) {
      if (auto onSwitch = dyn_cast<OnSwitchHandlerAttr>(attr)) {
        auto tagIdx = indexSpace.tryGetTagIndex(onSwitch.getTag().getValue());
        if (!tagIdx) {
          resumeLikeOp->emitError("unknown handler tag symbol ")
              << onSwitch.getTag();
          return false;
        }
        writer.writeULEB128(1); // switch handler kind
        writer.writeULEB128(*tagIdx);
        continue;
      }

      auto onLabel = dyn_cast<OnLabelHandlerAttr>(attr);
      if (!onLabel) {
        resumeLikeOp->emitError("handlers must contain #wasmstack.on_label or "
                                "#wasmstack.on_switch attributes");
        return false;
      }

      writer.writeULEB128(0); // ordinary suspension handler
      auto tagIdx = indexSpace.tryGetTagIndex(onLabel.getTag().getValue());
      if (!tagIdx) {
        resumeLikeOp->emitError("unknown handler tag symbol ")
            << onLabel.getTag();
        return false;
      }
      auto labelDepth = resolveLabelDepth(onLabel.getLabel().getValue());
      if (!labelDepth) {
        resumeLikeOp->emitError("handler label '")
            << onLabel.getLabel().getValue()
            << "' not found in active label stack";
        return false;
      }
      writer.writeULEB128(*tagIdx);
      writer.writeULEB128(*labelDepth);
    }
    return true;
  };

  if (auto resumeOp = dyn_cast<ResumeOp>(op)) {
    writer.writeByte(wc::Opcode::Resume);
    auto contIdx =
        indexSpace.tryGetContTypeIndex(resumeOp.getContTypeAttr().getValue());
    if (!contIdx) {
      resumeOp.emitOpError("unknown wasmstack.type.cont symbol ")
          << resumeOp.getContTypeAttr();
      return false;
    }
    writer.writeULEB128(*contIdx);
    return emitHandlers(op, resumeOp.getHandlers());
  }

  if (auto resumeThrowOp = dyn_cast<ResumeThrowOp>(op)) {
    // Current wasmstack.resume_throw IR shape has handler table but no explicit
    // throw-tag immediate; emit the handler-only form.
    writer.writeByte(wc::Opcode::ResumeThrowRef);
    auto contIdx = indexSpace.tryGetContTypeIndex(
        resumeThrowOp.getContTypeAttr().getValue());
    if (!contIdx) {
      resumeThrowOp.emitOpError("unknown wasmstack.type.cont symbol ")
          << resumeThrowOp.getContTypeAttr();
      return false;
    }
    writer.writeULEB128(*contIdx);
    return emitHandlers(op, resumeThrowOp.getHandlers());
  }

  if (isa<BarrierOp>(op)) {
    // Barrier is currently modeled as an identity stack fence in WasmStack IR.
    writer.writeByte(wc::Opcode::Nop);
    return true;
  }

  if (auto switchOp = dyn_cast<SwitchOp>(op)) {
    writer.writeByte(wc::Opcode::Switch);
    auto contIdx =
        indexSpace.tryGetContTypeIndex(switchOp.getContTypeAttr().getValue());
    if (!contIdx) {
      switchOp.emitOpError("unknown wasmstack.type.cont symbol ")
          << switchOp.getContTypeAttr();
      return false;
    }
    auto tagIdx = indexSpace.tryGetTagIndex(switchOp.getTagAttr().getValue());
    if (!tagIdx) {
      switchOp.emitOpError("unknown wasmstack.tag symbol ")
          << switchOp.getTagAttr();
      return false;
    }
    writer.writeULEB128(*contIdx);
    writer.writeULEB128(*tagIdx);
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Miscellaneous operations
//===----------------------------------------------------------------------===//

bool OpcodeEmitter::emitMiscOp(Operation *op) {
  if (isa<DropOp>(op)) {
    writer.writeByte(wc::Opcode::Drop);
    return true;
  }
  if (isa<SelectOp>(op)) {
    writer.writeByte(wc::Opcode::Select);
    return true;
  }
  return false;
}
