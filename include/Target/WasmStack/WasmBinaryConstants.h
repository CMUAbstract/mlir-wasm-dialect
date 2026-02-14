//===- WasmBinaryConstants.h - WebAssembly binary format constants -*- C++
//-*-=//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_WASMSTACK_WASMBINARYCONSTANTS_H
#define TARGET_WASMSTACK_WASMBINARYCONSTANTS_H

#include <cstdint>

namespace mlir::wasmstack::wasm {

// WebAssembly magic number and version
constexpr uint8_t Magic[] = {0x00, 0x61, 0x73, 0x6D}; // "\0asm"
constexpr uint8_t Version[] = {0x01, 0x00, 0x00, 0x00};

// Section IDs
enum class SectionId : uint8_t {
  Custom = 0,
  Type = 1,
  Import = 2,
  Function = 3,
  Table = 4,
  Memory = 5,
  Global = 6,
  Export = 7,
  Start = 8,
  Element = 9,
  Code = 10,
  Data = 11,
  DataCount = 12,
  Tag = 13,
};

// Value types
enum class ValType : uint8_t {
  I32 = 0x7F,
  I64 = 0x7E,
  F32 = 0x7D,
  F64 = 0x7C,
  FuncRef = 0x70,
  ExternRef = 0x6F,
  ContRef = 0x68,
};

// Composite reference type constructors.
namespace RefType {
constexpr uint8_t RefNull = 0x63;
constexpr uint8_t Ref = 0x64;
} // namespace RefType

// Block types
constexpr uint8_t BlockTypeVoid = 0x40;

// Function type constructor
constexpr uint8_t FuncTypeTag = 0x60;

// Export kinds
enum class ExportKind : uint8_t {
  Func = 0x00,
  Table = 0x01,
  Memory = 0x02,
  Global = 0x03,
  Tag = 0x04,
};

// Import kinds
enum class ImportKind : uint8_t {
  Func = 0x00,
  Table = 0x01,
  Memory = 0x02,
  Global = 0x03,
  Tag = 0x04,
};

// Mutability
enum class Mutability : uint8_t {
  Const = 0x00,
  Var = 0x01,
};

// Limit kinds
enum class LimitKind : uint8_t {
  Min = 0x00,
  MinMax = 0x01,
};

// Opcodes
namespace Opcode {
// Control flow
constexpr uint8_t Unreachable = 0x00;
constexpr uint8_t Nop = 0x01;
constexpr uint8_t Block = 0x02;
constexpr uint8_t Loop = 0x03;
constexpr uint8_t If = 0x04;
constexpr uint8_t Else = 0x05;
constexpr uint8_t End = 0x0B;
constexpr uint8_t Br = 0x0C;
constexpr uint8_t BrIf = 0x0D;
constexpr uint8_t BrTable = 0x0E;
constexpr uint8_t Return = 0x0F;
constexpr uint8_t Call = 0x10;
constexpr uint8_t CallIndirect = 0x11;

// Parametric
constexpr uint8_t Drop = 0x1A;
constexpr uint8_t Select = 0x1B;
constexpr uint8_t SelectTyped = 0x1C;

// Variable access
constexpr uint8_t LocalGet = 0x20;
constexpr uint8_t LocalSet = 0x21;
constexpr uint8_t LocalTee = 0x22;
constexpr uint8_t GlobalGet = 0x23;
constexpr uint8_t GlobalSet = 0x24;

// Memory - loads
constexpr uint8_t I32Load = 0x28;
constexpr uint8_t I64Load = 0x29;
constexpr uint8_t F32Load = 0x2A;
constexpr uint8_t F64Load = 0x2B;
constexpr uint8_t I32Load8S = 0x2C;
constexpr uint8_t I32Load8U = 0x2D;
constexpr uint8_t I32Load16S = 0x2E;
constexpr uint8_t I32Load16U = 0x2F;
constexpr uint8_t I64Load8S = 0x30;
constexpr uint8_t I64Load8U = 0x31;
constexpr uint8_t I64Load16S = 0x32;
constexpr uint8_t I64Load16U = 0x33;
constexpr uint8_t I64Load32S = 0x34;
constexpr uint8_t I64Load32U = 0x35;

// Memory - stores
constexpr uint8_t I32Store = 0x36;
constexpr uint8_t I64Store = 0x37;
constexpr uint8_t F32Store = 0x38;
constexpr uint8_t F64Store = 0x39;
constexpr uint8_t I32Store8 = 0x3A;
constexpr uint8_t I32Store16 = 0x3B;
constexpr uint8_t I64Store8 = 0x3C;
constexpr uint8_t I64Store16 = 0x3D;
constexpr uint8_t I64Store32 = 0x3E;

// Memory size/grow
constexpr uint8_t MemorySize = 0x3F;
constexpr uint8_t MemoryGrow = 0x40;

// Constants
constexpr uint8_t I32Const = 0x41;
constexpr uint8_t I64Const = 0x42;
constexpr uint8_t F32Const = 0x43;
constexpr uint8_t F64Const = 0x44;

// i32 comparison
constexpr uint8_t I32Eqz = 0x45;
constexpr uint8_t I32Eq = 0x46;
constexpr uint8_t I32Ne = 0x47;
constexpr uint8_t I32LtS = 0x48;
constexpr uint8_t I32LtU = 0x49;
constexpr uint8_t I32GtS = 0x4A;
constexpr uint8_t I32GtU = 0x4B;
constexpr uint8_t I32LeS = 0x4C;
constexpr uint8_t I32LeU = 0x4D;
constexpr uint8_t I32GeS = 0x4E;
constexpr uint8_t I32GeU = 0x4F;

// i64 comparison
constexpr uint8_t I64Eqz = 0x50;
constexpr uint8_t I64Eq = 0x51;
constexpr uint8_t I64Ne = 0x52;
constexpr uint8_t I64LtS = 0x53;
constexpr uint8_t I64LtU = 0x54;
constexpr uint8_t I64GtS = 0x55;
constexpr uint8_t I64GtU = 0x56;
constexpr uint8_t I64LeS = 0x57;
constexpr uint8_t I64LeU = 0x58;
constexpr uint8_t I64GeS = 0x59;
constexpr uint8_t I64GeU = 0x5A;

// f32 comparison
constexpr uint8_t F32Eq = 0x5B;
constexpr uint8_t F32Ne = 0x5C;
constexpr uint8_t F32Lt = 0x5D;
constexpr uint8_t F32Gt = 0x5E;
constexpr uint8_t F32Le = 0x5F;
constexpr uint8_t F32Ge = 0x60;

// f64 comparison
constexpr uint8_t F64Eq = 0x61;
constexpr uint8_t F64Ne = 0x62;
constexpr uint8_t F64Lt = 0x63;
constexpr uint8_t F64Gt = 0x64;
constexpr uint8_t F64Le = 0x65;
constexpr uint8_t F64Ge = 0x66;

// i32 arithmetic
constexpr uint8_t I32Clz = 0x67;
constexpr uint8_t I32Ctz = 0x68;
constexpr uint8_t I32Popcnt = 0x69;
constexpr uint8_t I32Add = 0x6A;
constexpr uint8_t I32Sub = 0x6B;
constexpr uint8_t I32Mul = 0x6C;
constexpr uint8_t I32DivS = 0x6D;
constexpr uint8_t I32DivU = 0x6E;
constexpr uint8_t I32RemS = 0x6F;
constexpr uint8_t I32RemU = 0x70;
constexpr uint8_t I32And = 0x71;
constexpr uint8_t I32Or = 0x72;
constexpr uint8_t I32Xor = 0x73;
constexpr uint8_t I32Shl = 0x74;
constexpr uint8_t I32ShrS = 0x75;
constexpr uint8_t I32ShrU = 0x76;
constexpr uint8_t I32Rotl = 0x77;
constexpr uint8_t I32Rotr = 0x78;

// i64 arithmetic
constexpr uint8_t I64Clz = 0x79;
constexpr uint8_t I64Ctz = 0x7A;
constexpr uint8_t I64Popcnt = 0x7B;
constexpr uint8_t I64Add = 0x7C;
constexpr uint8_t I64Sub = 0x7D;
constexpr uint8_t I64Mul = 0x7E;
constexpr uint8_t I64DivS = 0x7F;
constexpr uint8_t I64DivU = 0x80;
constexpr uint8_t I64RemS = 0x81;
constexpr uint8_t I64RemU = 0x82;
constexpr uint8_t I64And = 0x83;
constexpr uint8_t I64Or = 0x84;
constexpr uint8_t I64Xor = 0x85;
constexpr uint8_t I64Shl = 0x86;
constexpr uint8_t I64ShrS = 0x87;
constexpr uint8_t I64ShrU = 0x88;
constexpr uint8_t I64Rotl = 0x89;
constexpr uint8_t I64Rotr = 0x8A;

// f32 arithmetic
constexpr uint8_t F32Abs = 0x8B;
constexpr uint8_t F32Neg = 0x8C;
constexpr uint8_t F32Ceil = 0x8D;
constexpr uint8_t F32Floor = 0x8E;
constexpr uint8_t F32Trunc = 0x8F;
constexpr uint8_t F32Nearest = 0x90;
constexpr uint8_t F32Sqrt = 0x91;
constexpr uint8_t F32Add = 0x92;
constexpr uint8_t F32Sub = 0x93;
constexpr uint8_t F32Mul = 0x94;
constexpr uint8_t F32Div = 0x95;
constexpr uint8_t F32Min = 0x96;
constexpr uint8_t F32Max = 0x97;
constexpr uint8_t F32Copysign = 0x98;

// f64 arithmetic
constexpr uint8_t F64Abs = 0x99;
constexpr uint8_t F64Neg = 0x9A;
constexpr uint8_t F64Ceil = 0x9B;
constexpr uint8_t F64Floor = 0x9C;
constexpr uint8_t F64Trunc = 0x9D;
constexpr uint8_t F64Nearest = 0x9E;
constexpr uint8_t F64Sqrt = 0x9F;
constexpr uint8_t F64Add = 0xA0;
constexpr uint8_t F64Sub = 0xA1;
constexpr uint8_t F64Mul = 0xA2;
constexpr uint8_t F64Div = 0xA3;
constexpr uint8_t F64Min = 0xA4;
constexpr uint8_t F64Max = 0xA5;
constexpr uint8_t F64Copysign = 0xA6;

// Conversions
constexpr uint8_t I32WrapI64 = 0xA7;
constexpr uint8_t I32TruncF32S = 0xA8;
constexpr uint8_t I32TruncF32U = 0xA9;
constexpr uint8_t I32TruncF64S = 0xAA;
constexpr uint8_t I32TruncF64U = 0xAB;
constexpr uint8_t I64ExtendI32S = 0xAC;
constexpr uint8_t I64ExtendI32U = 0xAD;
constexpr uint8_t I64TruncF32S = 0xAE;
constexpr uint8_t I64TruncF32U = 0xAF;
constexpr uint8_t I64TruncF64S = 0xB0;
constexpr uint8_t I64TruncF64U = 0xB1;
constexpr uint8_t F32ConvertI32S = 0xB2;
constexpr uint8_t F32ConvertI32U = 0xB3;
constexpr uint8_t F32ConvertI64S = 0xB4;
constexpr uint8_t F32ConvertI64U = 0xB5;
constexpr uint8_t F32DemoteF64 = 0xB6;
constexpr uint8_t F64ConvertI32S = 0xB7;
constexpr uint8_t F64ConvertI32U = 0xB8;
constexpr uint8_t F64ConvertI64S = 0xB9;
constexpr uint8_t F64ConvertI64U = 0xBA;
constexpr uint8_t F64PromoteF32 = 0xBB;
constexpr uint8_t I32ReinterpretF32 = 0xBC;
constexpr uint8_t I64ReinterpretF64 = 0xBD;
constexpr uint8_t F32ReinterpretI32 = 0xBE;
constexpr uint8_t F64ReinterpretI64 = 0xBF;

// Reference types
constexpr uint8_t RefNull = 0xD0;
constexpr uint8_t RefFunc = 0xD2;

// Stack switching / typed continuations
constexpr uint8_t ContNew = 0xE0;
constexpr uint8_t ContBind = 0xE1;
constexpr uint8_t Suspend = 0xE2;
constexpr uint8_t Resume = 0xE3;
constexpr uint8_t ResumeThrow = 0xE4;
constexpr uint8_t ResumeThrowRef = 0xE5;
constexpr uint8_t Switch = 0xE6;

} // namespace Opcode

// Relocation types (used in "reloc.*" custom sections)
enum class RelocType : uint8_t {
  R_WASM_FUNCTION_INDEX_LEB = 0,
  R_WASM_TABLE_INDEX_SLEB = 1,
  R_WASM_MEMORY_ADDR_SLEB = 4,
  R_WASM_MEMORY_ADDR_I32 = 5,
  R_WASM_TYPE_INDEX_LEB = 6,
  R_WASM_GLOBAL_INDEX_LEB = 7,
};

// Symbol table kinds (in linking section WASM_SYMBOL_TABLE subsection)
enum class SymtabKind : uint8_t {
  Function = 0,
  Data = 1,
  Global = 2,
  Section = 3,
};

// Symbol flags
constexpr uint32_t WASM_SYMBOL_UNDEFINED = 0x10;
constexpr uint32_t WASM_SYMBOL_BINDING_LOCAL = 0x02;
constexpr uint32_t WASM_SYMBOL_VISIBILITY_HIDDEN = 0x04;
constexpr uint32_t WASM_SYMBOL_EXPORTED = 0x20;
constexpr uint32_t WASM_SYMBOL_EXPLICIT_NAME = 0x40;

// Linking subsection types
enum class LinkingSubsection : uint8_t {
  WASM_SEGMENT_INFO = 5,
  WASM_INIT_FUNCS = 6,
  WASM_COMDAT_INFO = 7,
  WASM_SYMBOL_TABLE = 8,
};

// Linking metadata version
constexpr uint32_t WasmMetadataVersion = 2;

} // namespace mlir::wasmstack::wasm

#endif // TARGET_WASMSTACK_WASMBINARYCONSTANTS_H
