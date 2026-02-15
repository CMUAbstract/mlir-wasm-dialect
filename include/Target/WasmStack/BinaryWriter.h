//===- BinaryWriter.h - Low-level binary encoding ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_WASMSTACK_BINARYWRITER_H
#define TARGET_WASMSTACK_BINARYWRITER_H

#include "Target/WasmStack/IndexSpace.h"
#include "Target/WasmStack/WasmBinaryConstants.h"
#include "mlir/IR/Types.h"
#include "wasmstack/WasmStackTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::wasmstack {

/// Low-level binary encoding to an in-memory buffer.
class BinaryWriter {
public:
  /// Write a single byte.
  void writeByte(uint8_t byte) { buffer.push_back(byte); }

  /// Write raw bytes.
  void writeBytes(const uint8_t *data, size_t size) {
    buffer.append(data, data + size);
  }

  /// Write unsigned LEB128.
  void writeULEB128(uint64_t value) {
    do {
      uint8_t byte = value & 0x7F;
      value >>= 7;
      if (value != 0)
        byte |= 0x80;
      buffer.push_back(byte);
    } while (value != 0);
  }

  /// Write signed LEB128.
  void writeSLEB128(int64_t value) {
    bool more = true;
    while (more) {
      uint8_t byte = value & 0x7F;
      value >>= 7;
      // Sign bit of byte is second high order bit (0x40)
      if ((value == 0 && (byte & 0x40) == 0) ||
          (value == -1 && (byte & 0x40) != 0)) {
        more = false;
      } else {
        byte |= 0x80;
      }
      buffer.push_back(byte);
    }
  }

  /// Write 5-byte fixed-size unsigned LEB128 (for relocation slots).
  void writeFixedULEB128(uint32_t value) {
    buffer.push_back((value & 0x7F) | 0x80);
    buffer.push_back(((value >> 7) & 0x7F) | 0x80);
    buffer.push_back(((value >> 14) & 0x7F) | 0x80);
    buffer.push_back(((value >> 21) & 0x7F) | 0x80);
    buffer.push_back((value >> 28) & 0x0F);
  }

  /// Write 5-byte fixed-size signed LEB128 (for relocation slots).
  void writeFixedSLEB128(int32_t value) {
    uint32_t uval = static_cast<uint32_t>(value);
    buffer.push_back((uval & 0x7F) | 0x80);
    buffer.push_back(((uval >> 7) & 0x7F) | 0x80);
    buffer.push_back(((uval >> 14) & 0x7F) | 0x80);
    buffer.push_back(((uval >> 21) & 0x7F) | 0x80);
    buffer.push_back((uval >> 28) & 0x0F);
  }

  /// Write IEEE 754 f32 in little-endian.
  void writeF32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    buffer.push_back(bits & 0xFF);
    buffer.push_back((bits >> 8) & 0xFF);
    buffer.push_back((bits >> 16) & 0xFF);
    buffer.push_back((bits >> 24) & 0xFF);
  }

  /// Write IEEE 754 f64 in little-endian.
  void writeF64(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 8; ++i)
      buffer.push_back((bits >> (i * 8)) & 0xFF);
  }

  /// Write a length-prefixed UTF-8 string.
  void writeString(llvm::StringRef str) {
    writeULEB128(str.size());
    writeBytes(reinterpret_cast<const uint8_t *>(str.data()), str.size());
  }

  /// Write a WebAssembly value type byte from an MLIR type.
  /// Returns true on success, false if the type is unsupported.
  bool writeValType(mlir::Type type, const IndexSpace *indexSpace = nullptr) {
    if (type.isInteger(32)) {
      writeByte(static_cast<uint8_t>(wasm::ValType::I32));
    } else if (type.isInteger(64)) {
      writeByte(static_cast<uint8_t>(wasm::ValType::I64));
    } else if (type.isF32()) {
      writeByte(static_cast<uint8_t>(wasm::ValType::F32));
    } else if (type.isF64()) {
      writeByte(static_cast<uint8_t>(wasm::ValType::F64));
    } else if (isa<FuncRefType>(type)) {
      writeByte(static_cast<uint8_t>(wasm::ValType::FuncRef));
    } else if (isa<ExternRefType>(type)) {
      writeByte(static_cast<uint8_t>(wasm::ValType::ExternRef));
    } else if (auto contRefType = dyn_cast<ContRefType>(type)) {
      if (!indexSpace) {
        writeByte(static_cast<uint8_t>(wasm::ValType::ContRef));
      } else {
        auto contIdx = indexSpace->tryGetContTypeIndex(
            contRefType.getTypeName().getValue());
        if (!contIdx)
          return false;
        writeByte(wasm::RefType::RefNull);
        writeSLEB128(static_cast<int64_t>(*contIdx));
      }
    } else if (auto contRefType = dyn_cast<ContRefNonNullType>(type)) {
      if (!indexSpace) {
        // Signature-level fallback keeps continuation refs in canonical generic
        // form when no heaptype index space is available.
        writeByte(static_cast<uint8_t>(wasm::ValType::ContRef));
      } else {
        auto contIdx = indexSpace->tryGetContTypeIndex(
            contRefType.getTypeName().getValue());
        if (!contIdx)
          return false;
        writeByte(wasm::RefType::Ref);
        writeSLEB128(static_cast<int64_t>(*contIdx));
      }
    } else {
      return false;
    }
    return true;
  }

  /// Write a heaptype immediate from an MLIR reference type.
  /// Returns true on success, false if the type is unsupported.
  bool writeHeapType(mlir::Type type, const IndexSpace *indexSpace = nullptr) {
    if (isa<FuncRefType>(type)) {
      writeByte(static_cast<uint8_t>(wasm::ValType::FuncRef));
      return true;
    }
    if (isa<ExternRefType>(type)) {
      writeByte(static_cast<uint8_t>(wasm::ValType::ExternRef));
      return true;
    }
    if (auto contRefType = dyn_cast<ContRefType>(type)) {
      if (!indexSpace) {
        writeByte(static_cast<uint8_t>(wasm::ValType::ContRef));
      } else {
        auto contIdx = indexSpace->tryGetContTypeIndex(
            contRefType.getTypeName().getValue());
        if (!contIdx)
          return false;
        writeSLEB128(static_cast<int64_t>(*contIdx));
      }
      return true;
    }
    if (auto contRefType = dyn_cast<ContRefNonNullType>(type)) {
      if (!indexSpace)
        return false;
      auto contIdx =
          indexSpace->tryGetContTypeIndex(contRefType.getTypeName().getValue());
      if (!contIdx)
        return false;
      writeSLEB128(static_cast<int64_t>(*contIdx));
      return true;
    }
    return false;
  }

  /// Write a complete section: sectionId + uleb128(size) + contents.
  /// The contents are provided as another BinaryWriter's buffer.
  void writeSection(wasm::SectionId id, const BinaryWriter &contents) {
    writeByte(static_cast<uint8_t>(id));
    writeULEB128(contents.size());
    writeBytes(contents.data(), contents.size());
  }

  /// Write a custom section with name.
  void writeCustomSection(llvm::StringRef name, const BinaryWriter &contents) {
    BinaryWriter payload;
    payload.writeString(name);
    payload.writeBytes(contents.data(), contents.size());
    writeSection(wasm::SectionId::Custom, payload);
  }

  /// Append another writer's buffer.
  void append(const BinaryWriter &other) {
    writeBytes(other.data(), other.size());
  }

  /// Get raw data pointer.
  const uint8_t *data() const { return buffer.data(); }

  /// Get current buffer size.
  size_t size() const { return buffer.size(); }

  /// Get current write offset (same as size).
  size_t offset() const { return buffer.size(); }

  /// Flush buffer contents to a raw_ostream.
  void flush(llvm::raw_ostream &os) const {
    os.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
  }

  /// Clear the buffer.
  void clear() { buffer.clear(); }

private:
  llvm::SmallVector<uint8_t, 1024> buffer;
};

} // namespace mlir::wasmstack

#endif // TARGET_WASMSTACK_BINARYWRITER_H
