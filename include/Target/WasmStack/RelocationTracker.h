//===- RelocationTracker.h - Track relocations for object files --*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_WASMSTACK_RELOCATIONTRACKER_H
#define TARGET_WASMSTACK_RELOCATIONTRACKER_H

#include "Target/WasmStack/WasmBinaryConstants.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::wasmstack {

/// A single relocation entry for a wasm object file.
struct RelocationEntry {
  uint8_t type;    // R_WASM_* constant (RelocType)
  uint32_t offset; // Byte offset within section contents
  uint32_t index;  // Symbol index in linking symbol table
  int32_t addend;  // For R_WASM_MEMORY_ADDR_* types

  RelocationEntry(uint8_t type, uint32_t offset, uint32_t index,
                  int32_t addend = 0)
      : type(type), offset(offset), index(index), addend(addend) {}
};

/// Returns true if the given relocation type uses an addend field.
inline bool relocTypeHasAddend(uint8_t type) {
  return type ==
             static_cast<uint8_t>(wasm::RelocType::R_WASM_MEMORY_ADDR_SLEB) ||
         type == static_cast<uint8_t>(wasm::RelocType::R_WASM_MEMORY_ADDR_I32);
}

/// Tracks relocation entries during code and data section emission.
class RelocationTracker {
public:
  void addCodeRelocation(uint8_t type, uint32_t offset, uint32_t symbolIndex,
                         int32_t addend = 0) {
    codeRelocs.emplace_back(type, offset, symbolIndex, addend);
  }

  void addDataRelocation(uint8_t type, uint32_t offset, uint32_t symbolIndex,
                         int32_t addend = 0) {
    dataRelocs.emplace_back(type, offset, symbolIndex, addend);
  }

  const llvm::SmallVector<RelocationEntry> &getCodeRelocations() const {
    return codeRelocs;
  }

  const llvm::SmallVector<RelocationEntry> &getDataRelocations() const {
    return dataRelocs;
  }

private:
  llvm::SmallVector<RelocationEntry> codeRelocs;
  llvm::SmallVector<RelocationEntry> dataRelocs;
};

} // namespace mlir::wasmstack

#endif // TARGET_WASMSTACK_RELOCATIONTRACKER_H
