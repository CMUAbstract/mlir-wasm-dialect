//===- IndexSpace.h - WebAssembly index space assignment ---------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_WASMSTACK_INDEXSPACE_H
#define TARGET_WASMSTACK_INDEXSPACE_H

#include "Target/WasmStack/WasmBinaryConstants.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir::wasmstack {

class ModuleOp;
class FuncOp;
class GlobalOp;
class MemoryOp;

/// Assigns WebAssembly index-space indices to module elements.
/// Type indices are de-duplicated; function, global, memory indices
/// are assigned in declaration order.
class IndexSpace {
public:
  /// A function signature for de-duplication in the type section.
  struct FuncSig {
    llvm::SmallVector<mlir::Type> params;
    llvm::SmallVector<mlir::Type> results;

    bool operator==(const FuncSig &other) const {
      return params == other.params && results == other.results;
    }
  };

  /// Analyze a WasmStack module and populate index spaces.
  void analyze(Operation *moduleOp);

  /// Get or create a type index for a function signature.
  uint32_t getOrCreateTypeIndex(const FuncSig &sig);

  /// Get type index for a function signature (must already exist).
  uint32_t getTypeIndex(const FuncSig &sig) const;

  /// Get function index by symbol name.
  uint32_t getFuncIndex(llvm::StringRef name) const;

  /// Try to get function index by symbol name.
  std::optional<uint32_t> tryGetFuncIndex(llvm::StringRef name) const;

  /// Get global index by symbol name.
  uint32_t getGlobalIndex(llvm::StringRef name) const;

  /// Get memory index by symbol name.
  uint32_t getMemoryIndex(llvm::StringRef name) const;

  /// Get all type signatures in order.
  const llvm::SmallVector<FuncSig> &getTypes() const { return types; }

  /// Number of function signatures that are emitted before continuation type
  /// declarations in the type section.
  uint32_t getPreContTypeCount() const { return preContTypeCount; }

  /// Get all function names in index order.
  const llvm::SmallVector<std::string> &getFuncNames() const {
    return funcNames;
  }

  /// Get all global names in index order.
  const llvm::SmallVector<std::string> &getGlobalNames() const {
    return globalNames;
  }

  /// Get all memory names in index order.
  const llvm::SmallVector<std::string> &getMemoryNames() const {
    return memoryNames;
  }

  /// Continuation type declaration information.
  struct ContTypeInfo {
    std::string name;
    uint32_t funcTypeIndex;
    uint32_t typeIndex;
  };

  /// Get wasm type index for a declared `wasmstack.type.cont`.
  uint32_t getContTypeIndex(llvm::StringRef name) const;

  /// Get all continuation declarations in index order.
  const llvm::SmallVector<ContTypeInfo> &getContTypes() const {
    return contTypes;
  }

  /// Get type index for a declared `wasmstack.type.func`.
  uint32_t getTypeFuncIndex(llvm::StringRef name) const;

  /// Get tag index by symbol name.
  uint32_t getTagIndex(llvm::StringRef name) const;

  /// Get all tag names in index order.
  const llvm::SmallVector<std::string> &getTagNames() const { return tagNames; }

  /// Get sorted function indices that require declarative `ref.func`
  /// declarations in the element section.
  const llvm::SmallVector<uint32_t> &getRefFuncDeclarationIndices() const {
    return refFuncDeclarationIndices;
  }

  /// Symbol information for the linking section symbol table.
  struct SymbolInfo {
    wasm::SymtabKind kind;
    std::string name;
    uint32_t elementIndex; // index in its wasm index space
    uint32_t flags;        // WASM_SYMBOL_* flags
    // Data symbols only:
    uint32_t segment = 0; // data segment index
    uint32_t offset = 0;  // offset within segment
    uint32_t size = 0;    // data size
  };

  /// Build the symbol table (call after analyze()).
  void buildSymbolTable(Operation *moduleOp);

  /// Get the symbol index for a given name.
  uint32_t getSymbolIndex(llvm::StringRef name) const;

  /// Get all symbols in order.
  const llvm::SmallVector<SymbolInfo> &getSymbols() const { return symbols; }

private:
  uint32_t toGlobalTypeIndex(uint32_t funcSigOrdinal) const;

  /// De-duplicated type signatures.
  llvm::SmallVector<FuncSig> types;

  /// Number of entries in `types` emitted before continuation declarations.
  uint32_t preContTypeCount = 0;

  /// Function name -> index mapping.
  llvm::StringMap<uint32_t> funcIndexMap;
  llvm::SmallVector<std::string> funcNames;

  /// Global name -> index mapping.
  llvm::StringMap<uint32_t> globalIndexMap;
  llvm::SmallVector<std::string> globalNames;

  /// Memory name -> index mapping.
  llvm::StringMap<uint32_t> memoryIndexMap;
  llvm::SmallVector<std::string> memoryNames;

  /// Function-type symbol -> type index mapping.
  llvm::StringMap<uint32_t> typeFuncIndexMap;

  /// Continuation-type symbol -> type index mapping.
  llvm::StringMap<uint32_t> contTypeIndexMap;
  llvm::SmallVector<ContTypeInfo> contTypes;

  /// Tag name -> index mapping.
  llvm::StringMap<uint32_t> tagIndexMap;
  llvm::SmallVector<std::string> tagNames;

  /// Unique sorted function indices referenced by wasmstack.ref.func.
  llvm::SmallVector<uint32_t> refFuncDeclarationIndices;

  /// Symbol table for linking section.
  llvm::SmallVector<SymbolInfo> symbols;
  llvm::StringMap<uint32_t> symbolIndexMap;
};

} // namespace mlir::wasmstack

#endif // TARGET_WASMSTACK_INDEXSPACE_H
