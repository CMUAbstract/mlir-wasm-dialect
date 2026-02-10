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

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

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

  /// Get global index by symbol name.
  uint32_t getGlobalIndex(llvm::StringRef name) const;

  /// Get memory index by symbol name.
  uint32_t getMemoryIndex(llvm::StringRef name) const;

  /// Get all type signatures in order.
  const llvm::SmallVector<FuncSig> &getTypes() const { return types; }

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

private:
  /// De-duplicated type signatures.
  llvm::SmallVector<FuncSig> types;

  /// Function name -> index mapping.
  llvm::StringMap<uint32_t> funcIndexMap;
  llvm::SmallVector<std::string> funcNames;

  /// Global name -> index mapping.
  llvm::StringMap<uint32_t> globalIndexMap;
  llvm::SmallVector<std::string> globalNames;

  /// Memory name -> index mapping.
  llvm::StringMap<uint32_t> memoryIndexMap;
  llvm::SmallVector<std::string> memoryNames;
};

} // namespace mlir::wasmstack

#endif // TARGET_WASMSTACK_INDEXSPACE_H
