//===- IndexSpace.cpp - WebAssembly index space assignment -------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target/WasmStack/IndexSpace.h"
#include "wasmstack/WasmStackOps.h"

namespace mlir::wasmstack {

void IndexSpace::analyze(Operation *moduleOp) {
  // Clear previous state
  types.clear();
  funcIndexMap.clear();
  funcNames.clear();
  globalIndexMap.clear();
  globalNames.clear();
  memoryIndexMap.clear();
  memoryNames.clear();

  // Collect all operations in the module
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      // Register the function signature as a type
      FuncSig sig;
      FunctionType funcType = funcOp.getFuncType();
      for (Type t : funcType.getInputs())
        sig.params.push_back(t);
      for (Type t : funcType.getResults())
        sig.results.push_back(t);
      getOrCreateTypeIndex(sig);

      // Register function index
      uint32_t idx = funcNames.size();
      funcIndexMap[funcOp.getSymName()] = idx;
      funcNames.push_back(funcOp.getSymName().str());

      // Scan function body for block types that need type section entries
      funcOp.getBody().walk([&](Operation *innerOp) {
        if (auto blockOp = dyn_cast<BlockOp>(innerOp)) {
          auto paramTypes = blockOp.getParamTypes();
          auto resultTypes = blockOp.getResultTypes();
          // Multi-value blocks need a type index
          if (!paramTypes.empty() || resultTypes.size() > 1) {
            FuncSig blockSig;
            for (Attribute a : paramTypes)
              blockSig.params.push_back(cast<TypeAttr>(a).getValue());
            for (Attribute a : resultTypes)
              blockSig.results.push_back(cast<TypeAttr>(a).getValue());
            getOrCreateTypeIndex(blockSig);
          }
        } else if (auto loopOp = dyn_cast<LoopOp>(innerOp)) {
          auto paramTypes = loopOp.getParamTypes();
          auto resultTypes = loopOp.getResultTypes();
          if (!paramTypes.empty() || resultTypes.size() > 1) {
            FuncSig blockSig;
            for (Attribute a : paramTypes)
              blockSig.params.push_back(cast<TypeAttr>(a).getValue());
            for (Attribute a : resultTypes)
              blockSig.results.push_back(cast<TypeAttr>(a).getValue());
            getOrCreateTypeIndex(blockSig);
          }
        } else if (auto ifOp = dyn_cast<IfOp>(innerOp)) {
          auto paramTypes = ifOp.getParamTypes();
          auto resultTypes = ifOp.getResultTypes();
          if (!paramTypes.empty() || resultTypes.size() > 1) {
            FuncSig blockSig;
            for (Attribute a : paramTypes)
              blockSig.params.push_back(cast<TypeAttr>(a).getValue());
            for (Attribute a : resultTypes)
              blockSig.results.push_back(cast<TypeAttr>(a).getValue());
            getOrCreateTypeIndex(blockSig);
          }
        }
      });
    } else if (auto globalOp = dyn_cast<GlobalOp>(op)) {
      uint32_t idx = globalNames.size();
      globalIndexMap[globalOp.getSymName()] = idx;
      globalNames.push_back(globalOp.getSymName().str());
    } else if (auto memoryOp = dyn_cast<MemoryOp>(op)) {
      uint32_t idx = memoryNames.size();
      memoryIndexMap[memoryOp.getSymName()] = idx;
      memoryNames.push_back(memoryOp.getSymName().str());
    }
  }
}

uint32_t IndexSpace::getOrCreateTypeIndex(const FuncSig &sig) {
  // Search for existing signature
  for (uint32_t i = 0; i < types.size(); ++i) {
    if (types[i] == sig)
      return i;
  }
  // Add new signature
  uint32_t idx = types.size();
  types.push_back(sig);
  return idx;
}

uint32_t IndexSpace::getTypeIndex(const FuncSig &sig) const {
  for (uint32_t i = 0; i < types.size(); ++i) {
    if (types[i] == sig)
      return i;
  }
  llvm_unreachable("type signature not found in index space");
}

uint32_t IndexSpace::getFuncIndex(llvm::StringRef name) const {
  auto it = funcIndexMap.find(name);
  assert(it != funcIndexMap.end() && "function not found in index space");
  return it->second;
}

uint32_t IndexSpace::getGlobalIndex(llvm::StringRef name) const {
  auto it = globalIndexMap.find(name);
  assert(it != globalIndexMap.end() && "global not found in index space");
  return it->second;
}

uint32_t IndexSpace::getMemoryIndex(llvm::StringRef name) const {
  auto it = memoryIndexMap.find(name);
  assert(it != memoryIndexMap.end() && "memory not found in index space");
  return it->second;
}

void IndexSpace::buildSymbolTable(Operation *moduleOp) {
  symbols.clear();
  symbolIndexMap.clear();

  // Functions first
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      SymbolInfo sym;
      sym.kind = wasm::SymtabKind::Function;
      sym.name = funcOp.getSymName().str();
      sym.elementIndex = getFuncIndex(funcOp.getSymName());
      sym.flags = 0;
      if (funcOp.getExportName())
        sym.flags |= wasm::WASM_SYMBOL_EXPORTED;
      uint32_t idx = symbols.size();
      symbolIndexMap[sym.name] = idx;
      symbols.push_back(std::move(sym));
    }
  }

  // Globals second
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto globalOp = dyn_cast<GlobalOp>(op)) {
      SymbolInfo sym;
      sym.kind = wasm::SymtabKind::Global;
      sym.name = globalOp.getSymName().str();
      sym.elementIndex = getGlobalIndex(globalOp.getSymName());
      sym.flags = 0;
      if (globalOp.getExportName())
        sym.flags |= wasm::WASM_SYMBOL_EXPORTED;
      uint32_t idx = symbols.size();
      symbolIndexMap[sym.name] = idx;
      symbols.push_back(std::move(sym));
    }
  }

  // Data segments third
  uint32_t segmentIndex = 0;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto dataOp = dyn_cast<DataOp>(op)) {
      SymbolInfo sym;
      sym.kind = wasm::SymtabKind::Data;
      // Use a synthetic name for data segments
      sym.name = (".data." + llvm::Twine(segmentIndex)).str();
      sym.elementIndex = 0;
      sym.flags = wasm::WASM_SYMBOL_BINDING_LOCAL;
      sym.segment = segmentIndex;
      sym.offset = 0;
      sym.size = dataOp.getData().size();
      uint32_t idx = symbols.size();
      symbolIndexMap[sym.name] = idx;
      symbols.push_back(std::move(sym));
      segmentIndex++;
    }
  }
}

uint32_t IndexSpace::getSymbolIndex(llvm::StringRef name) const {
  auto it = symbolIndexMap.find(name);
  assert(it != symbolIndexMap.end() && "symbol not found in symbol table");
  return it->second;
}

} // namespace mlir::wasmstack
