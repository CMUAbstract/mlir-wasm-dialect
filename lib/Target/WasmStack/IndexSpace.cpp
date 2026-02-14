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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::wasmstack {

LogicalResult IndexSpace::analyze(Operation *moduleOp) {
  // Clear previous state
  types.clear();
  funcIndexMap.clear();
  funcNames.clear();
  globalIndexMap.clear();
  globalNames.clear();
  memoryIndexMap.clear();
  memoryNames.clear();
  typeFuncIndexMap.clear();
  contTypeIndexMap.clear();
  contTypes.clear();
  preContTypeCount = 0;
  tagIndexMap.clear();
  tagNames.clear();
  refFuncDeclarationIndices.clear();

  auto addFunctionType = [&](FunctionType funcType) {
    FuncSig sig;
    for (Type t : funcType.getInputs())
      sig.params.push_back(t);
    for (Type t : funcType.getResults())
      sig.results.push_back(t);
    return getOrCreateTypeIndex(sig);
  };

  auto registerFunction = [&](llvm::StringRef symName, FunctionType funcType) {
    addFunctionType(funcType);

    uint32_t idx = funcNames.size();
    funcIndexMap[symName] = idx;
    funcNames.push_back(symName.str());
  };

  // Named function types.
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto typeFuncOp = dyn_cast<TypeFuncOp>(op)) {
      uint32_t typeIndex = addFunctionType(typeFuncOp.getType());
      typeFuncIndexMap[typeFuncOp.getSymName()] = typeIndex;
    }
  }

  // Imported functions are indexed before defined functions in wasm binaries.
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto importOp = dyn_cast<FuncImportOp>(op))
      registerFunction(importOp.getSymName(), importOp.getFuncType());
  }

  // Defined functions come after imports.
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      registerFunction(funcOp.getSymName(), funcOp.getFuncType());
    }
  }

  // Tags and their signature types.
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto tagOp = dyn_cast<TagOp>(op)) {
      addFunctionType(tagOp.getType());
      uint32_t idx = tagNames.size();
      tagIndexMap[tagOp.getSymName()] = idx;
      tagNames.push_back(tagOp.getSymName().str());
    }
  }

  // Globals and memories retain declaration order.
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto globalOp = dyn_cast<GlobalOp>(op)) {
      uint32_t idx = globalNames.size();
      globalIndexMap[globalOp.getSymName()] = idx;
      globalNames.push_back(globalOp.getSymName().str());
    } else if (auto memoryOp = dyn_cast<MemoryOp>(op)) {
      uint32_t idx = memoryNames.size();
      memoryIndexMap[memoryOp.getSymName()] = idx;
      memoryNames.push_back(memoryOp.getSymName().str());
    }
  }

  // Freeze base function-signature prefix. Continuations are emitted after this
  // prefix and before any additional control-flow signatures that may carry
  // typed continuation refs.
  preContTypeCount = types.size();

  // Continuation type declarations are appended after base function signatures.
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto contOp = dyn_cast<TypeContOp>(op)) {
      auto funcTypeRef = contOp->getAttrOfType<FlatSymbolRefAttr>("func_type");
      if (!funcTypeRef)
        return contOp.emitOpError("missing 'func_type' attribute");
      auto funcTypeIt = typeFuncIndexMap.find(funcTypeRef.getValue());
      if (funcTypeIt == typeFuncIndexMap.end()) {
        contOp.emitOpError("unknown wasmstack.type.func symbol ")
            << funcTypeRef;
        return failure();
      }

      uint32_t contTypeIndex = preContTypeCount + contTypes.size();
      contTypeIndexMap[contOp.getSymName()] = contTypeIndex;
      contTypes.push_back(ContTypeInfo{contOp.getSymName().str(),
                                       funcTypeIt->second, contTypeIndex});
    }
  }

  // Scan function bodies for multi-value control-flow signatures.
  // These are placed after continuation declarations so typed continuation
  // references in block signatures only point backward in the type section.
  auto collectControlSig = [&](ArrayAttr paramTypes, ArrayAttr resultTypes,
                               MLIRContext *ctx) {
    if (paramTypes.empty() && resultTypes.size() <= 1)
      return;
    SmallVector<Type> params;
    SmallVector<Type> results;
    params.reserve(paramTypes.size());
    results.reserve(resultTypes.size());
    for (Attribute a : paramTypes)
      params.push_back(cast<TypeAttr>(a).getValue());
    for (Attribute a : resultTypes)
      results.push_back(cast<TypeAttr>(a).getValue());
    addFunctionType(FunctionType::get(ctx, params, results));
  };

  for (Operation &op : moduleOp->getRegion(0).front()) {
    auto funcOp = dyn_cast<FuncOp>(op);
    if (!funcOp)
      continue;
    funcOp.getBody().walk([&](Operation *innerOp) {
      if (auto blockOp = dyn_cast<BlockOp>(innerOp)) {
        collectControlSig(blockOp.getParamTypes(), blockOp.getResultTypes(),
                          funcOp.getContext());
      } else if (auto loopOp = dyn_cast<LoopOp>(innerOp)) {
        collectControlSig(loopOp.getParamTypes(), loopOp.getResultTypes(),
                          funcOp.getContext());
      } else if (auto ifOp = dyn_cast<IfOp>(innerOp)) {
        collectControlSig(ifOp.getParamTypes(), ifOp.getResultTypes(),
                          funcOp.getContext());
      }
    });
  }

  // Collect all functions referenced by wasmstack.ref.func in emitted code.
  llvm::DenseSet<uint32_t> seenRefFuncIndices;
  auto collectRefFuncIndices = [&](Operation *root) {
    root->walk([&](RefFuncOp refFuncOp) {
      auto idx = tryGetFuncIndex(refFuncOp.getFuncAttr().getValue());
      if (!idx)
        return;
      if (seenRefFuncIndices.insert(*idx).second)
        refFuncDeclarationIndices.push_back(*idx);
    });
  };

  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      collectRefFuncIndices(funcOp.getOperation());
    } else if (auto globalOp = dyn_cast<GlobalOp>(op)) {
      collectRefFuncIndices(globalOp.getOperation());
    }
  }
  llvm::sort(refFuncDeclarationIndices);
  return success();
}

uint32_t IndexSpace::getOrCreateTypeIndex(const FuncSig &sig) {
  // Search for existing signature
  for (uint32_t i = 0; i < types.size(); ++i) {
    if (types[i] == sig)
      return toGlobalTypeIndex(i);
  }
  // Add new signature
  uint32_t idx = types.size();
  types.push_back(sig);
  return toGlobalTypeIndex(idx);
}

uint32_t IndexSpace::getTypeIndex(const FuncSig &sig) const {
  for (uint32_t i = 0; i < types.size(); ++i) {
    if (types[i] == sig)
      return toGlobalTypeIndex(i);
  }
  llvm_unreachable("type signature not found in index space");
}

uint32_t IndexSpace::toGlobalTypeIndex(uint32_t funcSigOrdinal) const {
  if (funcSigOrdinal < preContTypeCount)
    return funcSigOrdinal;
  return funcSigOrdinal + contTypes.size();
}

uint32_t IndexSpace::getFuncIndex(llvm::StringRef name) const {
  auto it = funcIndexMap.find(name);
  assert(it != funcIndexMap.end() && "function not found in index space");
  return it->second;
}

std::optional<uint32_t>
IndexSpace::tryGetFuncIndex(llvm::StringRef name) const {
  auto it = funcIndexMap.find(name);
  if (it == funcIndexMap.end())
    return std::nullopt;
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

uint32_t IndexSpace::getContTypeIndex(llvm::StringRef name) const {
  auto it = contTypeIndexMap.find(name);
  assert(it != contTypeIndexMap.end() &&
         "continuation type not found in index space");
  return it->second;
}

uint32_t IndexSpace::getTypeFuncIndex(llvm::StringRef name) const {
  auto it = typeFuncIndexMap.find(name);
  assert(it != typeFuncIndexMap.end() &&
         "function type not found in index space");
  return it->second;
}

uint32_t IndexSpace::getTagIndex(llvm::StringRef name) const {
  auto it = tagIndexMap.find(name);
  assert(it != tagIndexMap.end() && "tag not found in index space");
  return it->second;
}

void IndexSpace::buildSymbolTable(Operation *moduleOp) {
  symbols.clear();
  symbolIndexMap.clear();

  // Imported functions first.
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto importOp = dyn_cast<FuncImportOp>(op)) {
      SymbolInfo sym;
      sym.kind = wasm::SymtabKind::Function;
      sym.name = importOp.getSymName().str();
      sym.elementIndex = getFuncIndex(importOp.getSymName());
      sym.flags = wasm::WASM_SYMBOL_UNDEFINED | wasm::WASM_SYMBOL_EXPLICIT_NAME;
      uint32_t idx = symbols.size();
      symbolIndexMap[sym.name] = idx;
      symbols.push_back(std::move(sym));
    }
  }

  // Defined functions second.
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
