//===- WasmBinaryEmitter.cpp - Top-level binary emitter ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target/WasmStack/WasmBinaryEmitter.h"
#include "Target/WasmStack/BinaryWriter.h"
#include "Target/WasmStack/IndexSpace.h"
#include "Target/WasmStack/OpcodeEmitter.h"
#include "Target/WasmStack/RelocationTracker.h"
#include "Target/WasmStack/WasmBinaryConstants.h"
#include "wasmstack/WasmStackOps.h"

using namespace mlir;
using namespace mlir::wasmstack;
namespace wc = mlir::wasmstack::wasm;

//===----------------------------------------------------------------------===//
// Section emitters
//===----------------------------------------------------------------------===//

static IndexSpace::FuncSig toFuncSig(FunctionType funcType) {
  IndexSpace::FuncSig sig;
  for (Type t : funcType.getInputs())
    sig.params.push_back(t);
  for (Type t : funcType.getResults())
    sig.results.push_back(t);
  return sig;
}

/// Emit the type section (section 1).
static LogicalResult emitTypeSection(BinaryWriter &output,
                                     IndexSpace &indexSpace,
                                     Operation *moduleOp) {
  BinaryWriter section;
  const auto &funcTypes = indexSpace.getTypes();
  const auto &contTypes = indexSpace.getContTypes();
  uint32_t preContCount = indexSpace.getPreContTypeCount();
  section.writeULEB128(funcTypes.size() + contTypes.size());

  auto emitFuncSig = [&](const IndexSpace::FuncSig &sig) -> LogicalResult {
    section.writeByte(wc::FuncTypeTag); // 0x60
    // Params
    section.writeULEB128(sig.params.size());
    for (Type t : sig.params) {
      if (!section.writeValType(t, &indexSpace)) {
        moduleOp->emitError("unsupported value type in wasm type signature: ")
            << t;
        return failure();
      }
    }
    // Results
    section.writeULEB128(sig.results.size());
    for (Type t : sig.results) {
      if (!section.writeValType(t, &indexSpace)) {
        moduleOp->emitError("unsupported result type in wasm type signature: ")
            << t;
        return failure();
      }
    }
    return success();
  };

  for (uint32_t i = 0; i < preContCount; ++i) {
    if (failed(emitFuncSig(funcTypes[i])))
      return failure();
  }

  for (const auto &cont : contTypes) {
    section.writeByte(0x5D); // continuation type constructor
    section.writeSLEB128(static_cast<int32_t>(cont.funcTypeIndex));
  }

  for (uint32_t i = preContCount; i < funcTypes.size(); ++i) {
    if (failed(emitFuncSig(funcTypes[i])))
      return failure();
  }

  output.writeSection(wc::SectionId::Type, section);
  return success();
}

/// Emit the import section (section 2).
static LogicalResult emitImportSection(BinaryWriter &output,
                                       IndexSpace &indexSpace,
                                       Operation *moduleOp) {
  SmallVector<FuncImportOp> imports;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto importOp = dyn_cast<FuncImportOp>(op))
      imports.push_back(importOp);
  }

  if (imports.empty())
    return success();

  BinaryWriter section;
  section.writeULEB128(imports.size());
  for (auto importOp : imports) {
    section.writeString(importOp.getModuleName());
    section.writeString(importOp.getImportName());
    section.writeByte(static_cast<uint8_t>(wc::ImportKind::Func));

    IndexSpace::FuncSig sig;
    FunctionType funcType = importOp.getFuncType();
    for (Type t : funcType.getInputs())
      sig.params.push_back(t);
    for (Type t : funcType.getResults())
      sig.results.push_back(t);
    auto typeIdx = indexSpace.tryGetTypeIndex(sig);
    if (!typeIdx) {
      importOp.emitOpError("import signature was not indexed in type section");
      return failure();
    }
    section.writeULEB128(*typeIdx);
  }

  output.writeSection(wc::SectionId::Import, section);
  return success();
}

/// Emit the function section (section 3) - maps function index to type index.
static LogicalResult emitFunctionSection(BinaryWriter &output,
                                         IndexSpace &indexSpace,
                                         Operation *moduleOp) {
  BinaryWriter section;

  // Count defined functions
  SmallVector<FuncOp> funcs;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto funcOp = dyn_cast<FuncOp>(op))
      funcs.push_back(funcOp);
  }

  section.writeULEB128(funcs.size());
  for (auto funcOp : funcs) {
    IndexSpace::FuncSig sig;
    FunctionType funcType = funcOp.getFuncType();
    for (Type t : funcType.getInputs())
      sig.params.push_back(t);
    for (Type t : funcType.getResults())
      sig.results.push_back(t);
    auto typeIdx = indexSpace.tryGetTypeIndex(sig);
    if (!typeIdx) {
      funcOp.emitOpError("function signature was not indexed in type section");
      return failure();
    }
    section.writeULEB128(*typeIdx);
  }

  output.writeSection(wc::SectionId::Function, section);
  return success();
}

/// Returns true when relocatable emission needs a synthetic table section.
static bool needsSyntheticTable(Operation *moduleOp, IndexSpace &indexSpace,
                                bool relocatable) {
  if (!relocatable)
    return false;

  if (!indexSpace.getRefFuncDeclarationIndices().empty())
    return true;

  bool hasCallIndirect = false;
  moduleOp->walk([&](CallIndirectOp) { hasCallIndirect = true; });
  return hasCallIndirect;
}

/// Emit a synthetic table section with table 0 for relocatable ref.func users.
static void emitTableSection(BinaryWriter &output, bool emitSyntheticTable) {
  if (!emitSyntheticTable)
    return;

  BinaryWriter section;
  section.writeULEB128(1); // table count
  section.writeByte(static_cast<uint8_t>(wc::ValType::FuncRef));
  section.writeByte(static_cast<uint8_t>(wc::LimitKind::Min));
  section.writeULEB128(0); // min table size
  output.writeSection(wc::SectionId::Table, section);
}

/// Emit the memory section (section 5).
static void emitMemorySection(BinaryWriter &output, Operation *moduleOp) {
  SmallVector<MemoryOp> memories;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto memOp = dyn_cast<MemoryOp>(op))
      memories.push_back(memOp);
  }

  if (memories.empty())
    return;

  BinaryWriter section;
  section.writeULEB128(memories.size());
  for (auto memOp : memories) {
    if (memOp.getMaxPages()) {
      section.writeByte(static_cast<uint8_t>(wc::LimitKind::MinMax));
      section.writeULEB128(memOp.getMinPages());
      section.writeULEB128(*memOp.getMaxPages());
    } else {
      section.writeByte(static_cast<uint8_t>(wc::LimitKind::Min));
      section.writeULEB128(memOp.getMinPages());
    }
  }
  output.writeSection(wc::SectionId::Memory, section);
}

/// Emit the tag section.
static LogicalResult emitTagSection(BinaryWriter &output,
                                    IndexSpace &indexSpace,
                                    Operation *moduleOp) {
  SmallVector<TagOp> tags;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto tagOp = dyn_cast<TagOp>(op))
      tags.push_back(tagOp);
  }

  if (tags.empty())
    return success();

  BinaryWriter section;
  section.writeULEB128(tags.size());
  for (auto tagOp : tags) {
    section.writeByte(0x00); // attribute
    auto typeIndex = indexSpace.tryGetTypeIndex(toFuncSig(tagOp.getType()));
    if (!typeIndex) {
      tagOp.emitOpError("tag signature was not indexed in type section");
      return failure();
    }
    section.writeULEB128(*typeIndex);
  }

  output.writeSection(wc::SectionId::Tag, section);
  return success();
}

/// Emit the element section with declarative ref.func declarations.
static void emitElementSection(BinaryWriter &output, IndexSpace &indexSpace) {
  const auto &declaredFuncs = indexSpace.getRefFuncDeclarationIndices();
  if (declaredFuncs.empty())
    return;

  BinaryWriter section;
  section.writeULEB128(1); // one declarative segment
  section.writeULEB128(3); // flags=3: declarative, elemkind + funcidx vector
  section.writeByte(0x00); // elemkind=funcref
  section.writeULEB128(declaredFuncs.size());
  for (uint32_t idx : declaredFuncs)
    section.writeULEB128(idx);

  output.writeSection(wc::SectionId::Element, section);
}

/// Emit the global section (section 6).
static LogicalResult emitGlobalSection(BinaryWriter &output,
                                       IndexSpace &indexSpace,
                                       Operation *moduleOp) {
  SmallVector<GlobalOp> globals;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto globalOp = dyn_cast<GlobalOp>(op))
      globals.push_back(globalOp);
  }

  if (globals.empty())
    return success();

  BinaryWriter section;
  section.writeULEB128(globals.size());
  for (auto globalOp : globals) {
    // Global type: valtype + mutability
    if (!section.writeValType(globalOp.getTypeAttr().getValue(), &indexSpace)) {
      globalOp.emitOpError(
          "unsupported global value type for binary encoding: ")
          << globalOp.getTypeAttr().getValue();
      return failure();
    }
    section.writeByte(globalOp.getIsMutable()
                          ? static_cast<uint8_t>(wc::Mutability::Var)
                          : static_cast<uint8_t>(wc::Mutability::Const));

    // Init expression: emit the operations in the init region
    OpcodeEmitter emitter(section, indexSpace);
    for (Operation &initOp : globalOp.getInit().front()) {
      if (!emitter.emitOperation(&initOp))
        return failure();
    }
    section.writeByte(wc::Opcode::End);
  }
  output.writeSection(wc::SectionId::Global, section);
  return success();
}

/// Emit the export section (section 7).
static LogicalResult emitExportSection(BinaryWriter &output,
                                       IndexSpace &indexSpace,
                                       Operation *moduleOp) {
  BinaryWriter section;

  // Collect exports
  SmallVector<std::tuple<StringRef, wc::ExportKind, uint32_t>> exports;

  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      if (auto exportName = funcOp.getExportName()) {
        auto idx = indexSpace.tryGetFuncIndex(funcOp.getSymName());
        if (!idx) {
          funcOp.emitOpError("missing function index for exported symbol");
          return failure();
        }
        exports.push_back({*exportName, wc::ExportKind::Func, *idx});
      }
    } else if (auto memOp = dyn_cast<MemoryOp>(op)) {
      if (auto exportName = memOp.getExportName()) {
        auto idx = indexSpace.tryGetMemoryIndex(memOp.getSymName());
        if (!idx) {
          memOp.emitOpError("missing memory index for exported symbol");
          return failure();
        }
        exports.push_back({*exportName, wc::ExportKind::Memory, *idx});
      }
    } else if (auto globalOp = dyn_cast<GlobalOp>(op)) {
      if (auto exportName = globalOp.getExportName()) {
        auto idx = indexSpace.tryGetGlobalIndex(globalOp.getSymName());
        if (!idx) {
          globalOp.emitOpError("missing global index for exported symbol");
          return failure();
        }
        exports.push_back({*exportName, wc::ExportKind::Global, *idx});
      }
    } else if (auto tagOp = dyn_cast<TagOp>(op)) {
      if (auto exportName = tagOp.getExportName()) {
        auto idx = indexSpace.tryGetTagIndex(tagOp.getSymName());
        if (!idx) {
          tagOp.emitOpError("missing tag index for exported symbol");
          return failure();
        }
        exports.push_back({*exportName, wc::ExportKind::Tag, *idx});
      }
    }
  }

  if (exports.empty())
    return success();

  section.writeULEB128(exports.size());
  for (auto &[name, kind, idx] : exports) {
    section.writeString(name);
    section.writeByte(static_cast<uint8_t>(kind));
    section.writeULEB128(idx);
  }

  output.writeSection(wc::SectionId::Export, section);
  return success();
}

/// Emit the data count section (section 12).
static void emitDataCountSection(BinaryWriter &output, Operation *moduleOp) {
  uint32_t count = 0;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (isa<DataOp>(op))
      count++;
  }
  if (count == 0)
    return;

  BinaryWriter section;
  section.writeULEB128(count);
  output.writeSection(wc::SectionId::DataCount, section);
}

/// Emit the code section (section 10).
/// When tracker is non-null, uses fixed-width LEB128 and records relocations.
static LogicalResult emitCodeSection(BinaryWriter &output,
                                     IndexSpace &indexSpace,
                                     Operation *moduleOp,
                                     RelocationTracker *tracker) {
  SmallVector<FuncOp> funcs;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto funcOp = dyn_cast<FuncOp>(op))
      funcs.push_back(funcOp);
  }

  BinaryWriter section;
  section.writeULEB128(funcs.size());

  for (auto funcOp : funcs) {
    BinaryWriter funcBody;

    // Local declarations: collect LocalOp declarations (not parameters).
    // Group consecutive locals of the same type for the preamble.
    SmallVector<std::pair<uint32_t, Type>> localGroups;
    for (Operation &op : funcOp.getBody().front()) {
      if (auto localOp = dyn_cast<LocalOp>(op)) {
        Type localType = localOp.getTypeAttr().getValue();
        if (!localGroups.empty() && localGroups.back().second == localType) {
          localGroups.back().first++;
        } else {
          localGroups.push_back({1, localType});
        }
      }
    }

    funcBody.writeULEB128(localGroups.size());
    for (auto &[count, type] : localGroups) {
      funcBody.writeULEB128(count);
      if (!funcBody.writeValType(type, &indexSpace)) {
        funcOp.emitOpError("unsupported local type for binary encoding: ")
            << type;
        return failure();
      }
    }

    // The sectionOffset for relocations is the current position in the section
    // buffer where the function body bytes will land. This is after the
    // func count prefix and all previous (size_prefix + body) entries.
    uint32_t sectionOffset = section.size();

    // Emit function body opcodes
    OpcodeEmitter emitter(funcBody, indexSpace, tracker, sectionOffset);
    if (!emitter.emitFunctionBody(funcOp))
      return failure();

    // Write function body with size prefix
    section.writeULEB128(funcBody.size());

    // Now adjust relocation offsets: they were recorded relative to the
    // start of section buffer at the time of funcBody emission, but the
    // size prefix was written after. We need to account for the size prefix.
    // The actual offset within the section = sectionOffset was computed
    // before the size prefix. The size prefix bytes are now between
    // sectionOffset and section.size(). So we add the delta.
    if (tracker) {
      uint32_t sizePrefixBytes = section.size() - sectionOffset;
      auto &relocs = const_cast<SmallVector<RelocationEntry> &>(
          tracker->getCodeRelocations());
      for (auto &r : relocs) {
        if (r.offset >= sectionOffset &&
            r.offset < sectionOffset + funcBody.size()) {
          r.offset += sizePrefixBytes;
        }
      }
    }

    section.append(funcBody);
  }

  output.writeSection(wc::SectionId::Code, section);
  return success();
}

/// Emit the data section (section 11).
/// When tracker is non-null, uses placeholder offsets with relocations.
static LogicalResult emitDataSection(BinaryWriter &output,
                                     IndexSpace &indexSpace,
                                     Operation *moduleOp,
                                     RelocationTracker *tracker) {
  SmallVector<DataOp> dataOps;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto dataOp = dyn_cast<DataOp>(op))
      dataOps.push_back(dataOp);
  }

  if (dataOps.empty())
    return success();

  BinaryWriter section;
  section.writeULEB128(dataOps.size());

  uint32_t segmentIndex = 0;
  for (auto dataOp : dataOps) {
    // Active data segment: flags=0, offset expr, data bytes
    section.writeULEB128(0); // flags: active, memory 0
    // Offset init expression: i32.const <offset>
    section.writeByte(wc::Opcode::I32Const);
    if (tracker) {
      // Record relocation for the data segment offset
      std::string symName = (".data." + llvm::Twine(segmentIndex)).str();
      auto symIdx = indexSpace.tryGetSymbolIndex(symName);
      if (!symIdx) {
        dataOp.emitOpError("missing relocation symbol for data segment ")
            << "'" << symName << "'";
        return failure();
      }
      tracker->addDataRelocation(
          static_cast<uint8_t>(wc::RelocType::R_WASM_MEMORY_ADDR_SLEB),
          section.offset(), *symIdx, dataOp.getOffset());
      section.writeFixedSLEB128(0); // placeholder
    } else {
      section.writeSLEB128(dataOp.getOffset());
    }
    section.writeByte(wc::Opcode::End);
    // Data bytes
    StringRef data = dataOp.getData();
    section.writeULEB128(data.size());
    section.writeBytes(reinterpret_cast<const uint8_t *>(data.data()),
                       data.size());
    segmentIndex++;
  }

  output.writeSection(wc::SectionId::Data, section);
  return success();
}

//===----------------------------------------------------------------------===//
// Linking and relocation sections (relocatable mode only)
//===----------------------------------------------------------------------===//

/// Emit the "linking" custom section.
static void emitLinkingSection(BinaryWriter &output, IndexSpace &indexSpace,
                               Operation *moduleOp,
                               bool emitSyntheticTableSymbol) {
  BinaryWriter section;

  // Linking metadata version
  section.writeULEB128(wc::WasmMetadataVersion);

  // WASM_SYMBOL_TABLE subsection
  {
    BinaryWriter subsection;
    const auto &symbols = indexSpace.getSymbols();
    uint32_t symbolCount = symbols.size() + (emitSyntheticTableSymbol ? 1 : 0);
    subsection.writeULEB128(symbolCount);

    for (const auto &sym : symbols) {
      subsection.writeByte(static_cast<uint8_t>(sym.kind));
      subsection.writeULEB128(sym.flags);

      if (sym.kind == wasm::SymtabKind::Function ||
          sym.kind == wasm::SymtabKind::Global ||
          sym.kind == wasm::SymtabKind::Tag ||
          sym.kind == wasm::SymtabKind::Table) {
        subsection.writeULEB128(sym.elementIndex);
        // Write name if the symbol has WASM_SYMBOL_EXPLICIT_NAME flag
        // or if it's a defined symbol (always write names for our symbols)
        subsection.writeString(sym.name);
      } else if (sym.kind == wasm::SymtabKind::Data) {
        subsection.writeString(sym.name);
        // Data symbols also carry segment info
        subsection.writeULEB128(sym.segment);
        subsection.writeULEB128(sym.offset);
        subsection.writeULEB128(sym.size);
      }
    }

    if (emitSyntheticTableSymbol) {
      subsection.writeByte(static_cast<uint8_t>(wasm::SymtabKind::Table));
      subsection.writeULEB128(0); // defined, default visibility/binding flags
      subsection.writeULEB128(0); // table index 0
      subsection.writeString("__wasmstack.table0");
    }

    section.writeByte(
        static_cast<uint8_t>(wc::LinkingSubsection::WASM_SYMBOL_TABLE));
    section.writeULEB128(subsection.size());
    section.append(subsection);
  }

  // WASM_SEGMENT_INFO subsection (data segment names and properties)
  {
    SmallVector<DataOp> dataOps;
    for (Operation &op : moduleOp->getRegion(0).front()) {
      if (auto dataOp = dyn_cast<DataOp>(op))
        dataOps.push_back(dataOp);
    }

    if (!dataOps.empty()) {
      BinaryWriter subsection;
      subsection.writeULEB128(dataOps.size());
      for (uint32_t i = 0; i < dataOps.size(); ++i) {
        std::string segName = (".data." + llvm::Twine(i)).str();
        subsection.writeString(segName);
        subsection.writeULEB128(0); // alignment (p2align = 0 -> byte aligned)
        subsection.writeULEB128(0); // flags
      }

      section.writeByte(
          static_cast<uint8_t>(wc::LinkingSubsection::WASM_SEGMENT_INFO));
      section.writeULEB128(subsection.size());
      section.append(subsection);
    }
  }

  output.writeCustomSection("linking", section);
}

/// Emit a "reloc.*" custom section for a set of relocations.
static void emitRelocSection(BinaryWriter &output, llvm::StringRef name,
                             uint32_t sectionIndex,
                             const SmallVector<RelocationEntry> &relocs) {
  if (relocs.empty())
    return;

  BinaryWriter section;
  section.writeULEB128(sectionIndex); // target section index
  section.writeULEB128(relocs.size());

  for (const auto &r : relocs) {
    section.writeByte(r.type);
    section.writeULEB128(r.offset);
    section.writeULEB128(r.index);
    if (relocTypeHasAddend(r.type))
      section.writeSLEB128(r.addend);
  }

  output.writeCustomSection(name, section);
}

/// Emit the "target_features" custom section declaring mutable-globals.
static void emitTargetFeaturesSection(BinaryWriter &output) {
  BinaryWriter section;
  section.writeULEB128(1); // feature count
  section.writeByte(0x2B); // '+' prefix = feature used
  section.writeString("mutable-globals");
  output.writeCustomSection("target_features", section);
}

//===----------------------------------------------------------------------===//
// Top-level emitter
//===----------------------------------------------------------------------===//

LogicalResult mlir::wasmstack::emitWasmBinary(Operation *op,
                                              llvm::raw_ostream &output,
                                              bool relocatable) {
  wasmstack::ModuleOp wasmModule;
  if (auto directModule = dyn_cast<wasmstack::ModuleOp>(op)) {
    wasmModule = directModule;
  } else if (auto topModule = dyn_cast<mlir::ModuleOp>(op)) {
    SmallVector<wasmstack::ModuleOp> nestedModules;
    for (auto candidate : topModule.getOps<wasmstack::ModuleOp>())
      nestedModules.push_back(candidate);
    if (nestedModules.empty()) {
      op->emitError("expected one top-level wasmstack.module, found none");
      return failure();
    }
    if (nestedModules.size() > 1) {
      op->emitError("expected one top-level wasmstack.module, found ")
          << nestedModules.size();
      return failure();
    }
    wasmModule = nestedModules.front();
  } else {
    op->emitError("emitWasmBinary expects input op to be module or "
                  "wasmstack.module");
    return failure();
  }

  // 1. Analyze index spaces
  IndexSpace indexSpace;
  if (failed(indexSpace.analyze(wasmModule.getOperation())))
    return failure();

  // Build symbol table for relocatable mode
  RelocationTracker tracker;
  RelocationTracker *trackerPtr = relocatable ? &tracker : nullptr;
  if (relocatable)
    indexSpace.buildSymbolTable(wasmModule.getOperation());

  // 2. Build output
  BinaryWriter writer;
  bool emitSyntheticTable =
      needsSyntheticTable(wasmModule.getOperation(), indexSpace, relocatable);
  uint32_t sectionIndex = 0;
  std::optional<uint32_t> codeSectionIndex;
  std::optional<uint32_t> dataSectionIndex;
  auto noteSectionEmission = [&](size_t beforeSize,
                                 std::optional<uint32_t> *outIndex = nullptr) {
    if (writer.size() == beforeSize)
      return;
    if (outIndex)
      *outIndex = sectionIndex;
    ++sectionIndex;
  };

  // Magic number and version
  writer.writeBytes(wc::Magic, sizeof(wc::Magic));
  writer.writeBytes(wc::Version, sizeof(wc::Version));

  // Emit sections in order.
  size_t beforeSize = writer.size();
  if (failed(emitTypeSection(writer, indexSpace, wasmModule.getOperation())))
    return failure();
  noteSectionEmission(beforeSize);

  beforeSize = writer.size();
  if (failed(emitImportSection(writer, indexSpace, wasmModule.getOperation())))
    return failure();
  noteSectionEmission(beforeSize);

  beforeSize = writer.size();
  if (failed(
          emitFunctionSection(writer, indexSpace, wasmModule.getOperation())))
    return failure();
  noteSectionEmission(beforeSize);

  beforeSize = writer.size();
  emitTableSection(writer, emitSyntheticTable);
  noteSectionEmission(beforeSize);

  beforeSize = writer.size();
  emitMemorySection(writer, wasmModule.getOperation());
  noteSectionEmission(beforeSize);

  beforeSize = writer.size();
  if (failed(emitTagSection(writer, indexSpace, wasmModule.getOperation())))
    return failure();
  noteSectionEmission(beforeSize);

  beforeSize = writer.size();
  if (failed(emitGlobalSection(writer, indexSpace, wasmModule.getOperation())))
    return failure();
  noteSectionEmission(beforeSize);

  // Export section: skip in relocatable mode (exports become symbol flags)
  if (!relocatable) {
    beforeSize = writer.size();
    if (failed(
            emitExportSection(writer, indexSpace, wasmModule.getOperation())))
      return failure();
    noteSectionEmission(beforeSize);
  }

  beforeSize = writer.size();
  emitElementSection(writer, indexSpace);
  noteSectionEmission(beforeSize);

  beforeSize = writer.size();
  emitDataCountSection(writer, wasmModule.getOperation());
  noteSectionEmission(beforeSize);

  beforeSize = writer.size();
  if (failed(emitCodeSection(writer, indexSpace, wasmModule.getOperation(),
                             trackerPtr)))
    return failure();
  noteSectionEmission(beforeSize, &codeSectionIndex);

  beforeSize = writer.size();
  if (failed(emitDataSection(writer, indexSpace, wasmModule.getOperation(),
                             trackerPtr)))
    return failure();
  noteSectionEmission(beforeSize, &dataSectionIndex);

  // Emit linking and relocation sections in relocatable mode
  if (relocatable) {
    beforeSize = writer.size();
    emitLinkingSection(writer, indexSpace, wasmModule.getOperation(),
                       emitSyntheticTable);
    noteSectionEmission(beforeSize);

    if (!tracker.getCodeRelocations().empty()) {
      if (!codeSectionIndex) {
        wasmModule.emitError(
            "internal error: code relocations present without emitted code "
            "section");
        return failure();
      }
      beforeSize = writer.size();
      emitRelocSection(writer, "reloc.CODE", *codeSectionIndex,
                       tracker.getCodeRelocations());
      noteSectionEmission(beforeSize);
    }

    if (!tracker.getDataRelocations().empty()) {
      if (!dataSectionIndex) {
        wasmModule.emitError(
            "internal error: data relocations present without emitted data "
            "section");
        return failure();
      }
      beforeSize = writer.size();
      emitRelocSection(writer, "reloc.DATA", *dataSectionIndex,
                       tracker.getDataRelocations());
      noteSectionEmission(beforeSize);
    }

    beforeSize = writer.size();
    emitTargetFeaturesSection(writer);
    noteSectionEmission(beforeSize);
  }

  // 3. Flush to output
  writer.flush(output);
  return success();
}
