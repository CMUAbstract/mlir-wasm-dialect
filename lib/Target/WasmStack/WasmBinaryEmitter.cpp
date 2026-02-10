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

/// Emit the type section (section 1).
static void emitTypeSection(BinaryWriter &output, IndexSpace &indexSpace) {
  BinaryWriter section;
  const auto &types = indexSpace.getTypes();
  section.writeULEB128(types.size());

  for (const auto &sig : types) {
    section.writeByte(wc::FuncTypeTag); // 0x60
    // Params
    section.writeULEB128(sig.params.size());
    for (Type t : sig.params)
      section.writeValType(t);
    // Results
    section.writeULEB128(sig.results.size());
    for (Type t : sig.results)
      section.writeValType(t);
  }

  output.writeSection(wc::SectionId::Type, section);
}

/// Emit the function section (section 3) - maps function index to type index.
static void emitFunctionSection(BinaryWriter &output, IndexSpace &indexSpace,
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
    uint32_t typeIdx = indexSpace.getTypeIndex(sig);
    section.writeULEB128(typeIdx);
  }

  output.writeSection(wc::SectionId::Function, section);
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

/// Emit the global section (section 6).
static void emitGlobalSection(BinaryWriter &output, IndexSpace &indexSpace,
                              Operation *moduleOp) {
  SmallVector<GlobalOp> globals;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto globalOp = dyn_cast<GlobalOp>(op))
      globals.push_back(globalOp);
  }

  if (globals.empty())
    return;

  BinaryWriter section;
  section.writeULEB128(globals.size());
  for (auto globalOp : globals) {
    // Global type: valtype + mutability
    section.writeValType(globalOp.getTypeAttr().getValue());
    section.writeByte(globalOp.getIsMutable()
                          ? static_cast<uint8_t>(wc::Mutability::Var)
                          : static_cast<uint8_t>(wc::Mutability::Const));

    // Init expression: emit the operations in the init region
    OpcodeEmitter emitter(section, indexSpace);
    for (Operation &initOp : globalOp.getInit().front()) {
      emitter.emitOperation(&initOp);
    }
    section.writeByte(wc::Opcode::End);
  }
  output.writeSection(wc::SectionId::Global, section);
}

/// Emit the export section (section 7).
static void emitExportSection(BinaryWriter &output, IndexSpace &indexSpace,
                              Operation *moduleOp) {
  BinaryWriter section;

  // Collect exports
  SmallVector<std::tuple<StringRef, wc::ExportKind, uint32_t>> exports;

  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      if (auto exportName = funcOp.getExportName()) {
        uint32_t idx = indexSpace.getFuncIndex(funcOp.getSymName());
        exports.push_back({*exportName, wc::ExportKind::Func, idx});
      }
    } else if (auto memOp = dyn_cast<MemoryOp>(op)) {
      if (auto exportName = memOp.getExportName()) {
        uint32_t idx = indexSpace.getMemoryIndex(memOp.getSymName());
        exports.push_back({*exportName, wc::ExportKind::Memory, idx});
      }
    } else if (auto globalOp = dyn_cast<GlobalOp>(op)) {
      if (auto exportName = globalOp.getExportName()) {
        uint32_t idx = indexSpace.getGlobalIndex(globalOp.getSymName());
        exports.push_back({*exportName, wc::ExportKind::Global, idx});
      }
    }
  }

  if (exports.empty())
    return;

  section.writeULEB128(exports.size());
  for (auto &[name, kind, idx] : exports) {
    section.writeString(name);
    section.writeByte(static_cast<uint8_t>(kind));
    section.writeULEB128(idx);
  }

  output.writeSection(wc::SectionId::Export, section);
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
      funcBody.writeValType(type);
    }

    // The sectionOffset for relocations is the current position in the section
    // buffer where the function body bytes will land. This is after the
    // func count prefix and all previous (size_prefix + body) entries.
    uint32_t sectionOffset = section.size();

    // Emit function body opcodes
    OpcodeEmitter emitter(funcBody, indexSpace, tracker, sectionOffset);
    if (!emitter.emitFunctionBody(funcOp)) {
      funcOp.emitError("failed to emit function body");
      return failure();
    }

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
static void emitDataSection(BinaryWriter &output, IndexSpace &indexSpace,
                            Operation *moduleOp, RelocationTracker *tracker) {
  SmallVector<DataOp> dataOps;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto dataOp = dyn_cast<DataOp>(op))
      dataOps.push_back(dataOp);
  }

  if (dataOps.empty())
    return;

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
      uint32_t symIdx = indexSpace.getSymbolIndex(symName);
      tracker->addDataRelocation(
          static_cast<uint8_t>(wc::RelocType::R_WASM_MEMORY_ADDR_SLEB),
          section.offset(), symIdx, dataOp.getOffset());
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
}

//===----------------------------------------------------------------------===//
// Linking and relocation sections (relocatable mode only)
//===----------------------------------------------------------------------===//

/// Emit the "linking" custom section.
static void emitLinkingSection(BinaryWriter &output, IndexSpace &indexSpace,
                               Operation *moduleOp) {
  BinaryWriter section;

  // Linking metadata version
  section.writeULEB128(wc::WasmMetadataVersion);

  // WASM_SYMBOL_TABLE subsection
  {
    BinaryWriter subsection;
    const auto &symbols = indexSpace.getSymbols();
    subsection.writeULEB128(symbols.size());

    for (const auto &sym : symbols) {
      subsection.writeByte(static_cast<uint8_t>(sym.kind));
      subsection.writeULEB128(sym.flags);

      if (sym.kind == wasm::SymtabKind::Function ||
          sym.kind == wasm::SymtabKind::Global) {
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
  // Find the wasmstack.module inside the MLIR module
  Operation *wasmModule = nullptr;
  if (isa<wasmstack::ModuleOp>(op)) {
    wasmModule = op;
  } else {
    // Walk the top-level mlir::ModuleOp to find wasmstack.module
    op->walk([&](wasmstack::ModuleOp mod) { wasmModule = mod; });
  }

  if (!wasmModule) {
    op->emitError("no wasmstack.module found");
    return failure();
  }

  // 1. Analyze index spaces
  IndexSpace indexSpace;
  indexSpace.analyze(wasmModule);

  // Build symbol table for relocatable mode
  RelocationTracker tracker;
  RelocationTracker *trackerPtr = relocatable ? &tracker : nullptr;
  if (relocatable)
    indexSpace.buildSymbolTable(wasmModule);

  // 2. Build output
  BinaryWriter writer;

  // Magic number and version
  writer.writeBytes(wc::Magic, sizeof(wc::Magic));
  writer.writeBytes(wc::Version, sizeof(wc::Version));

  // Emit sections in order.
  emitTypeSection(writer, indexSpace);
  emitFunctionSection(writer, indexSpace, wasmModule);
  emitMemorySection(writer, wasmModule);
  emitGlobalSection(writer, indexSpace, wasmModule);

  // Export section: skip in relocatable mode (exports become symbol flags)
  if (!relocatable)
    emitExportSection(writer, indexSpace, wasmModule);

  emitDataCountSection(writer, wasmModule);

  // The code section index: count sections emitted before it.
  // type(1) + function(3) + memory(5) + global(6) + export(7, if !reloc) +
  // datacount(12). We need the actual section index in the binary.
  // Section indices are sequential starting from 0 for each section in the
  // binary, not by section ID.
  // Let's count properly.
  uint32_t codeSectionIdx = 0;
  uint32_t dataSectionIdx = 0;
  {
    // Count sections before code section
    uint32_t idx = 0;
    idx++; // type
    idx++; // function
    // memory section may be absent
    bool hasMemory = false;
    for (Operation &mop : wasmModule->getRegion(0).front())
      if (isa<MemoryOp>(mop)) {
        hasMemory = true;
        break;
      }
    if (hasMemory)
      idx++;
    // global section may be absent
    bool hasGlobal = false;
    for (Operation &mop : wasmModule->getRegion(0).front())
      if (isa<GlobalOp>(mop)) {
        hasGlobal = true;
        break;
      }
    if (hasGlobal)
      idx++;
    // export section (only in non-relocatable mode)
    if (!relocatable) {
      bool hasExport = false;
      for (Operation &mop : wasmModule->getRegion(0).front()) {
        if (auto f = dyn_cast<FuncOp>(mop))
          if (f.getExportName()) {
            hasExport = true;
            break;
          }
        if (auto m = dyn_cast<MemoryOp>(mop))
          if (m.getExportName()) {
            hasExport = true;
            break;
          }
        if (auto g = dyn_cast<GlobalOp>(mop))
          if (g.getExportName()) {
            hasExport = true;
            break;
          }
      }
      if (hasExport)
        idx++;
    }
    // data count section may be absent
    bool hasData = false;
    for (Operation &mop : wasmModule->getRegion(0).front())
      if (isa<DataOp>(mop)) {
        hasData = true;
        break;
      }
    if (hasData)
      idx++; // data count
    codeSectionIdx = idx;
    idx++; // code
    dataSectionIdx = idx;
  }

  if (failed(emitCodeSection(writer, indexSpace, wasmModule, trackerPtr)))
    return failure();

  emitDataSection(writer, indexSpace, wasmModule, trackerPtr);

  // Emit linking and relocation sections in relocatable mode
  if (relocatable) {
    emitLinkingSection(writer, indexSpace, wasmModule);
    emitRelocSection(writer, "reloc.CODE", codeSectionIdx,
                     tracker.getCodeRelocations());
    emitRelocSection(writer, "reloc.DATA", dataSectionIdx,
                     tracker.getDataRelocations());
    emitTargetFeaturesSection(writer);
  }

  // 3. Flush to output
  writer.flush(output);
  return success();
}
