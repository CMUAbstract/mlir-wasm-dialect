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
static LogicalResult emitCodeSection(BinaryWriter &output,
                                     IndexSpace &indexSpace,
                                     Operation *moduleOp) {
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

    // Emit function body opcodes
    OpcodeEmitter emitter(funcBody, indexSpace);
    if (!emitter.emitFunctionBody(funcOp)) {
      funcOp.emitError("failed to emit function body");
      return failure();
    }

    // Write function body with size prefix
    section.writeULEB128(funcBody.size());
    section.append(funcBody);
  }

  output.writeSection(wc::SectionId::Code, section);
  return success();
}

/// Emit the data section (section 11).
static void emitDataSection(BinaryWriter &output, Operation *moduleOp) {
  SmallVector<DataOp> dataOps;
  for (Operation &op : moduleOp->getRegion(0).front()) {
    if (auto dataOp = dyn_cast<DataOp>(op))
      dataOps.push_back(dataOp);
  }

  if (dataOps.empty())
    return;

  BinaryWriter section;
  section.writeULEB128(dataOps.size());

  for (auto dataOp : dataOps) {
    // Active data segment: flags=0, offset expr, data bytes
    section.writeULEB128(0); // flags: active, memory 0
    // Offset init expression: i32.const <offset>
    section.writeByte(wc::Opcode::I32Const);
    section.writeSLEB128(dataOp.getOffset());
    section.writeByte(wc::Opcode::End);
    // Data bytes
    StringRef data = dataOp.getData();
    section.writeULEB128(data.size());
    section.writeBytes(reinterpret_cast<const uint8_t *>(data.data()),
                       data.size());
  }

  output.writeSection(wc::SectionId::Data, section);
}

//===----------------------------------------------------------------------===//
// Top-level emitter
//===----------------------------------------------------------------------===//

LogicalResult mlir::wasmstack::emitWasmBinary(Operation *op,
                                              llvm::raw_ostream &output) {
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

  // 2. Build output
  BinaryWriter writer;

  // Magic number and version
  writer.writeBytes(wc::Magic, sizeof(wc::Magic));
  writer.writeBytes(wc::Version, sizeof(wc::Version));

  // Emit sections in order
  emitTypeSection(writer, indexSpace);
  // Import section would go here (Phase 4+)
  emitFunctionSection(writer, indexSpace, wasmModule);
  // Table section would go here (if call_indirect used)
  emitMemorySection(writer, wasmModule);
  emitGlobalSection(writer, indexSpace, wasmModule);
  emitExportSection(writer, indexSpace, wasmModule);
  emitDataCountSection(writer, wasmModule);

  if (failed(emitCodeSection(writer, indexSpace, wasmModule)))
    return failure();

  emitDataSection(writer, wasmModule);

  // 3. Flush to output
  writer.flush(output);
  return success();
}
