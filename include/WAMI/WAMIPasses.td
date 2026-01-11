//===- WAMIPasses.td - WAMI dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines passes for the WAMI dialect and conversion passes
// to/from the upstream WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_PASSES
#define WAMI_PASSES

include "mlir/Pass/PassBase.td"

def WAMIConvertArith : Pass<"wami-convert-arith", "ModuleOp"> {
  let summary = "Convert Arith dialect to WasmSSA dialect";
  let description = [{
    This pass converts operations from the MLIR Arith dialect to the
    upstream WasmSSA dialect. This is the first step in the WAMI
    compilation pipeline for targeting WebAssembly.
  }];
  let dependentDialects = [
    "wasmssa::WasmSSADialect",
    "arith::ArithDialect",
    "wami::WAMIDialect"
  ];
}

def WAMIConvertFunc : Pass<"wami-convert-func", "ModuleOp"> {
  let summary = "Convert Func dialect to WasmSSA dialect";
  let description = [{
    This pass converts operations from the MLIR Func dialect to the
    upstream WasmSSA dialect. It handles function definitions, calls,
    and returns.
  }];
  let dependentDialects = [
    "wasmssa::WasmSSADialect",
    "func::FuncDialect"
  ];
}

def WAMIConvertScf : Pass<"wami-convert-scf", "ModuleOp"> {
  let summary = "Convert SCF dialect to WasmSSA dialect";
  let description = [{
    This pass converts operations from the MLIR SCF (Structured Control Flow)
    dialect to the upstream WasmSSA dialect. It handles if, for, and while
    loops using direct structured control flow mapping without a relooper
    algorithm.
  }];
  let dependentDialects = [
    "wasmssa::WasmSSADialect",
    "scf::SCFDialect"
  ];
}

def WAMIConvertMemref : Pass<"wami-convert-memref", "ModuleOp"> {
  let summary = "Convert MemRef dialect to WasmSSA/WAMI dialects";
  let description = [{
    This pass converts operations from the MLIR MemRef dialect to the
    upstream WasmSSA dialect and WAMI dialect. It handles:
    - Global memory declarations (memref.global -> wami.data + wasmssa.global)
    - Memory loads (memref.load -> wami.load)
    - Memory stores (memref.store -> wami.store)
    - Dynamic allocation (memref.alloc -> malloc call)
    - Deallocation (memref.dealloc -> free call)

    Memory addressing is computed explicitly using WebAssembly's linear
    memory model with 32-bit addresses starting at offset 1024.
  }];
  let dependentDialects = [
    "wasmssa::WasmSSADialect",
    "wami::WAMIDialect",
    "memref::MemRefDialect"
  ];
}

def WAMIConvertAll : Pass<"wami-convert-all", "ModuleOp"> {
  let summary = "Convert all supported dialects to WasmSSA/WAMI";
  let description = [{
    Unified pass that converts arith, func, scf, and memref dialects to
    WasmSSA and WAMI dialects in a single conversion. This handles cases
    where operations from different dialects are interleaved (e.g., scf.for
    with memref.load inside the loop body).

    Using separate passes can cause issues when operations depend on each
    other across dialect boundaries. This unified pass ensures all patterns
    are available during a single conversion, allowing MLIR's inside-out
    region traversal to convert operations in the correct order.
  }];
  let dependentDialects = [
    "wasmssa::WasmSSADialect",
    "wami::WAMIDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect"
  ];
}

#endif // WAMI_PASSES
