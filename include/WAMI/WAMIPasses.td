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

def WAMIConvertMath : Pass<"wami-convert-math", "ModuleOp"> {
  let summary = "Convert Math dialect to WasmSSA dialect";
  let description = [{
    This pass converts supported operations from the MLIR Math dialect to the
    upstream WasmSSA dialect. In this iteration, it lowers:
    - math.sqrt -> wasmssa.sqrt

    Unsupported math operations are rejected to keep the pipeline behavior
    deterministic.
  }];
  let dependentDialects = [
    "wasmssa::WasmSSADialect",
    "math::MathDialect"
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
    Unified pass that converts arith, math, func, scf, and memref dialects to
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
    "math::MathDialect",
    "func::FuncDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect"
  ];
}

def CoroVerifyIntrinsics : Pass<"coro-verify-intrinsics", "ModuleOp"> {
  let summary = "Verify canonical coro.* intrinsic ABI usage";
  let description = [{
    Validates source-level coro intrinsic calls:
    - coro.spawn.<kind>
    - coro.resume.<kind>
    - coro.yield.<tag>
    - coro.is_done.<kind>
    - coro.cancel.<kind>

    It enforces naming conventions, basic signature shape, and consistency
    between spawn/resume calls and their corresponding implementation function
    symbol @coro.impl.<kind>.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "arith::ArithDialect"
  ];
}

def CoroNormalize : Pass<"coro-normalize", "ModuleOp"> {
  let summary = "Normalize canonical coro intrinsic declarations";
  let description = [{
    Lightweight normalization for the coro intrinsic ABI.
    This pass currently enforces declaration-only intrinsic symbols and
    canonical private visibility for declarations.
  }];
  let dependentDialects = [
    "func::FuncDialect"
  ];
}

def CoroToWAMI : Pass<"coro-to-wami", "ModuleOp"> {
  let summary = "Lower coro intrinsic calls to WAMI stack-switching ops";
  let description = [{
    Lowers wasmssa.call sites of coro.spawn/resume/yield to explicit WAMI
    continuation and suspension operations.
  }];
  let dependentDialects = [
    "wasmssa::WasmSSADialect",
    "wami::WAMIDialect"
  ];
}

def CoroToLLVM : Pass<"coro-to-llvm", "ModuleOp"> {
  let summary = "Lower coro intrinsic calls to a direct-call LLVM-style baseline";
  let description = [{
    Lowers func.call sites of coro intrinsics to a direct-call baseline:
    - spawn -> i64 constant handle
    - resume -> call @coro.impl.<kind>
    - yield -> passthrough value (or erase for void)
    - is_done -> constant true
    - cancel -> erase
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "arith::ArithDialect"
  ];
}

#endif // WAMI_PASSES
