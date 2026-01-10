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

#endif // WAMI_PASSES
