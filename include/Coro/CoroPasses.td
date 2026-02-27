//===- CoroPasses.td - Coro passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines passes for the coroutine intrinsic ABI.
//
//===----------------------------------------------------------------------===//

#ifndef CORO_PASSES
#define CORO_PASSES

include "mlir/Pass/PassBase.td"

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

    The lowering uses stable i64 runtime handles and module-level mutable
    runtime slots (continuation + state) so resume can continue from a paused
    continuation rather than rematerializing a fresh one.
  }];
  let dependentDialects = [
    "wasmssa::WasmSSADialect",
    "wami::WAMIDialect"
  ];
}

def CoroToLLVM : Pass<"coro-to-llvm", "ModuleOp"> {
  let summary = "Lower coro intrinsics to LLVM coroutine runtime/helpers";
  let description = [{
    Expects LLVM dialect input (typically after convert-func/cf/scf/arith-to-LLVM)
    and lowers coro intrinsics to per-kind LLVM runtime helpers with:
    - opaque pointer handle encoded as i64
    - persistent per-instance runtime object state
    - coroutine suspend/resume via llvm.coro.* intrinsics
    - deterministic invalid-handle trap paths
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
}

#endif // CORO_PASSES
