//===- IntermittentPass.td - Wasm dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INTERMITTENT_PASS
#define INTERMITTENT_PASS

include "mlir/Pass/PassBase.td"

def PolygeistToIntermittent : Pass<"polygeist-to-intermittent", "ModuleOp"> {
  let summary = "Converts MLIR file produced by Polygeist to Intermittent dialect";
  let description = [{
    We produce a MLIR file from Polygeist as follows:
    cgeist example.c --c-style-memref  -S -o example.mlir
    Example C file:
    """
    __attribute((nonvolatile_var)) int global_var_1 = 0;
    __attribute((nonvolatile_var)) int global_var_2 = 0;

    __attribute__((intermittent_task(0))) int task_init() {
      // transition to task_a
      return 1;
    }

    __attribute__((intermittent_task(1))) int task_a() {
      global_var_1++;
      // transition to task_b
      return 2;
    }

    __attribute__((intermittent_task(2))) int task_b() {
      global_var_2++;
      // transition to task_a
      return 1;
    }
    """
    Example Output MLIR:
    """
    module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
      memref.global @global_var_2 : memref<1xi32> = dense<0> {intermittent.nonvolatile}
      memref.global @global_var_1 : memref<1xi32> = dense<0> {intermittent.nonvolatile}
      func.func @task_init() -> i32 attributes {intermittent.task = 0 : i32, llvm.linkage = #llvm.linkage<external>} {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      func.func @task_a() -> i32 attributes {intermittent.task = 1 : i32, llvm.linkage = #llvm.linkage<external>} {
        %c2_i32 = arith.constant 2 : i32
        %c1_i32 = arith.constant 1 : i32
        %0 = memref.get_global @global_var_1 : memref<1xi32>
        %1 = affine.load %0[0] : memref<1xi32>
        %2 = arith.addi %1, %c1_i32 : i32
        affine.store %2, %0[0] : memref<1xi32>
        return %c2_i32 : i32
      }
      func.func @task_b() -> i32 attributes {intermittent.task = 2 : i32, llvm.linkage = #llvm.linkage<external>} {
        %c1_i32 = arith.constant 1 : i32
        %0 = memref.get_global @global_var_2 : memref<1xi32>
        %1 = affine.load %0[0] : memref<1xi32>
        %2 = arith.addi %1, %c1_i32 : i32
        affine.store %2, %0[0] : memref<1xi32>
        return %c1_i32 : i32
      }
    }
    """
  }];
  let dependentDialects = [
    "affine::AffineDialect",
    "intermittent::IntermittentDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "func::FuncDialect"
  ];
}

def PreprocessIntermittentToWasm : Pass<"preprocess-intermittent-to-wasm", "ModuleOp"> {
  let summary = "Add preludes";
  let dependentDialects = [
    "intermittent::IntermittentDialect",
    "wasm::WasmDialect"
  ];
}

def ConvertIntermittentToWasm : Pass<"convert-intermittent-to-wasm", "ModuleOp"> {
  let summary = "Converts Intermittent dialect to Wasm dialect";
  let dependentDialects = [
    "intermittent::IntermittentDialect",
    "wasm::WasmDialect"
  ];
}

// intermittent to async passes

def PreprocessIntermittentToStd
    : Pass<"preprocess-intermittent-to-std", "ModuleOp"> {
  let summary = "Create a main function that calls async.runtime.create for each IdempotentTaskOp and then calls the entry task.";
  let description = [{
    This pass walks through the module, finds all IdempotentTaskOps,
    and for each one, inserts an `async.runtime.create` call in a newly
    created (or updated) `@main` function. If a task named by the option
    --entry-task is found, it is called last in the `@main`.
  }];
  let options = [
    Option<"entryTask", "entry-task", "std::string", "\"entry\"",
    "The name of the task to call last in the main function.">,
  ];

  let dependentDialects = [
    "intermittent::IntermittentDialect",
    "async::AsyncDialect",
    "func::FuncDialect",
  ];
}

def ConvertNonVolatileToMemRefForIntermittentToStd
    : Pass<"convert-nonvolatile-to-memref", "ModuleOp"> {
  let summary = "Converts nonvolatile variables to memref";
  let dependentDialects = [
    "intermittent::IntermittentDialect",
    "memref::MemRefDialect",
  ];
}

def ConvertIntermittentToLLVM
    : Pass<"convert-intermittent-task-to-async", "ModuleOp"> {
  let summary = "Convert Intermittent dialect to LLVM dialect";
  let description = [{
    This pass converts the Intermittent dialect operations to LLVM dialect.
  }];

  let dependentDialects = [
    "intermittent::IntermittentDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "cf::ControlFlowDialect",
  ];
}

#endif // INTERMITTENT_PASS
