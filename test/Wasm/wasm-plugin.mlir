// RUN: mlir-opt %s --load-dialect-plugin=%wasm_libs/WasmPlugin%shlibext --pass-pipeline="builtin.module(convert-to-wasm)" | FileCheck %s

module {
  func.func @foo(%i: i32) {
    // CHECK: = wasm.temp.local<i32>
    // CHECK: wasm.constant 2 : i32
    // CHECK: wasm.temp.local.set
    %a = arith.constant 2 : i32
    // CHECK: = wasm.temp.local<i32>
    // CHECK: wasm.constant 4 : i32
    // CHECK: wasm.temp.local.set
    %b = arith.constant 4 : i32
    %c = arith.addi %a, %b : i32
    %d = arith.addi %c, %i : i32
    return
  }
}

