// RUN: wasm-opt %s | wasm-opt | FileCheck %s

// Simple test for wasmstack dialect parsing/printing

// CHECK-LABEL: wasmstack.module @test
wasmstack.module @test {
  // CHECK: wasmstack.func @add : (i32, i32) -> i32
  wasmstack.func @add : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    wasmstack.return
  }
}
