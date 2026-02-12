// REQUIRES: wabt
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack | wasm-emit --mlir-to-wasm --relocatable -o %t.o
// RUN: wasm-validate %t.o
// RUN: wasm-objdump -x %t.o | FileCheck %s

// TDD target: memref.alloc should lower to a call that is represented as a
// WebAssembly import from env.malloc.
// CHECK: Import[2]:
// CHECK-DAG: <malloc>
// CHECK-DAG: <- env.malloc
// CHECK-DAG: <free>
// CHECK-DAG: <- env.free

func.func @alloc_and_free() -> i32 {
  %m = memref.alloc() : memref<4xi32>
  memref.dealloc %m : memref<4xi32>
  %one = arith.constant 1 : i32
  return %one : i32
}
