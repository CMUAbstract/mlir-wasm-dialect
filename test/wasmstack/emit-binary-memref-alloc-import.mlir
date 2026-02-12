// REQUIRES: wabt
// XFAIL: *
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack | wasm-emit --mlir-to-wasm --relocatable -o %t.o
// RUN: wasm-validate %t.o
// RUN: wasm-objdump -x %t.o | FileCheck %s

// TDD target: memref.alloc should lower to a call that is represented as a
// WebAssembly import from env.malloc.
// CHECK: Import[
// CHECK: <malloc>
// CHECK: <- env.malloc

func.func @alloc_and_load() -> i32 {
  %c0 = arith.constant 0 : index
  %v = arith.constant 7 : i32
  %m = memref.alloc() : memref<4xi32>
  memref.store %v, %m[%c0] : memref<4xi32>
  %r = memref.load %m[%c0] : memref<4xi32>
  return %r : i32
}
