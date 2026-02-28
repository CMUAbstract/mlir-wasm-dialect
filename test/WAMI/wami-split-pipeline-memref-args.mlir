// RUN: wasm-opt %s \
// RUN:   --wami-convert-memref \
// RUN:   --wami-convert-scf --wami-convert-arith --wami-convert-func \
// RUN:   --reconcile-unrealized-casts \
// RUN:   --convert-to-wasmstack --verify-wasmstack | FileCheck %s

// Verify that the split conversion pipeline correctly handles functions with
// memref parameters. When memref conversion runs before func conversion, it
// creates unrealized_conversion_cast(memref -> i32) for base addresses. The
// later func conversion must produce reconcilable casts so the chain
// local_ref(i32) -> memref -> i32 folds cleanly via reconcile-unrealized-casts.

// CHECK-LABEL: wasmstack.func @memref_arg_load
// CHECK:         wasmstack.local.get 0 : i32
// CHECK:         wasmstack.f64.load
// CHECK:         wasmstack.return
func.func @memref_arg_load(%mem: memref<?xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %v = memref.load %mem[%c0] : memref<?xf64>
  return %v : f64
}

// CHECK-LABEL: wasmstack.func @memref_arg_store
// CHECK:         wasmstack.local.get 0 : i32
// CHECK:         wasmstack.f64.store
// CHECK:         wasmstack.return
func.func @memref_arg_store(%mem: memref<?xf64>, %val: f64) {
  %c0 = arith.constant 0 : index
  memref.store %val, %mem[%c0] : memref<?xf64>
  return
}

// CHECK-LABEL: wasmstack.func @memref_arg_loop
// CHECK:         wasmstack.loop
func.func @memref_arg_loop(%mem: memref<?xf64>, %n: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f64
  %bound = arith.index_cast %n : i32 to index
  scf.for %i = %c0 to %bound step %c1 {
    memref.store %cst, %mem[%i] : memref<?xf64>
  }
  return
}

// CHECK-LABEL: wasmstack.func @multi_memref_args
// CHECK:         wasmstack.local.get 0 : i32
// CHECK:         wasmstack.f64.load
func.func @multi_memref_args(%a: memref<?xf64>, %b: memref<?x10xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %v = memref.load %a[%c0] : memref<?xf64>
  return %v : f64
}
