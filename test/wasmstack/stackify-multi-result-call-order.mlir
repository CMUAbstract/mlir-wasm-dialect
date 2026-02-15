// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// Regression test: multi-result call materialization must store results in
// stack-pop order, so SSA result ordering remains correct.


// CHECK-LABEL: wasmstack.func @diff
// CHECK: wasmstack.local [[L0:[0-9]+]] : i32
// CHECK: wasmstack.local [[L1:[0-9]+]] : i32
// CHECK: wasmstack.call @swap : (i32, i32) -> (i32, i32)
// CHECK-NEXT: wasmstack.local.set [[L1]] : i32
// CHECK-NEXT: wasmstack.local.set [[L0]] : i32
// CHECK: wasmstack.local.get [[L0]] : i32
// CHECK-NEXT: wasmstack.local.get [[L1]] : i32
// CHECK-NEXT: wasmstack.sub : i32

module {
  func.func @swap(%a: i32, %b: i32) -> (i32, i32) {
    return %b, %a : i32, i32
  }

  func.func @diff(%x: i32, %y: i32) -> i32 {
    %a, %b = func.call @swap(%x, %y) : (i32, i32) -> (i32, i32)
    %r = arith.subi %a, %b : i32
    return %r : i32
  }
}
