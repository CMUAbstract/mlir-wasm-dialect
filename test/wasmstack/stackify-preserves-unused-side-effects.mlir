// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// Regression test: stackification must not drop side-effecting or trap-capable
// ops just because their SSA results are unused.


// CHECK-LABEL: wasmstack.func @unused_call_should_be_preserved
// CHECK: wasmstack.call @foo : () -> i32

// CHECK-LABEL: wasmstack.func @unused_div_should_be_preserved
// CHECK: wasmstack.div_s : i32

module {
  func.func private @foo() -> i32

  func.func @unused_call_should_be_preserved() -> i32 {
    %v = func.call @foo() : () -> i32
    %c7 = arith.constant 7 : i32
    return %c7 : i32
  }

  func.func @unused_div_should_be_preserved() -> i32 {
    %a = arith.constant 1 : i32
    %z = arith.constant 0 : i32
    %d = arith.divsi %a, %z : i32
    %c9 = arith.constant 9 : i32
    return %c9 : i32
  }
}
