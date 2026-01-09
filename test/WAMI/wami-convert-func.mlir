// RUN: wasm-opt %s --wami-convert-func | FileCheck %s

// CHECK-LABEL: wasmssa.func @simple_func
// CHECK-SAME: () -> i32
func.func @simple_func() -> i32 {
  %c42 = arith.constant 42 : i32
  // CHECK: wasmssa.return
  return %c42 : i32
}

// CHECK-LABEL: wasmssa.func @func_with_args
// CHECK-SAME: (%{{.*}}: !wasmssa<local ref to i32>, %{{.*}}: !wasmssa<local ref to i32>) -> i32
func.func @func_with_args(%arg0: i32, %arg1: i32) -> i32 {
  %sum = arith.addi %arg0, %arg1 : i32
  // CHECK: wasmssa.return
  return %sum : i32
}

// CHECK-LABEL: wasmssa.func @func_no_return
// CHECK-SAME: ()
func.func @func_no_return() {
  // CHECK: wasmssa.return
  return
}

// CHECK-LABEL: wasmssa.func @func_multiple_results
// CHECK-SAME: () -> (i32, i64)
func.func @func_multiple_results() -> (i32, i64) {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i64
  // CHECK: wasmssa.return
  return %c1, %c2 : i32, i64
}

// Test function calls
// CHECK-LABEL: wasmssa.func @caller
func.func @caller() -> i32 {
  // CHECK: wasmssa.call @simple_func : () -> i32
  %result = func.call @simple_func() : () -> i32
  // CHECK: wasmssa.return
  return %result : i32
}

// CHECK-LABEL: wasmssa.func @caller_with_args
// CHECK-SAME: (%{{.*}}: !wasmssa<local ref to i32>, %{{.*}}: !wasmssa<local ref to i32>) -> i32
func.func @caller_with_args(%a: i32, %b: i32) -> i32 {
  // CHECK: wasmssa.call @func_with_args
  %result = func.call @func_with_args(%a, %b) : (i32, i32) -> i32
  // CHECK: wasmssa.return
  return %result : i32
}
