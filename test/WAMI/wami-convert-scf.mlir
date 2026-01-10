// RUN: wasm-opt %s --wami-convert-scf | FileCheck %s

//===----------------------------------------------------------------------===//
// scf.if tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_if_no_result
func.func @test_if_no_result(%cond: i1) {
  // CHECK: wasmssa.if
  scf.if %cond {
    // Empty then block
  }
  return
}

// CHECK-LABEL: func.func @test_if_with_result
func.func @test_if_with_result(%cond: i1) -> i32 {
  // CHECK: wasmssa.if
  %result = scf.if %cond -> (i32) {
    %c1 = arith.constant 1 : i32
    // CHECK: wasmssa.block_return
    scf.yield %c1 : i32
  } else {
    %c0 = arith.constant 0 : i32
    // CHECK: wasmssa.block_return
    scf.yield %c0 : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @test_if_multiple_results
func.func @test_if_multiple_results(%cond: i1) -> (i32, i64) {
  // CHECK: wasmssa.if
  %r1, %r2 = scf.if %cond -> (i32, i64) {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i64
    scf.yield %c1, %c2 : i32, i64
  } else {
    %c0 = arith.constant 0 : i32
    %c3 = arith.constant 3 : i64
    scf.yield %c0, %c3 : i32, i64
  }
  return %r1, %r2 : i32, i64
}

//===----------------------------------------------------------------------===//
// scf.for tests
//===----------------------------------------------------------------------===//

// Simple for loop with no iter_args
// CHECK-LABEL: func.func @test_for_no_iter_args
func.func @test_for_no_iter_args(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  scf.for %i = %c0 to %n step %c1 {
    // Empty body
  }
  return
}

// For loop with iter_args (sum computation)
// CHECK-LABEL: func.func @test_for_with_iter_args
func.func @test_for_with_iter_args(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    // Add 1 each iteration (avoid index_cast for now)
    %new_acc = arith.addi %acc, %one : i32
    scf.yield %new_acc : i32
  }
  return %sum : i32
}

// For loop with multiple iter_args
// CHECK-LABEL: func.func @test_for_multiple_iter_args
func.func @test_for_multiple_iter_args(%n: index) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %sum, %prod = scf.for %i = %c0 to %n step %c1 iter_args(%s = %i0, %p = %i1) -> (i32, i32) {
    // Add/multiply constants (avoid index_cast for now)
    %new_s = arith.addi %s, %i1 : i32
    %new_p = arith.muli %p, %i2 : i32
    scf.yield %new_s, %new_p : i32, i32
  }
  return %sum, %prod : i32, i32
}

//===----------------------------------------------------------------------===//
// scf.while tests
//===----------------------------------------------------------------------===//

// Simple while loop
// CHECK-LABEL: func.func @test_while_simple
func.func @test_while_simple(%init: i32, %limit: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %result = scf.while (%arg = %init) : (i32) -> (i32) {
    %cond = arith.cmpi slt, %arg, %limit : i32
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%val: i32):
    %next = arith.addi %val, %c1 : i32
    scf.yield %next : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Nested control flow tests
//===----------------------------------------------------------------------===//

// Nested if inside for
// CHECK-LABEL: func.func @test_nested_if_in_for
func.func @test_nested_if_in_for(%n: index, %cond: i1) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %i0) -> (i32) {
    // CHECK: wasmssa.if
    %delta = scf.if %cond -> (i32) {
      scf.yield %i1 : i32
    } else {
      scf.yield %i2 : i32
    }
    %new_acc = arith.addi %acc, %delta : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}
