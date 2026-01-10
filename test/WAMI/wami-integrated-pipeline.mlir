// RUN: wasm-opt %s --wami-convert-scf --wami-convert-arith --wami-convert-func | FileCheck %s

//===----------------------------------------------------------------------===//
// Integrated tests: arith + func + scf → WasmSSA
//
// These tests verify that the full pipeline correctly converts standard MLIR
// dialects (arith, func, scf) to the WasmSSA dialect.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Basic function with arithmetic
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @add_two_numbers
// CHECK-SAME: (%{{.*}}: !wasmssa<local ref to i32>, %{{.*}}: !wasmssa<local ref to i32>) -> i32
func.func @add_two_numbers(%a: i32, %b: i32) -> i32 {
  // CHECK: wasmssa.add
  %sum = arith.addi %a, %b : i32
  // CHECK: wasmssa.return
  return %sum : i32
}

//===----------------------------------------------------------------------===//
// Function with if-else control flow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @max_of_two
// CHECK-SAME: (%{{.*}}: !wasmssa<local ref to i32>, %{{.*}}: !wasmssa<local ref to i32>) -> i32
func.func @max_of_two(%a: i32, %b: i32) -> i32 {
  // CHECK: wasmssa.gt_si
  %cond = arith.cmpi sgt, %a, %b : i32
  // CHECK: wasmssa.if
  %result = scf.if %cond -> (i32) {
    // CHECK: wasmssa.block_return
    scf.yield %a : i32
  } else {
    // CHECK: wasmssa.block_return
    scf.yield %b : i32
  }
  // CHECK: wasmssa.return
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Simple for loop - sum from 0 to n
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @sum_to_n
// CHECK-SAME: (%{{.*}}: !wasmssa<local ref to i32>) -> i32
func.func @sum_to_n(%n: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n_idx = arith.index_cast %n : i32 to index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32

  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %sum = scf.for %i = %c0 to %n_idx step %c1 iter_args(%acc = %init) -> (i32) {
    // CHECK: wasmssa.add
    %new_acc = arith.addi %acc, %one : i32
    scf.yield %new_acc : i32
  }

  // CHECK: wasmssa.return
  return %sum : i32
}

//===----------------------------------------------------------------------===//
// Nested control flow: for loop with conditional inside
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @conditional_sum
func.func @conditional_sum(%n: i32, %cond: i1) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n_idx = arith.index_cast %n : i32 to index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32

  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %sum = scf.for %i = %c0 to %n_idx step %c1 iter_args(%acc = %init) -> (i32) {
    // CHECK: wasmssa.if
    %delta = scf.if %cond -> (i32) {
      scf.yield %one : i32
    } else {
      scf.yield %two : i32
    }
    // CHECK: wasmssa.add
    %new_acc = arith.addi %acc, %delta : i32
    scf.yield %new_acc : i32
  }

  // CHECK: wasmssa.return
  return %sum : i32
}

//===----------------------------------------------------------------------===//
// While loop - count up to limit
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @count_to_limit
// CHECK-SAME: (%{{.*}}: !wasmssa<local ref to i32>, %{{.*}}: !wasmssa<local ref to i32>) -> i32
func.func @count_to_limit(%init: i32, %limit: i32) -> i32 {
  %c1 = arith.constant 1 : i32

  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %result = scf.while (%arg = %init) : (i32) -> (i32) {
    // CHECK: wasmssa.lt_si
    %cond = arith.cmpi slt, %arg, %limit : i32
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%val: i32):
    // CHECK: wasmssa.add
    %next = arith.addi %val, %c1 : i32
    scf.yield %next : i32
  }

  // CHECK: wasmssa.return
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Function with multiple arithmetic operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @arithmetic_expr
// CHECK-SAME: (%{{.*}}: !wasmssa<local ref to i32>, %{{.*}}: !wasmssa<local ref to i32>, %{{.*}}: !wasmssa<local ref to i32>) -> i32
func.func @arithmetic_expr(%a: i32, %b: i32, %c: i32) -> i32 {
  // Compute (a + b) * c - a
  // CHECK: wasmssa.add
  %sum = arith.addi %a, %b : i32
  // CHECK: wasmssa.mul
  %prod = arith.muli %sum, %c : i32
  // CHECK: wasmssa.sub
  %result = arith.subi %prod, %a : i32
  // CHECK: wasmssa.return
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Floating point operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @float_ops
// CHECK-SAME: (%{{.*}}: !wasmssa<local ref to f32>, %{{.*}}: !wasmssa<local ref to f32>) -> f32
func.func @float_ops(%a: f32, %b: f32) -> f32 {
  // CHECK: wasmssa.add
  %sum = arith.addf %a, %b : f32
  // CHECK: wasmssa.mul
  %prod = arith.mulf %sum, %b : f32
  // CHECK: wasmssa.return
  return %prod : f32
}

//===----------------------------------------------------------------------===//
// Factorial using while loop
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @factorial
func.func @factorial(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // fact = 1, i = n
  // while (i > 0) { fact *= i; i--; }
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %result:2 = scf.while (%fact = %c1, %i = %n) : (i32, i32) -> (i32, i32) {
    // CHECK: wasmssa.gt_si
    %cond = arith.cmpi sgt, %i, %c0 : i32
    scf.condition(%cond) %fact, %i : i32, i32
  } do {
  ^bb0(%f: i32, %iv: i32):
    // CHECK: wasmssa.mul
    %new_fact = arith.muli %f, %iv : i32
    // CHECK: wasmssa.sub
    %new_i = arith.subi %iv, %c1 : i32
    scf.yield %new_fact, %new_i : i32, i32
  }

  // CHECK: wasmssa.return
  return %result#0 : i32
}

//===----------------------------------------------------------------------===//
// Multiple function calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @callee
func.func @callee(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  // CHECK: wasmssa.mul
  %doubled = arith.muli %x, %c2 : i32
  // CHECK: wasmssa.return
  return %doubled : i32
}

// CHECK-LABEL: wasmssa.func @caller_example
func.func @caller_example(%a: i32, %b: i32) -> i32 {
  // CHECK: wasmssa.call @callee
  %doubled_a = func.call @callee(%a) : (i32) -> i32
  // CHECK: wasmssa.call @callee
  %doubled_b = func.call @callee(%b) : (i32) -> i32
  // CHECK: wasmssa.add
  %result = arith.addi %doubled_a, %doubled_b : i32
  // CHECK: wasmssa.return
  return %result : i32
}

//===----------------------------------------------------------------------===//
// If without else (empty else branch)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @if_no_else
func.func @if_no_else(%cond: i1, %val: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  // CHECK: wasmssa.if
  %result = scf.if %cond -> (i32) {
    // CHECK: wasmssa.add
    %incremented = arith.addi %val, %c1 : i32
    // CHECK: wasmssa.block_return
    scf.yield %incremented : i32
  } else {
    // CHECK: wasmssa.block_return
    scf.yield %val : i32
  }
  // CHECK: wasmssa.return
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Bitwise operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @bitwise_ops
func.func @bitwise_ops(%a: i32, %b: i32) -> i32 {
  // CHECK: wasmssa.and
  %and_result = arith.andi %a, %b : i32
  // CHECK: wasmssa.or
  %or_result = arith.ori %and_result, %a : i32
  // CHECK: wasmssa.xor
  %xor_result = arith.xori %or_result, %b : i32
  // CHECK: wasmssa.return
  return %xor_result : i32
}

//===----------------------------------------------------------------------===//
// Shift operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @shift_ops
func.func @shift_ops(%val: i32, %amount: i32) -> i32 {
  // CHECK: wasmssa.shl
  %shifted_left = arith.shli %val, %amount : i32
  // CHECK: wasmssa.shr_u
  %shifted_right_u = arith.shrui %shifted_left, %amount : i32
  // CHECK: wasmssa.shr_s
  %shifted_right_s = arith.shrsi %shifted_right_u, %amount : i32
  // CHECK: wasmssa.return
  return %shifted_right_s : i32
}

//===----------------------------------------------------------------------===//
// 64-bit integer operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @i64_ops
func.func @i64_ops(%a: i64, %b: i64) -> i64 {
  // CHECK: wasmssa.add
  %sum = arith.addi %a, %b : i64
  // CHECK: wasmssa.mul
  %prod = arith.muli %sum, %b : i64
  // CHECK: wasmssa.return
  return %prod : i64
}

//===----------------------------------------------------------------------===//
// Type conversions (extension/truncation)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @type_conversions
func.func @type_conversions(%i32_val: i32) -> i32 {
  // CHECK: wasmssa.extend_i32_s
  %i64_val = arith.extsi %i32_val : i32 to i64
  // Add something to use the i64 value
  %c1 = arith.constant 1 : i64
  // CHECK: wasmssa.add
  %i64_sum = arith.addi %i64_val, %c1 : i64
  // CHECK: wasmssa.wrap
  %result = arith.trunci %i64_sum : i64 to i32
  // CHECK: wasmssa.return
  return %result : i32
}
