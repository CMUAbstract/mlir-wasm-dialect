// RUN: wasm-opt %s \
// RUN:   --wami-convert-memref --wami-convert-scf --wami-convert-arith --wami-convert-func \
// RUN:   --reconcile-unrealized-casts \
// RUN:   --convert-to-wasmstack \
// RUN:   --verify-wasmstack 2>&1 | FileCheck %s

// Full pipeline tests: Standard MLIR → WasmSSA → WasmStack → Verification
//
// This file tests the complete compilation pipeline from standard MLIR dialects
// (arith, scf, func, memref) all the way through stackification and verification.
// This ensures that:
// 1. The conversion passes produce valid WasmSSA
// 2. The stackification pass produces valid WasmStack
// 3. The verification pass confirms stack semantics are correct
//
// If any step produces invalid IR, the verification pass will catch it.

// CHECK: ConvertToWasmStack pass running on module
// CHECK-NOT: error:
// CHECK-NOT: stack underflow
// CHECK-NOT: type mismatch

//===----------------------------------------------------------------------===//
// Basic Arithmetic Functions
//===----------------------------------------------------------------------===//

// Simple addition
// CHECK-LABEL: wasmstack.func @add
func.func @add(%a: i32, %b: i32) -> i32 {
  %result = arith.addi %a, %b : i32
  return %result : i32
}

// All basic arithmetic operations
// CHECK-LABEL: wasmstack.func @all_arith
func.func @all_arith(%a: i32, %b: i32) -> i32 {
  %add = arith.addi %a, %b : i32
  %sub = arith.subi %add, %b : i32
  %mul = arith.muli %sub, %b : i32
  %div = arith.divsi %mul, %b : i32
  %rem = arith.remsi %div, %b : i32
  return %rem : i32
}

// Unsigned arithmetic
// CHECK-LABEL: wasmstack.func @unsigned_arith
func.func @unsigned_arith(%a: i32, %b: i32) -> i32 {
  %div = arith.divui %a, %b : i32
  %rem = arith.remui %div, %b : i32
  return %rem : i32
}

// Bitwise operations
// CHECK-LABEL: wasmstack.func @bitwise
func.func @bitwise(%a: i32, %b: i32) -> i32 {
  %and = arith.andi %a, %b : i32
  %or = arith.ori %and, %b : i32
  %xor = arith.xori %or, %b : i32
  return %xor : i32
}

// Shift operations
// CHECK-LABEL: wasmstack.func @shifts
func.func @shifts(%val: i32, %amt: i32) -> i32 {
  %shl = arith.shli %val, %amt : i32
  %shr_u = arith.shrui %shl, %amt : i32
  %shr_s = arith.shrsi %shr_u, %amt : i32
  return %shr_s : i32
}

// Expression tree (tests stackification order)
// CHECK-LABEL: wasmstack.func @expr_tree
func.func @expr_tree(%a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
  %ab = arith.addi %a, %b : i32
  %cd = arith.addi %c, %d : i32
  %result = arith.muli %ab, %cd : i32
  return %result : i32
}

// Multi-use value (tests tee pattern)
// CHECK-LABEL: wasmstack.func @multi_use
func.func @multi_use(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %squared = arith.muli %sum, %sum : i32
  return %squared : i32
}

// Constant folding and rematerialization
// CHECK-LABEL: wasmstack.func @constants
func.func @constants(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c10 = arith.constant 10 : i32
  %add1 = arith.addi %x, %c1 : i32
  %add2 = arith.addi %add1, %c2 : i32
  %result = arith.muli %add2, %c10 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @comparisons
func.func @comparisons(%a: i32, %b: i32) -> i32 {
  %eq = arith.cmpi eq, %a, %b : i32
  %ne = arith.cmpi ne, %a, %b : i32
  %slt = arith.cmpi slt, %a, %b : i32
  %sle = arith.cmpi sle, %a, %b : i32
  %sgt = arith.cmpi sgt, %a, %b : i32
  %sge = arith.cmpi sge, %a, %b : i32
  // Convert i1 to i32 and combine results
  %eq_i32 = arith.extui %eq : i1 to i32
  %ne_i32 = arith.extui %ne : i1 to i32
  %slt_i32 = arith.extui %slt : i1 to i32
  %sle_i32 = arith.extui %sle : i1 to i32
  %sgt_i32 = arith.extui %sgt : i1 to i32
  %sge_i32 = arith.extui %sge : i1 to i32
  %r1 = arith.addi %eq_i32, %ne_i32 : i32
  %r2 = arith.addi %r1, %slt_i32 : i32
  %r3 = arith.addi %r2, %sle_i32 : i32
  %r4 = arith.addi %r3, %sgt_i32 : i32
  %result = arith.addi %r4, %sge_i32 : i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @unsigned_comparisons
func.func @unsigned_comparisons(%a: i32, %b: i32) -> i32 {
  %ult = arith.cmpi ult, %a, %b : i32
  %ule = arith.cmpi ule, %a, %b : i32
  %ugt = arith.cmpi ugt, %a, %b : i32
  %uge = arith.cmpi uge, %a, %b : i32
  %ult_i32 = arith.extui %ult : i1 to i32
  %ule_i32 = arith.extui %ule : i1 to i32
  %ugt_i32 = arith.extui %ugt : i1 to i32
  %uge_i32 = arith.extui %uge : i1 to i32
  %r1 = arith.addi %ult_i32, %ule_i32 : i32
  %r2 = arith.addi %r1, %ugt_i32 : i32
  %result = arith.addi %r2, %uge_i32 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// 64-bit Integer Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @i64_arithmetic
func.func @i64_arithmetic(%a: i64, %b: i64) -> i64 {
  %add = arith.addi %a, %b : i64
  %sub = arith.subi %add, %b : i64
  %mul = arith.muli %sub, %b : i64
  return %mul : i64
}

// CHECK-LABEL: wasmstack.func @i64_comparisons
func.func @i64_comparisons(%a: i64, %b: i64) -> i32 {
  %lt = arith.cmpi slt, %a, %b : i64
  %gt = arith.cmpi sgt, %a, %b : i64
  %lt_i32 = arith.extui %lt : i1 to i32
  %gt_i32 = arith.extui %gt : i1 to i32
  %result = arith.addi %lt_i32, %gt_i32 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Floating Point Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @f32_arithmetic
func.func @f32_arithmetic(%a: f32, %b: f32) -> f32 {
  %add = arith.addf %a, %b : f32
  %sub = arith.subf %add, %b : f32
  %mul = arith.mulf %sub, %b : f32
  %div = arith.divf %mul, %b : f32
  return %div : f32
}

// CHECK-LABEL: wasmstack.func @f64_arithmetic
func.func @f64_arithmetic(%a: f64, %b: f64) -> f64 {
  %add = arith.addf %a, %b : f64
  %mul = arith.mulf %add, %b : f64
  return %mul : f64
}

// CHECK-LABEL: wasmstack.func @f32_comparisons
func.func @f32_comparisons(%a: f32, %b: f32) -> i32 {
  %olt = arith.cmpf olt, %a, %b : f32
  %ole = arith.cmpf ole, %a, %b : f32
  %ogt = arith.cmpf ogt, %a, %b : f32
  %oge = arith.cmpf oge, %a, %b : f32
  %oeq = arith.cmpf oeq, %a, %b : f32
  %one = arith.cmpf one, %a, %b : f32
  %olt_i32 = arith.extui %olt : i1 to i32
  %ole_i32 = arith.extui %ole : i1 to i32
  %ogt_i32 = arith.extui %ogt : i1 to i32
  %oge_i32 = arith.extui %oge : i1 to i32
  %oeq_i32 = arith.extui %oeq : i1 to i32
  %one_i32 = arith.extui %one : i1 to i32
  %r1 = arith.addi %olt_i32, %ole_i32 : i32
  %r2 = arith.addi %r1, %ogt_i32 : i32
  %r3 = arith.addi %r2, %oge_i32 : i32
  %r4 = arith.addi %r3, %oeq_i32 : i32
  %result = arith.addi %r4, %one_i32 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Control Flow - If/Else
//===----------------------------------------------------------------------===//

// Simple if-else (condition generated from comparison inside)
// CHECK-LABEL: wasmstack.func @simple_if
func.func @simple_if(%selector: i32, %a: i32, %b: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %cond = arith.cmpi ne, %selector, %zero : i32
  %result = scf.if %cond -> (i32) {
    scf.yield %a : i32
  } else {
    scf.yield %b : i32
  }
  return %result : i32
}

// If-else with computation
// CHECK-LABEL: wasmstack.func @if_with_compute
func.func @if_with_compute(%selector: i32, %a: i32, %b: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %cond = arith.cmpi ne, %selector, %zero : i32
  %result = scf.if %cond -> (i32) {
    %sum = arith.addi %a, %b : i32
    scf.yield %sum : i32
  } else {
    %diff = arith.subi %a, %b : i32
    scf.yield %diff : i32
  }
  return %result : i32
}

// Max function using if
// CHECK-LABEL: wasmstack.func @max
func.func @max(%a: i32, %b: i32) -> i32 {
  %cond = arith.cmpi sgt, %a, %b : i32
  %result = scf.if %cond -> (i32) {
    scf.yield %a : i32
  } else {
    scf.yield %b : i32
  }
  return %result : i32
}

// Abs function using if
// CHECK-LABEL: wasmstack.func @abs
func.func @abs(%x: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %neg = arith.subi %zero, %x : i32
  %cond = arith.cmpi sge, %x, %zero : i32
  %result = scf.if %cond -> (i32) {
    scf.yield %x : i32
  } else {
    scf.yield %neg : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Control Flow - For Loops
//===----------------------------------------------------------------------===//

// Simple counting loop
// CHECK-LABEL: wasmstack.func @count_loop
func.func @count_loop(%n: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %n_idx = arith.index_cast %n : i32 to index

  %sum = scf.for %i = %c0 to %n_idx step %c1 iter_args(%acc = %init) -> (i32) {
    %new = arith.addi %acc, %one : i32
    scf.yield %new : i32
  }
  return %sum : i32
}

//===----------------------------------------------------------------------===//
// Control Flow - While Loops
//===----------------------------------------------------------------------===//

// Simple while loop
// CHECK-LABEL: wasmstack.func @simple_while
func.func @simple_while(%init: i32, %limit: i32) -> i32 {
  %c1 = arith.constant 1 : i32

  %result = scf.while (%x = %init) : (i32) -> (i32) {
    %cond = arith.cmpi slt, %x, %limit : i32
    scf.condition(%cond) %x : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return %result : i32
}

// Factorial using while
// CHECK-LABEL: wasmstack.func @factorial
func.func @factorial(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  %result:2 = scf.while (%fact = %c1, %i = %n) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi sgt, %i, %c0 : i32
    scf.condition(%cond) %fact, %i : i32, i32
  } do {
  ^bb0(%f: i32, %iv: i32):
    %new_fact = arith.muli %f, %iv : i32
    %new_i = arith.subi %iv, %c1 : i32
    scf.yield %new_fact, %new_i : i32, i32
  }
  return %result#0 : i32
}

// GCD using while
// CHECK-LABEL: wasmstack.func @gcd
func.func @gcd(%a: i32, %b: i32) -> i32 {
  %c0 = arith.constant 0 : i32

  %result:2 = scf.while (%x = %a, %y = %b) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi ne, %y, %c0 : i32
    scf.condition(%cond) %x, %y : i32, i32
  } do {
  ^bb0(%xv: i32, %yv: i32):
    %rem = arith.remsi %xv, %yv : i32
    scf.yield %yv, %rem : i32, i32
  }
  return %result#0 : i32
}

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

// Helper function
// CHECK-LABEL: wasmstack.func @helper
func.func @helper(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %result = arith.addi %x, %c1 : i32
  return %result : i32
}

// Simple call
// CHECK-LABEL: wasmstack.func @call_helper
func.func @call_helper(%x: i32) -> i32 {
  %result = func.call @helper(%x) : (i32) -> i32
  return %result : i32
}

// Chain of calls
// CHECK-LABEL: wasmstack.func @call_chain
func.func @call_chain(%x: i32) -> i32 {
  %r1 = func.call @helper(%x) : (i32) -> i32
  %r2 = func.call @helper(%r1) : (i32) -> i32
  %r3 = func.call @helper(%r2) : (i32) -> i32
  return %r3 : i32
}

// Call with computation
// CHECK-LABEL: wasmstack.func @call_with_compute
func.func @call_with_compute(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %result = func.call @helper(%sum) : (i32) -> i32
  return %result : i32
}

// Multiple parameter function
// CHECK-LABEL: wasmstack.func @multi_param
func.func @multi_param(%a: i32, %b: i32, %c: i32) -> i32 {
  %ab = arith.addi %a, %b : i32
  %result = arith.addi %ab, %c : i32
  return %result : i32
}

// Calling multi-param function
// CHECK-LABEL: wasmstack.func @call_multi_param
func.func @call_multi_param(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %result = func.call @multi_param(%x, %c1, %c2) : (i32, i32, i32) -> i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Void Functions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @void_func
func.func @void_func() {
  return
}

// CHECK-LABEL: wasmstack.func @void_with_params
func.func @void_with_params(%a: i32, %b: i32) {
  return
}

// Calling void function
// CHECK-LABEL: wasmstack.func @call_void
func.func @call_void() -> i32 {
  func.call @void_func() : () -> ()
  %c42 = arith.constant 42 : i32
  return %c42 : i32
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

// Global arrays for memory tests
memref.global @array1 : memref<4xi32> = dense<[1, 2, 3, 4]>
memref.global @array2 : memref<4xi32> = dense<[10, 20, 30, 40]>
memref.global @float_arr : memref<3xf32> = dense<[1.5, 2.5, 3.5]>

// Simple load
// CHECK-LABEL: wasmstack.func @load_element
func.func @load_element() -> i32 {
  %arr = memref.get_global @array1 : memref<4xi32>
  %c0 = arith.constant 0 : index
  %val = memref.load %arr[%c0] : memref<4xi32>
  return %val : i32
}

// Load with index computation
// CHECK-LABEL: wasmstack.func @load_indexed
func.func @load_indexed(%idx: i32) -> i32 {
  %arr = memref.get_global @array1 : memref<4xi32>
  %idx_index = arith.index_cast %idx : i32 to index
  %val = memref.load %arr[%idx_index] : memref<4xi32>
  return %val : i32
}

// Simple store
// CHECK-LABEL: wasmstack.func @store_element
func.func @store_element(%val: i32) {
  %arr = memref.get_global @array1 : memref<4xi32>
  %c0 = arith.constant 0 : index
  memref.store %val, %arr[%c0] : memref<4xi32>
  return
}

// Load, compute, store
// CHECK-LABEL: wasmstack.func @increment_element
func.func @increment_element(%idx: i32) {
  %arr = memref.get_global @array1 : memref<4xi32>
  %idx_index = arith.index_cast %idx : i32 to index
  %val = memref.load %arr[%idx_index] : memref<4xi32>
  %c1 = arith.constant 1 : i32
  %new_val = arith.addi %val, %c1 : i32
  memref.store %new_val, %arr[%idx_index] : memref<4xi32>
  return
}

// Sum array elements
// CHECK-LABEL: wasmstack.func @sum_array
func.func @sum_array() -> i32 {
  %arr = memref.get_global @array1 : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %v0 = memref.load %arr[%c0] : memref<4xi32>
  %v1 = memref.load %arr[%c1] : memref<4xi32>
  %v2 = memref.load %arr[%c2] : memref<4xi32>
  %v3 = memref.load %arr[%c3] : memref<4xi32>

  %s1 = arith.addi %v0, %v1 : i32
  %s2 = arith.addi %s1, %v2 : i32
  %sum = arith.addi %s2, %v3 : i32
  return %sum : i32
}

// Float array operations
// CHECK-LABEL: wasmstack.func @sum_float_array
func.func @sum_float_array() -> f32 {
  %arr = memref.get_global @float_arr : memref<3xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %v0 = memref.load %arr[%c0] : memref<3xf32>
  %v1 = memref.load %arr[%c1] : memref<3xf32>
  %v2 = memref.load %arr[%c2] : memref<3xf32>

  %s1 = arith.addf %v0, %v1 : f32
  %sum = arith.addf %s1, %v2 : f32
  return %sum : f32
}

// Copy between arrays
// CHECK-LABEL: wasmstack.func @copy_element
func.func @copy_element(%src_idx: i32, %dst_idx: i32) {
  %src = memref.get_global @array1 : memref<4xi32>
  %dst = memref.get_global @array2 : memref<4xi32>
  %src_i = arith.index_cast %src_idx : i32 to index
  %dst_i = arith.index_cast %dst_idx : i32 to index

  %val = memref.load %src[%src_i] : memref<4xi32>
  memref.store %val, %dst[%dst_i] : memref<4xi32>
  return
}

//===----------------------------------------------------------------------===//
// Complex Patterns
//===----------------------------------------------------------------------===//

// Power function (x^n)
// CHECK-LABEL: wasmstack.func @power
func.func @power(%x: i32, %n: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 1 : i32
  %n_idx = arith.index_cast %n : i32 to index

  %result = scf.for %i = %c0 to %n_idx step %c1 iter_args(%acc = %init) -> (i32) {
    %new = arith.muli %acc, %x : i32
    scf.yield %new : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

// Empty function
// CHECK-LABEL: wasmstack.func @empty
func.func @empty() -> i32 {
  %c0 = arith.constant 0 : i32
  return %c0 : i32
}

// Single instruction
// CHECK-LABEL: wasmstack.func @identity
func.func @identity(%x: i32) -> i32 {
  return %x : i32
}

// Multiple return values
// CHECK-LABEL: wasmstack.func @swap
func.func @swap(%a: i32, %b: i32) -> (i32, i32) {
  return %b, %a : i32, i32
}

// Using multiple return values
// CHECK-LABEL: wasmstack.func @use_swap
func.func @use_swap(%x: i32, %y: i32) -> i32 {
  %a, %b = func.call @swap(%x, %y) : (i32, i32) -> (i32, i32)
  %result = arith.addi %a, %b : i32
  return %result : i32
}

