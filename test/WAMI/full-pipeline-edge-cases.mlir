// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack -verify-wasmstack 2>&1 | FileCheck %s

// Full pipeline edge case tests: standard MLIR -> WasmSSA -> WasmStack with verification
// These tests exercise complex patterns that could expose bugs in conversion or verification

// Verify no unrealized conversion casts remain
// CHECK-NOT: unrealized_conversion_cast

//===----------------------------------------------------------------------===//
// Test 1: Nested loops with multiple iter_args crossing control flow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @nested_loops_multi_iter
func.func @nested_loops_multi_iter(%n: index, %m: index) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init_sum = arith.constant 0 : i32
  %init_prod = arith.constant 1 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32

  // Outer loop
  %r1, %r2 = scf.for %i = %c0 to %n step %c1 iter_args(%sum_outer = %init_sum, %prod_outer = %init_prod) -> (i32, i32) {
    // Inner loop
    %sum_inner, %prod_inner = scf.for %j = %c0 to %m step %c1 iter_args(%sum = %sum_outer, %prod = %prod_outer) -> (i32, i32) {
      %new_sum = arith.addi %sum, %one : i32
      %new_prod = arith.muli %prod, %two : i32
      scf.yield %new_sum, %new_prod : i32, i32
    }
    scf.yield %sum_inner, %prod_inner : i32, i32
  }
  return %r1, %r2 : i32, i32
}

//===----------------------------------------------------------------------===//
// Test 2: If-else inside loop with values from outer scope
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @if_in_loop_outer_values
func.func @if_in_loop_outer_values(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : i32
  %init = arith.constant 0 : i32

  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    // Use outer value %c10 inside conditional
    %cond = arith.cmpi slt, %acc, %c10 : i32
    %delta = scf.if %cond -> (i32) {
      // Add the outer constant
      scf.yield %c10 : i32
    } else {
      %neg = arith.constant -1 : i32
      scf.yield %neg : i32
    }
    %new_acc = arith.addi %acc, %delta : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 3: Complex expression with many operands requiring local spills
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @complex_expression_spills
func.func @complex_expression_spills(%a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
  // Expression: ((a + b) * (c - d)) + ((a - c) * (b + d))
  // This may require local spills during stackification
  %ab = arith.addi %a, %b : i32
  %cd = arith.subi %c, %d : i32
  %part1 = arith.muli %ab, %cd : i32

  %ac = arith.subi %a, %c : i32
  %bd = arith.addi %b, %d : i32
  %part2 = arith.muli %ac, %bd : i32

  %result = arith.addi %part1, %part2 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 4: Value used in multiple branches and after control flow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @value_used_across_cf
func.func @value_used_across_cf(%cond: i1, %x: i32) -> i32 {
  // %x is used in both branches and after
  %doubled = arith.muli %x, %x : i32

  %result = scf.if %cond -> (i32) {
    %r = arith.addi %doubled, %x : i32
    scf.yield %r : i32
  } else {
    %r = arith.subi %doubled, %x : i32
    scf.yield %r : i32
  }

  // Use %doubled again after the if
  %final = arith.addi %result, %doubled : i32
  return %final : i32
}

//===----------------------------------------------------------------------===//
// Test 5: While loop with complex condition and body
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @while_complex
func.func @while_complex(%init: i32, %limit: i32, %step: i32) -> i32 {
  %result = scf.while (%arg = %init) : (i32) -> (i32) {
    // Complex condition with multiple ops
    %half_limit = arith.shrsi %limit, %step : i32
    %cond1 = arith.cmpi slt, %arg, %limit : i32
    %cond2 = arith.cmpi sgt, %arg, %half_limit : i32
    // cmpi returns i1, so use ori on i1 type
    %combined = arith.ori %cond1, %cond2 : i1
    scf.condition(%combined) %arg : i32
  } do {
  ^bb0(%val: i32):
    // Complex body
    %next = arith.addi %val, %step : i32
    %doubled = arith.muli %next, %next : i32
    %final = arith.remsi %doubled, %limit : i32
    scf.yield %final : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 6: Triple nested control flow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @triple_nested
func.func @triple_nested(%cond1: i1, %cond2: i1, %cond3: i1, %x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32

  %r1 = scf.if %cond1 -> (i32) {
    %r2 = scf.if %cond2 -> (i32) {
      %r3 = scf.if %cond3 -> (i32) {
        %val = arith.addi %x, %c1 : i32
        scf.yield %val : i32
      } else {
        %val = arith.addi %x, %c2 : i32
        scf.yield %val : i32
      }
      scf.yield %r3 : i32
    } else {
      %val = arith.addi %x, %c3 : i32
      scf.yield %val : i32
    }
    scf.yield %r2 : i32
  } else {
    scf.yield %x : i32
  }
  return %r1 : i32
}

//===----------------------------------------------------------------------===//
// Test 7: Loop with early exit via if
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @loop_early_exit
func.func @loop_early_exit(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %threshold = arith.constant 100 : i32

  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %new_acc = arith.addi %acc, %acc : i32
    %new_acc2 = arith.addi %new_acc, %acc : i32
    scf.yield %new_acc2 : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 8: Multiple results from if used in computation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @multi_result_if_used
func.func @multi_result_if_used(%cond: i1, %x: i32, %y: i32) -> i32 {
  %r1, %r2 = scf.if %cond -> (i32, i32) {
    %a = arith.addi %x, %y : i32
    %b = arith.muli %x, %y : i32
    scf.yield %a, %b : i32, i32
  } else {
    %a = arith.subi %x, %y : i32
    %b = arith.divsi %x, %y : i32
    scf.yield %a, %b : i32, i32
  }

  // Use both results
  %sum = arith.addi %r1, %r2 : i32
  %final = arith.muli %sum, %r1 : i32
  return %final : i32
}

//===----------------------------------------------------------------------===//
// Test 9: Comparison operations in loop conditions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @various_comparisons
func.func @various_comparisons(%a: i32, %b: i32) -> i32 {
  %eq = arith.cmpi eq, %a, %b : i32
  %ne = arith.cmpi ne, %a, %b : i32
  %slt = arith.cmpi slt, %a, %b : i32
  %sle = arith.cmpi sle, %a, %b : i32
  %sgt = arith.cmpi sgt, %a, %b : i32
  %sge = arith.cmpi sge, %a, %b : i32
  %ult = arith.cmpi ult, %a, %b : i32
  %ule = arith.cmpi ule, %a, %b : i32

  // Chain results together
  %v1 = arith.extui %eq : i1 to i32
  %v2 = arith.extui %ne : i1 to i32
  %v3 = arith.extui %slt : i1 to i32
  %v4 = arith.extui %sle : i1 to i32
  %v5 = arith.extui %sgt : i1 to i32
  %v6 = arith.extui %sge : i1 to i32
  %v7 = arith.extui %ult : i1 to i32
  %v8 = arith.extui %ule : i1 to i32

  %r1 = arith.addi %v1, %v2 : i32
  %r2 = arith.addi %r1, %v3 : i32
  %r3 = arith.addi %r2, %v4 : i32
  %r4 = arith.addi %r3, %v5 : i32
  %r5 = arith.addi %r4, %v6 : i32
  %r6 = arith.addi %r5, %v7 : i32
  %result = arith.addi %r6, %v8 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 10: All integer types (i32, i64)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @mixed_int_types
func.func @mixed_int_types(%a32: i32, %b32: i32, %a64: i64, %b64: i64) -> i64 {
  // i32 operations
  %sum32 = arith.addi %a32, %b32 : i32

  // i64 operations
  %sum64 = arith.addi %a64, %b64 : i64

  // Extend i32 to i64
  %extended = arith.extsi %sum32 : i32 to i64

  // Combine
  %result = arith.addi %sum64, %extended : i64
  return %result : i64
}

//===----------------------------------------------------------------------===//
// Test 11: Floating point operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @float_operations
func.func @float_operations(%a: f32, %b: f32) -> f32 {
  %sum = arith.addf %a, %b : f32
  %diff = arith.subf %a, %b : f32
  %prod = arith.mulf %sum, %diff : f32
  %quot = arith.divf %prod, %a : f32
  return %quot : f32
}

//===----------------------------------------------------------------------===//
// Test 12: Select operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @select_chains
func.func @select_chains(%cond1: i1, %cond2: i1, %a: i32, %b: i32, %c: i32) -> i32 {
  %s1 = arith.select %cond1, %a, %b : i32
  %s2 = arith.select %cond2, %s1, %c : i32
  %s3 = arith.select %cond1, %s2, %s1 : i32
  return %s3 : i32
}

//===----------------------------------------------------------------------===//
// Test 13: Bitwise operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @bitwise_ops
func.func @bitwise_ops(%a: i32, %b: i32, %shift: i32) -> i32 {
  %and_val = arith.andi %a, %b : i32
  %or_val = arith.ori %a, %b : i32
  %xor_val = arith.xori %and_val, %or_val : i32
  %shl_val = arith.shli %xor_val, %shift : i32
  %shr_val = arith.shrsi %shl_val, %shift : i32
  return %shr_val : i32
}

//===----------------------------------------------------------------------===//
// Test 14: Empty loop body
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @empty_loop
func.func @empty_loop(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    // Empty body
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 15: Loop with only constants
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @loop_constants_only
func.func @loop_constants_only() -> i32 {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %add_val = arith.constant 7 : i32

  %result = scf.for %i = %c0 to %c5 step %c1 iter_args(%acc = %init) -> (i32) {
    %new_acc = arith.addi %acc, %add_val : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 16: Function with many parameters
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @many_params
func.func @many_params(%a: i32, %b: i32, %c: i32, %d: i32, %e: i32, %f: i32) -> i32 {
  %r1 = arith.addi %a, %b : i32
  %r2 = arith.addi %c, %d : i32
  %r3 = arith.addi %e, %f : i32
  %r4 = arith.addi %r1, %r2 : i32
  %result = arith.addi %r4, %r3 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 17: Division and remainder operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @div_rem_ops
func.func @div_rem_ops(%a: i32, %b: i32) -> i32 {
  %div_s = arith.divsi %a, %b : i32
  %div_u = arith.divui %a, %b : i32
  %rem_s = arith.remsi %a, %b : i32
  %rem_u = arith.remui %a, %b : i32

  %r1 = arith.addi %div_s, %div_u : i32
  %r2 = arith.addi %rem_s, %rem_u : i32
  %result = arith.addi %r1, %r2 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 18: If without else
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @if_no_else
func.func @if_no_else(%cond: i1, %x: i32) -> i32 {
  scf.if %cond {
    // Side effect only, no result
  }
  return %x : i32
}

//===----------------------------------------------------------------------===//
// Test 19: Multiple sequential loops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @sequential_loops
func.func @sequential_loops(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32

  // First loop
  %r1 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %new = arith.addi %acc, %one : i32
    scf.yield %new : i32
  }

  // Second loop starts with result of first
  %r2 = scf.for %j = %c0 to %n step %c1 iter_args(%acc = %r1) -> (i32) {
    %doubled = arith.muli %acc, %acc : i32
    scf.yield %doubled : i32
  }

  return %r2 : i32
}

//===----------------------------------------------------------------------===//
// Test 20: Constant folding opportunities
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @constant_folding
func.func @constant_folding() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32

  // These could potentially be folded
  %sum1 = arith.addi %c1, %c2 : i32  // 3
  %sum2 = arith.addi %c3, %c4 : i32  // 7
  %result = arith.muli %sum1, %sum2 : i32  // 21
  return %result : i32
}
