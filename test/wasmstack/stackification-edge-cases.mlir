// RUN: wasm-opt %s --wami-convert-scf --wami-convert-arith --wami-convert-func --reconcile-unrealized-casts --convert-to-wasmstack 2>&1 | FileCheck %s
// RUN: wasm-opt %s --wami-convert-scf --wami-convert-arith --wami-convert-func --reconcile-unrealized-casts --convert-to-wasmstack -verify-wasmstack 2>&1 | FileCheck %s --check-prefix=VERIFY

// Edge cases in stackification: WasmSSA -> WasmStack conversion
// Tests complex value orderings, multi-use values, and control flow interactions

// Verify conversion runs
// CHECK: ConvertToWasmStack pass running on module
// VERIFY-NOT: error

// Verify no unrealized conversion casts remain
// CHECK-NOT: unrealized_conversion_cast

//===----------------------------------------------------------------------===//
// Test 1: Value used multiple times (requires local spill)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @multi_use_value
func.func @multi_use_value(%x: i32) -> i32 {
  // %x is used 3 times - should use local.tee or multiple local.get
  %r1 = arith.addi %x, %x : i32
  %r2 = arith.muli %r1, %x : i32
  return %r2 : i32
}

//===----------------------------------------------------------------------===//
// Test 2: Non-tree DAG pattern (shared subexpression)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @shared_subexpr
func.func @shared_subexpr(%a: i32, %b: i32) -> i32 {
  // (a + b) is used twice
  %sum = arith.addi %a, %b : i32
  %prod = arith.muli %sum, %sum : i32
  return %prod : i32
}

//===----------------------------------------------------------------------===//
// Test 3: Cross-expression value ordering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @cross_expr_ordering
func.func @cross_expr_ordering(%a: i32, %b: i32, %c: i32) -> i32 {
  // Interleaved computations that may require reordering
  %ab = arith.addi %a, %b : i32
  %bc = arith.addi %b, %c : i32
  %ac = arith.addi %a, %c : i32
  %r1 = arith.muli %ab, %bc : i32
  %r2 = arith.muli %r1, %ac : i32
  return %r2 : i32
}

//===----------------------------------------------------------------------===//
// Test 4: Value defined in one branch, used after control flow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @value_across_cf
func.func @value_across_cf(%cond: i1, %x: i32, %y: i32) -> i32 {
  // Compute before the if
  %pre = arith.addi %x, %y : i32

  // Result from if depends on %pre
  %result = scf.if %cond -> (i32) {
    %r = arith.addi %pre, %x : i32
    scf.yield %r : i32
  } else {
    %r = arith.subi %pre, %y : i32
    scf.yield %r : i32
  }

  // Use %pre again after the if
  %final = arith.muli %result, %pre : i32
  return %final : i32
}

//===----------------------------------------------------------------------===//
// Test 5: Loop with value from before loop used inside
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @outer_value_in_loop
func.func @outer_value_in_loop(%n: index, %base: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32

  // %base is defined outside but used inside loop
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %new_acc = arith.addi %acc, %base : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 6: Multiple results with complex dependency
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @multi_result_deps
func.func @multi_result_deps(%a: i32, %b: i32) -> (i32, i32) {
  %sum = arith.addi %a, %b : i32
  %diff = arith.subi %a, %b : i32
  // Results depend on each other
  %prod = arith.muli %sum, %diff : i32
  %quot = arith.divsi %sum, %a : i32
  return %prod, %quot : i32, i32
}

//===----------------------------------------------------------------------===//
// Test 7: Chain of dependent operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @long_dependency_chain
func.func @long_dependency_chain(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %r1 = arith.addi %x, %c1 : i32
  %r2 = arith.muli %r1, %c1 : i32
  %r3 = arith.subi %r2, %c1 : i32
  %r4 = arith.addi %r3, %c1 : i32
  %r5 = arith.muli %r4, %c1 : i32
  %r6 = arith.subi %r5, %c1 : i32
  %r7 = arith.addi %r6, %c1 : i32
  %r8 = arith.muli %r7, %c1 : i32
  return %r8 : i32
}

//===----------------------------------------------------------------------===//
// Test 8: Diamond pattern in control flow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @diamond_pattern
func.func @diamond_pattern(%cond1: i1, %cond2: i1, %x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32

  %r1 = scf.if %cond1 -> (i32) {
    %r = arith.addi %x, %c1 : i32
    scf.yield %r : i32
  } else {
    %r = arith.subi %x, %c1 : i32
    scf.yield %r : i32
  }

  %r2 = scf.if %cond2 -> (i32) {
    %r = arith.muli %r1, %c2 : i32
    scf.yield %r : i32
  } else {
    %r = arith.divsi %r1, %c2 : i32
    scf.yield %r : i32
  }

  return %r2 : i32
}

//===----------------------------------------------------------------------===//
// Test 9: Value used in both condition and body
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @value_in_cond_and_body
func.func @value_in_cond_and_body(%x: i32, %limit: i32) -> i32 {
  %c10 = arith.constant 10 : i32

  // %x used in condition
  %cond = arith.cmpi slt, %x, %limit : i32

  %result = scf.if %cond -> (i32) {
    // %x also used in body
    %r = arith.addi %x, %c10 : i32
    scf.yield %r : i32
  } else {
    scf.yield %x : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 10: Nested loops with shared outer variable
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @nested_loops_shared_var
func.func @nested_loops_shared_var(%n: index, %m: index, %factor: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32

  // %factor is shared between both loop levels
  %outer_result = scf.for %i = %c0 to %n step %c1 iter_args(%outer_acc = %init) -> (i32) {
    %inner_result = scf.for %j = %c0 to %m step %c1 iter_args(%inner_acc = %outer_acc) -> (i32) {
      // Use %factor from outer scope
      %new = arith.addi %inner_acc, %factor : i32
      scf.yield %new : i32
    }
    scf.yield %inner_result : i32
  }
  return %outer_result : i32
}

//===----------------------------------------------------------------------===//
// Test 11: Operand order matters for non-commutative ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @operand_order_matters
func.func @operand_order_matters(%a: i32, %b: i32) -> i32 {
  // sub and div are not commutative
  %sub = arith.subi %a, %b : i32
  %div = arith.divsi %a, %b : i32
  %r1 = arith.subi %sub, %div : i32
  %r2 = arith.divsi %div, %sub : i32
  %result = arith.addi %r1, %r2 : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 12: Constant rematerialization vs spilling
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @constant_multi_use
func.func @constant_multi_use(%x: i32) -> i32 {
  // Same constant used multiple times - should be rematerialized
  %c42 = arith.constant 42 : i32
  %r1 = arith.addi %x, %c42 : i32
  %r2 = arith.muli %r1, %c42 : i32
  %r3 = arith.subi %r2, %c42 : i32
  %r4 = arith.addi %r3, %c42 : i32
  return %r4 : i32
}

//===----------------------------------------------------------------------===//
// Test 13: If result used in subsequent if condition
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @if_chain_condition
func.func @if_chain_condition(%cond: i1, %x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32

  %r1 = scf.if %cond -> (i32) {
    scf.yield %c10 : i32
  } else {
    scf.yield %c0 : i32
  }

  // Use %r1 to compute new condition
  %cond2 = arith.cmpi sgt, %r1, %c0 : i32

  %r2 = scf.if %cond2 -> (i32) {
    %t = arith.addi %r1, %x : i32
    scf.yield %t : i32
  } else {
    scf.yield %x : i32
  }

  return %r2 : i32
}

//===----------------------------------------------------------------------===//
// Test 14: Loop counter used in complex expression
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @loop_counter_complex
func.func @loop_counter_complex(%n: index, %base: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32

  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    // Convert and use index in complex expression
    %i_i32 = arith.index_cast %i : index to i32
    %prod = arith.muli %i_i32, %base : i32
    %sum = arith.addi %prod, %acc : i32
    %final = arith.muli %sum, %i_i32 : i32
    scf.yield %final : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 15: Multiple live values across region boundary
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @multi_live_across_region
func.func @multi_live_across_region(%cond: i1, %a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
  // All four values should be available after the if
  %ab = arith.addi %a, %b : i32
  %cd = arith.addi %c, %d : i32

  %result = scf.if %cond -> (i32) {
    %r = arith.addi %ab, %cd : i32
    scf.yield %r : i32
  } else {
    %r = arith.muli %ab, %cd : i32
    scf.yield %r : i32
  }

  // Use original values after if
  %post = arith.addi %a, %d : i32
  %final = arith.addi %result, %post : i32
  return %final : i32
}

//===----------------------------------------------------------------------===//
// Test 16: Select with complex operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @select_complex_operands
func.func @select_complex_operands(%cond: i1, %a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %diff = arith.subi %a, %b : i32
  %result = arith.select %cond, %sum, %diff : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 17: Deeply nested expression tree
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @deep_expr_tree
func.func @deep_expr_tree(%a: i32, %b: i32, %c: i32, %d: i32, %e: i32, %f: i32, %g: i32, %h: i32) -> i32 {
  // Binary tree of additions
  %ab = arith.addi %a, %b : i32
  %cd = arith.addi %c, %d : i32
  %ef = arith.addi %e, %f : i32
  %gh = arith.addi %g, %h : i32
  %abcd = arith.addi %ab, %cd : i32
  %efgh = arith.addi %ef, %gh : i32
  %result = arith.addi %abcd, %efgh : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 18: While loop with multiple carried values
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @while_multi_carry
func.func @while_multi_carry(%init1: i32, %init2: i32, %limit: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : i32

  %r1, %r2 = scf.while (%arg1 = %init1, %arg2 = %init2) : (i32, i32) -> (i32, i32) {
    %sum = arith.addi %arg1, %arg2 : i32
    %cond = arith.cmpi slt, %sum, %limit : i32
    scf.condition(%cond) %arg1, %arg2 : i32, i32
  } do {
  ^bb0(%val1: i32, %val2: i32):
    %next1 = arith.addi %val1, %c1 : i32
    %next2 = arith.muli %val2, %val1 : i32
    scf.yield %next1, %next2 : i32, i32
  }

  return %r1, %r2 : i32, i32
}

//===----------------------------------------------------------------------===//
// Test 19: Comparison results feeding into multiple uses
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @comparison_multi_use
func.func @comparison_multi_use(%a: i32, %b: i32) -> i32 {
  %cmp = arith.cmpi slt, %a, %b : i32
  // Use comparison in select
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %sel = arith.select %cmp, %c1, %c0 : i32
  // Also use in conditional add (zext)
  %zext = arith.extui %cmp : i1 to i32
  %result = arith.addi %sel, %zext : i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Test 20: Function call results in expressions
//===----------------------------------------------------------------------===//

func.func private @helper(%x: i32) -> i32

// CHECK-LABEL: wasmstack.func @call_in_expression
func.func @call_in_expression(%a: i32, %b: i32) -> i32 {
  %call1 = func.call @helper(%a) : (i32) -> i32
  %call2 = func.call @helper(%b) : (i32) -> i32
  %sum = arith.addi %call1, %call2 : i32
  return %sum : i32
}
