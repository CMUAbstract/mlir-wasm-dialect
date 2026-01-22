// RUN: wasm-opt %s --wami-convert-scf --wami-convert-arith --wami-convert-func --reconcile-unrealized-casts --convert-to-wasmstack 2>&1 | FileCheck %s

// Test end-to-end conversion of SCF control flow operations through WasmSSA to WasmStack dialect.
// This verifies that block/loop parameters and results are correctly handled during stackification.

// CHECK: ConvertToWasmStack pass running on module

module {
  //===--------------------------------------------------------------------===//
  // Test 1: Simple if with result type (no input parameters)
  //===--------------------------------------------------------------------===//

  // An if that returns a constant value from each branch
  // CHECK-LABEL: wasmstack.func @if_with_result
  // CHECK:         wasmstack.i32.const 1
  // CHECK:         wasmstack.if : ([]) -> [i32] then
  // CHECK:           wasmstack.i32.const 100
  // CHECK:         } else {
  // CHECK:           wasmstack.i32.const 200
  // CHECK:         }
  func.func @if_with_result() -> i32 {
    %cond = arith.constant 1 : i1
    %result = scf.if %cond -> (i32) {
      %t = arith.constant 100 : i32
      scf.yield %t : i32
    } else {
      %e = arith.constant 200 : i32
      scf.yield %e : i32
    }
    return %result : i32
  }

  //===--------------------------------------------------------------------===//
  // Test 2: For loop with iteration variable (block+loop with parameters)
  //===--------------------------------------------------------------------===//

  // A for loop that sums values - tests block and loop with input parameters
  // CHECK-LABEL: wasmstack.func @for_loop_sum
  // Block wrapping the loop should have params: counter_init, acc_init
  // CHECK:         wasmstack.block @block_{{[0-9]+}} : ([i32, i32]) -> [i32]
  // Inner loop also has params: counter, accumulator
  // CHECK:           wasmstack.loop @loop_{{[0-9]+}} : ([i32, i32]) -> []
  func.func @for_loop_sum() -> i32 {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %sum = scf.for %i = %c0 to %c5 step %c1 iter_args(%acc = %init) -> (i32) {
      %new_acc = arith.addi %acc, %one : i32
      scf.yield %new_acc : i32
    }
    return %sum : i32
  }

  //===--------------------------------------------------------------------===//
  // Test 3: For loop with multiple iter_args
  //===--------------------------------------------------------------------===//

  // Tests that multiple block parameters are handled correctly
  // CHECK-LABEL: wasmstack.func @for_loop_multi_iter
  // CHECK:         wasmstack.block @block_{{[0-9]+}} : ([i32, i32, i32]) -> [i32, i32]
  // CHECK:           wasmstack.loop @loop_{{[0-9]+}} : ([i32, i32, i32]) -> []
  func.func @for_loop_multi_iter() -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %init1 = arith.constant 0 : i32
    %init2 = arith.constant 1 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %sum, %prod = scf.for %i = %c0 to %c3 step %c1 iter_args(%s = %init1, %p = %init2) -> (i32, i32) {
      %new_s = arith.addi %s, %one : i32
      %new_p = arith.muli %p, %two : i32
      scf.yield %new_s, %new_p : i32, i32
    }
    return %sum, %prod : i32, i32
  }

  //===--------------------------------------------------------------------===//
  // Test 4: While loop
  //===--------------------------------------------------------------------===//

  // Tests while loop conversion with parameters
  // CHECK-LABEL: wasmstack.func @while_loop
  // CHECK:         wasmstack.block @block_{{[0-9]+}} : ([i32]) -> [i32]
  // CHECK:           wasmstack.loop @loop_{{[0-9]+}} : ([i32]) -> []
  func.func @while_loop(%init: i32) -> i32 {
    %limit = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
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

  //===--------------------------------------------------------------------===//
  // Test 5: If with multiple results
  //===--------------------------------------------------------------------===//

  // Tests if with multiple result values
  // CHECK-LABEL: wasmstack.func @if_multi_result
  // CHECK:         wasmstack.if : ([]) -> [i32, i32] then
  func.func @if_multi_result(%cond: i1) -> (i32, i32) {
    %r1, %r2 = scf.if %cond -> (i32, i32) {
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      scf.yield %c1, %c2 : i32, i32
    } else {
      %c3 = arith.constant 3 : i32
      %c4 = arith.constant 4 : i32
      scf.yield %c3, %c4 : i32, i32
    }
    return %r1, %r2 : i32, i32
  }

  //===--------------------------------------------------------------------===//
  // Test 6: For loop with no iter_args (block has no results)
  //===--------------------------------------------------------------------===//

  // Tests that blocks with no results work correctly
  // CHECK-LABEL: wasmstack.func @for_no_iter_args
  // CHECK:         wasmstack.block @block_{{[0-9]+}} : ([i32]) -> []
  // CHECK:           wasmstack.loop @loop_{{[0-9]+}} : ([i32]) -> []
  func.func @for_no_iter_args(%n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      // Empty body
    }
    return
  }
}
