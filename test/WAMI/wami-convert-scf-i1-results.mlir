// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts | FileCheck %s

// Regression coverage for SCF boundary values that originate as i1.
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: : i1

module {
  // CHECK-LABEL: wasmssa.func @if_i1_result
  // CHECK: wasmssa.if
  // CHECK: wasmssa.block_return
  func.func @if_i1_result(%cond: i1) -> i32 {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32

    %flag = scf.if %cond -> (i1) {
      scf.yield %true : i1
    } else {
      scf.yield %false : i1
    }

    %r = arith.select %flag, %c0, %c0 : i32
    return %r : i32
  }

  // CHECK-LABEL: wasmssa.func @for_i1_iter_arg
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  // CHECK: wasmssa.block_return
  func.func @for_i1_iter_arg(%n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %false = arith.constant false

    %flag = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %false) -> (i1) {
      %next = arith.xori %acc, %true : i1
      scf.yield %next : i1
    }

    %r = arith.select %flag, %c0_i32, %c0_i32 : i32
    return %r : i32
  }

  // CHECK-LABEL: wasmssa.func @while_i1_carried
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  // CHECK: wasmssa.block_return
  func.func @while_i1_carried(%n: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false

    %result:2 = scf.while (%i = %c0, %flag = %false) : (i32, i1) -> (i32, i1) {
      %in_bounds = arith.cmpi slt, %i, %n : i32
      scf.condition(%in_bounds) %i, %flag : i32, i1
    } do {
    ^bb0(%i_in: i32, %flag_in: i1):
      %next_i = arith.addi %i_in, %c1 : i32
      %next_flag = arith.xori %flag_in, %true : i1
      scf.yield %next_i, %next_flag : i32, i1
    }

    %out = arith.select %result#1, %result#0, %n : i32
    return %out : i32
  }
}
