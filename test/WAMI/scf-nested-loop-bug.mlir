// RUN: wasm-opt %s --convert-scf-to-ssawasm --convert-arith-to-ssawasm --convert-func-to-ssawasm --reconcile-unrealized-casts 2>&1 | FileCheck %s
// XFAIL: *

// This test documents a bug in nested loop conversion.
// The ForOpLowering pattern uses RewritePattern instead of OpConversionPattern,
// causing issues when inner loops reference values from outer scopes that
// haven't been type-converted yet.
//
// Bug location: lib/SsaWasm/ConversionPatterns/ScfToSsaWasm.cpp, ForOpLowering
//
// When this test passes, the bug has been fixed.

// CHECK-NOT: error
// CHECK-NOT: failed to legalize

//===----------------------------------------------------------------------===//
// Bug 1: Nested loops with function parameter bounds
//===----------------------------------------------------------------------===//

// The outer loop's bound %n comes from a function parameter (index type).
// The inner loop also uses %m from function parameter.
// The conversion fails because when the inner loop is processed,
// the bounds are still index type instead of i32.

func.func @nested_loops_param_bounds(%n: index, %m: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32

  %outer = scf.for %i = %c0 to %n step %c1 iter_args(%outer_acc = %init) -> (i32) {
    %inner = scf.for %j = %c0 to %m step %c1 iter_args(%inner_acc = %outer_acc) -> (i32) {
      %new = arith.addi %inner_acc, %one : i32
      scf.yield %new : i32
    }
    scf.yield %inner : i32
  }
  return %outer : i32
}

//===----------------------------------------------------------------------===//
// Bug 2: Triple nested loops
//===----------------------------------------------------------------------===//

func.func @triple_nested_loops(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32

  %r1 = scf.for %i = %c0 to %n step %c1 iter_args(%a1 = %init) -> (i32) {
    %r2 = scf.for %j = %c0 to %n step %c1 iter_args(%a2 = %a1) -> (i32) {
      %r3 = scf.for %k = %c0 to %n step %c1 iter_args(%a3 = %a2) -> (i32) {
        %new = arith.addi %a3, %one : i32
        scf.yield %new : i32
      }
      scf.yield %r3 : i32
    }
    scf.yield %r2 : i32
  }
  return %r1 : i32
}

//===----------------------------------------------------------------------===//
// Bug 3: Nested loop with outer induction variable used in inner bound
//===----------------------------------------------------------------------===//

// This is an even more complex case where the inner loop's bound
// depends on the outer loop's induction variable.

func.func @nested_loop_dependent_bound(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32

  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    // Inner loop bound depends on outer induction variable
    %inner = scf.for %j = %c0 to %i step %c1 iter_args(%inner_acc = %acc) -> (i32) {
      %new = arith.addi %inner_acc, %one : i32
      scf.yield %new : i32
    }
    scf.yield %inner : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Bug 4: Nested loops with multiple iter_args
//===----------------------------------------------------------------------===//

func.func @nested_loops_multi_iter(%n: index, %m: index) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init1 = arith.constant 0 : i32
  %init2 = arith.constant 1 : i32
  %one = arith.constant 1 : i32

  %r1, %r2 = scf.for %i = %c0 to %n step %c1 iter_args(%a1 = %init1, %a2 = %init2) -> (i32, i32) {
    %s1, %s2 = scf.for %j = %c0 to %m step %c1 iter_args(%b1 = %a1, %b2 = %a2) -> (i32, i32) {
      %new1 = arith.addi %b1, %one : i32
      %new2 = arith.muli %b2, %b1 : i32
      scf.yield %new1, %new2 : i32, i32
    }
    scf.yield %s1, %s2 : i32, i32
  }
  return %r1, %r2 : i32, i32
}
