// RUN: wasm-opt %s --reassociate-reductions | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 1: 4x max chain via cmpi+select → tree reduction
//
// Original: max(max(max(max(acc, a), b), c), d)  — 4-deep from acc
// Expected: max(acc, max(max(a, b), max(c, d)))   — 1-deep from acc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @max_chain_4x
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] =
// Three tree cmpi+selects (not referencing ACC):
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.cmpi sge
// CHECK: arith.select
// Single acc-dependent reduction:
// CHECK: arith.cmpi sge, %[[ACC]],
// CHECK: %[[R:.*]] = arith.select
// CHECK: scf.yield %[[R]]
func.func @max_chain_4x(%n: index, %a: i32, %b: i32, %c: i32, %d: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %cmp0 = arith.cmpi sge, %acc, %a : i32
    %r0 = arith.select %cmp0, %acc, %a : i32
    %cmp1 = arith.cmpi sge, %r0, %b : i32
    %r1 = arith.select %cmp1, %r0, %b : i32
    %cmp2 = arith.cmpi sge, %r1, %c : i32
    %r2 = arith.select %cmp2, %r1, %c : i32
    %cmp3 = arith.cmpi sge, %r2, %d : i32
    %r3 = arith.select %cmp3, %r2, %d : i32
    scf.yield %r3 : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: 3x addi chain → tree reduction
//
// Original: add(add(add(acc, a), b), c)  — 3-deep
// Expected: add(acc, add(add(a, b), c))  — 1-deep from acc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_chain_3x
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] =
// Tree reductions (not using ACC):
// CHECK: arith.addi
// CHECK: arith.addi
// Final acc-dependent:
// CHECK: %[[R:.*]] = arith.addi %[[ACC]],
// CHECK: scf.yield %[[R]]
func.func @addi_chain_3x(%n: index, %a: i32, %b: i32, %c: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %r0 = arith.addi %acc, %a : i32
    %r1 = arith.addi %r0, %b : i32
    %r2 = arith.addi %r1, %c : i32
    scf.yield %r2 : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: 2-element chain — no change (too short for tree benefit)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @chain_too_short
// Chain length 2 < 3, no reassociation.
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] =
// ACC should be in the FIRST cmpi (original order preserved):
// CHECK: arith.cmpi sge, %[[ACC]],
// CHECK: arith.select
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: scf.yield
func.func @chain_too_short(%n: index, %a: i32, %b: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %cmp0 = arith.cmpi sge, %acc, %a : i32
    %r0 = arith.select %cmp0, %acc, %a : i32
    %cmp1 = arith.cmpi sge, %r0, %b : i32
    %r1 = arith.select %cmp1, %r0, %b : i32
    scf.yield %r1 : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Non-associative op (subi) — no change
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @non_associative_subi
// subi is not recognized as an associative reduction.
// CHECK: scf.for
// CHECK: arith.subi
// CHECK: arith.subi
// CHECK: arith.subi
// CHECK: scf.yield
func.func @non_associative_subi(%n: index, %a: i32, %b: i32, %c: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %r0 = arith.subi %acc, %a : i32
    %r1 = arith.subi %r0, %b : i32
    %r2 = arith.subi %r1, %c : i32
    scf.yield %r2 : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Fresh operand depends on iter_arg — no change
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fresh_depends_on_acc
// Chain broken because dep depends on iter_arg.
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] =
// Original sequential order preserved (ACC in first cmpi):
// CHECK: arith.cmpi sge, %[[ACC]],
// CHECK: arith.select
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: scf.yield
func.func @fresh_depends_on_acc(%n: index, %a: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %dep = arith.muli %acc, %c2_i32 : i32
    %cmp0 = arith.cmpi sge, %acc, %a : i32
    %r0 = arith.select %cmp0, %acc, %a : i32
    %cmp1 = arith.cmpi sge, %r0, %dep : i32
    %r1 = arith.select %cmp1, %r0, %dep : i32
    %cmp2 = arith.cmpi sge, %r1, %a : i32
    %r2 = arith.select %cmp2, %r1, %a : i32
    scf.yield %r2 : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: 5-element chain (odd) → tree with leftover
//
// Original: max(max(max(max(max(acc,a),b),c),d),e) — 5-deep
// Expected: max(acc, tree(a,b,c,d,e))              — 1-deep from acc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @max_chain_5x
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] =
// Four tree cmpi+selects (not using ACC):
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.cmpi sge
// CHECK: arith.select
// Final acc-dependent:
// CHECK: arith.cmpi sge, %[[ACC]],
// CHECK: %[[R:.*]] = arith.select
// CHECK: scf.yield %[[R]]
func.func @max_chain_5x(%n: index, %a: i32, %b: i32, %c: i32, %d: i32, %e: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %cmp0 = arith.cmpi sge, %acc, %a : i32
    %r0 = arith.select %cmp0, %acc, %a : i32
    %cmp1 = arith.cmpi sge, %r0, %b : i32
    %r1 = arith.select %cmp1, %r0, %b : i32
    %cmp2 = arith.cmpi sge, %r1, %c : i32
    %r2 = arith.select %cmp2, %r1, %c : i32
    %cmp3 = arith.cmpi sge, %r2, %d : i32
    %r3 = arith.select %cmp3, %r2, %d : i32
    %cmp4 = arith.cmpi sge, %r3, %e : i32
    %r4 = arith.select %cmp4, %r3, %e : i32
    scf.yield %r4 : i32
  }
  return
}
