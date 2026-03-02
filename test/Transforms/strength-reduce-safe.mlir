// RUN: wasm-opt %s --strength-reduce | FileCheck %s

func.func private @use_i32(i32) -> ()
func.func private @use_i64(i64) -> ()
func.func private @use_index(index) -> ()

//===----------------------------------------------------------------------===//
// Test 1: IV only used through single muli — IV dead after replacement
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @safe_single_muli
// CHECK-DAG:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @safe_single_muli(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: IV used through multiple mulis — all are candidates, IV dead
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @safe_multiple_mulis
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC1:.*]] = %{{.*}}, %[[ACC2:.*]] = %{{.*}}) -> (i32, i32)
// CHECK-NOT:       arith.muli
// CHECK-DAG:       call @use_i32(%[[ACC1]])
// CHECK-DAG:       call @use_i32(%[[ACC2]])
// CHECK-DAG:       arith.addi %[[ACC1]],
// CHECK-DAG:       arith.addi %[[ACC2]],
// CHECK:           scf.yield
func.func @safe_multiple_mulis(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul1) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: muli absorbed into addi — IV dead after erasure
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @safe_muli_absorbed_by_addi
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[BASE]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @safe_muli_absorbed_by_addi(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %base = arith.constant 100 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    %addr = arith.addi %mul, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Negative — IV cast has non-candidate use (IV still live)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nosafe_iv_cast_still_live
// CHECK:         scf.for
// CHECK-NOT:       iter_args
// CHECK:           arith.muli
// CHECK:         }
func.func @nosafe_iv_cast_still_live(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
    // Extra use of cast — IV remains live after muli erasure
    func.call @use_i32(%cast) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Negative — IV has direct non-candidate use
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nosafe_iv_direct_use
// CHECK:         scf.for
// CHECK-NOT:       iter_args
// CHECK:           arith.muli
// CHECK:         }
func.func @nosafe_iv_direct_use(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %n step %c1 {
    %mul = arith.muli %i, %c4 : index
    func.call @use_index(%mul) : (index) -> ()
    // Direct use of IV — remains live
    func.call @use_index(%i) : (index) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Phase 0 + safe — muli(cast(iv+k), factor), iv+k has no other uses
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @safe_phase0_decompose
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : i32
// CHECK:         %[[INIT:.*]] = arith.muli %{{.*}}, %[[C8]] : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C8]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @safe_phase0_decompose(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %ip1 = arith.addi %i, %c1 : index
    %cast = arith.index_cast %ip1 : index to i32
    %mul = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}
