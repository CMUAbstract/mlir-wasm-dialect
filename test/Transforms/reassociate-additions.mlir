// RUN: wasm-opt %s --reassociate-additions | FileCheck %s

func.func private @use_i32(i32) -> ()
func.func private @use_index(index) -> ()

//===----------------------------------------------------------------------===//
// Test 1: Basic 2-candidate group (same base, different constants)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_two_candidates
// CHECK-SAME:    (%[[N:.*]]: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for %[[I:.*]] =
// CHECK:           %[[IV:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           %[[ANCHOR:.*]] = arith.addi %[[BASE]], %[[IV]] : i32
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:           call @use_i32
// CHECK:           call @use_i32
func.func @basic_two_candidates(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %h1 = arith.addi %base, %c8 : i32
  %h2 = arith.addi %base, %c16 : i32
  scf.for %i = %c0 to %n step %c1 {
    %iv = arith.index_cast %i : index to i32
    %r1 = arith.addi %iv, %h1 : i32
    %r2 = arith.addi %iv, %h2 : i32
    func.call @use_i32(%r1) : (i32) -> ()
    func.call @use_i32(%r2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: Three-candidate group
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @three_candidates
// CHECK-SAME:    (%[[N:.*]]: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for %[[I:.*]] =
// CHECK:           %[[IV:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           %[[ANCHOR:.*]] = arith.addi %[[BASE]], %[[IV]] : i32
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:           call @use_i32
// CHECK:           call @use_i32
// CHECK:           call @use_i32
func.func @three_candidates(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %h1 = arith.addi %base, %c4 : i32
  %h2 = arith.addi %base, %c8 : i32
  %h3 = arith.addi %base, %c16 : i32
  scf.for %i = %c0 to %n step %c1 {
    %iv = arith.index_cast %i : index to i32
    %r1 = arith.addi %iv, %h1 : i32
    %r2 = arith.addi %iv, %h2 : i32
    %r3 = arith.addi %iv, %h3 : i32
    func.call @use_i32(%r1) : (i32) -> ()
    func.call @use_i32(%r2) : (i32) -> ()
    func.call @use_i32(%r3) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: Deep chain (3-level addi flattening)
// h2 = addi(addi(base, 8), 4) flattens to base + 12
// h3 = addi(base, 4) flattens to base + 4
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @deep_chain
// CHECK-SAME:    (%[[N:.*]]: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for %[[I:.*]] =
// CHECK:           %[[IV:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           %[[ANCHOR:.*]] = arith.addi %[[BASE]], %[[IV]] : i32
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:           call @use_i32
// CHECK:           call @use_i32
func.func @deep_chain(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %h1 = arith.addi %base, %c8 : i32
  %h2 = arith.addi %h1, %c4 : i32     // base + 12
  %h3 = arith.addi %base, %c4 : i32   // base + 4
  scf.for %i = %c0 to %n step %c1 {
    %iv = arith.index_cast %i : index to i32
    %r1 = arith.addi %iv, %h2 : i32
    %r2 = arith.addi %iv, %h3 : i32
    func.call @use_i32(%r1) : (i32) -> ()
    func.call @use_i32(%r2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Index type candidates
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @index_type
// CHECK-SAME:    (%[[N:.*]]: index, %[[BASE:.*]]: index)
// CHECK:         scf.for %[[I:.*]] =
// CHECK:           %[[ANCHOR:.*]] = arith.addi %[[BASE]], %[[I]] : index
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : index
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : index
// CHECK:           call @use_index
// CHECK:           call @use_index
func.func @index_type(%n: index, %base: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %h1 = arith.addi %base, %c8 : index
  %h2 = arith.addi %base, %c16 : index
  scf.for %i = %c0 to %n step %c1 {
    %r1 = arith.addi %i, %h1 : index
    %r2 = arith.addi %i, %h2 : index
    func.call @use_index(%r1) : (index) -> ()
    func.call @use_index(%r2) : (index) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Nested loops (inner loop reassociated independently)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nested_loops
// CHECK:         scf.for
// CHECK:           scf.for %[[J:.*]] =
// CHECK:             %[[JV:.*]] = arith.index_cast %[[J]] : index to i32
// CHECK:             %[[ANCHOR:.*]] = arith.addi %{{.*}}, %[[JV]] : i32
// CHECK:             arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:             arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:             call @use_i32
// CHECK:             call @use_i32
func.func @nested_loops(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %h1 = arith.addi %base, %c8 : i32
  %h2 = arith.addi %base, %c16 : i32
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %jv = arith.index_cast %j : index to i32
      %r1 = arith.addi %jv, %h1 : i32
      %r2 = arith.addi %jv, %h2 : i32
      func.call @use_i32(%r1) : (i32) -> ()
      func.call @use_i32(%r2) : (i32) -> ()
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Zero-constant candidate in the group
// One candidate has constSum=0, replaced with anchor directly.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @zero_const_candidate
// CHECK-SAME:    (%[[N:.*]]: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for %[[I:.*]] =
// CHECK:           %[[IV:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           %[[ANCHOR:.*]] = arith.addi %[[BASE]], %[[IV]] : i32
// CHECK:           arith.addi %[[ANCHOR]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[ANCHOR]])
// CHECK:           call @use_i32
func.func @zero_const_candidate(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  %h1 = arith.addi %base, %c8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %iv = arith.index_cast %i : index to i32
    %r1 = arith.addi %iv, %h1 : i32
    // This addi flattens to {base, iv} + 0 — same non-const terms as r1
    %r2 = arith.addi %iv, %base : i32
    func.call @use_i32(%r2) : (i32) -> ()
    func.call @use_i32(%r1) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Negative tests
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 7: No constants in tree (negative — single candidate, no transformation)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_constants
// CHECK:         scf.for
// CHECK:           arith.addi
// CHECK:           call @use_i32
// CHECK:         }
func.func @no_constants(%n: index, %x: i32, %y: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %a = arith.addi %x, %y : i32
    func.call @use_i32(%a) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 8: Single candidate (negative — group size 1, skipped)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @single_candidate
// CHECK:         scf.for
// CHECK:           arith.index_cast
// CHECK:           arith.addi
// CHECK:           call @use_i32
// CHECK:         }
func.func @single_candidate(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  %h = arith.addi %base, %c8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %iv = arith.index_cast %i : index to i32
    %r = arith.addi %iv, %h : i32
    func.call @use_i32(%r) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 9: Different non-const terms (negative — no shared group)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @different_terms
// CHECK:         scf.for
// CHECK:           arith.addi
// CHECK:           arith.addi
// CHECK:           call @use_i32
// CHECK:           call @use_i32
// CHECK:         }
func.func @different_terms(%n: index, %base1: i32, %base2: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %h1 = arith.addi %base1, %c8 : i32
  %h2 = arith.addi %base2, %c16 : i32
  scf.for %i = %c0 to %n step %c1 {
    %iv = arith.index_cast %i : index to i32
    // r1 has terms {iv, base1}, r2 has terms {iv, base2} — different groups
    %r1 = arith.addi %iv, %h1 : i32
    %r2 = arith.addi %iv, %h2 : i32
    func.call @use_i32(%r1) : (i32) -> ()
    func.call @use_i32(%r2) : (i32) -> ()
  }
  return
}
