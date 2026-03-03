// RUN: wasm-opt %s --affine-simplify-if | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 1: Always-true single inequality — affine.if eliminated
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @always_true_simple
// CHECK-NOT:     affine.if
// CHECK:         arith.constant 42
// CHECK:         return
func.func @always_true_simple() {
  %alloc = memref.alloc() : memref<200xi32>
  %c42 = arith.constant 42 : i32
  affine.for %i = 1 to 10 {
    // %i >= 1 is always true since the loop starts at 1.
    affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%i) {
      affine.store %c42, %alloc[%i] : memref<200xi32>
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: Always-true conjunction — both constraints implied
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @always_true_conjunction
// CHECK-NOT:     affine.if
// CHECK:         affine.store
func.func @always_true_conjunction() {
  %alloc = memref.alloc() : memref<200x200xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 1 to 10 {
    affine.for %j = 1 to 20 {
      // Both %i >= 1 and %j >= 1 are always true.
      affine.if affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 1 >= 0)>(%i, %j) {
        affine.store %c1, %alloc[%i, %j] : memref<200x200xi32>
      }
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: Not always true — condition can be false, preserved
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @not_always_true
// CHECK:         affine.if
func.func @not_always_true() {
  %alloc = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 10 {
    // %i >= 5 is NOT always true (false for i = 0..4).
    affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
      affine.store %c1, %alloc[%i] : memref<200xi32>
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Always-false — condition impossible given loop bounds
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @always_false
// CHECK-NOT:     affine.if
// CHECK-NOT:     affine.store
func.func @always_false() {
  %alloc = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 10 {
    // %i >= 100 is always false since i < 10.
    affine.if affine_set<(d0) : (d0 - 100 >= 0)>(%i) {
      affine.store %c1, %alloc[%i] : memref<200xi32>
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Always-true with else block — then-block inlined, else removed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @always_true_with_else
// CHECK-NOT:     affine.if
// CHECK:         arith.constant 1
// CHECK:         affine.store
// CHECK-NOT:     arith.constant 2
func.func @always_true_with_else() {
  %alloc = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  affine.for %i = 1 to 10 {
    affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%i) {
      affine.store %c1, %alloc[%i] : memref<200xi32>
    } else {
      affine.store %c2, %alloc[%i] : memref<200xi32>
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Always-false with else block — else-block inlined
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @always_false_with_else
// CHECK-NOT:     affine.if
// CHECK:         arith.constant 2
// CHECK:         affine.store
func.func @always_false_with_else() {
  %alloc = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  affine.for %i = 0 to 10 {
    affine.if affine_set<(d0) : (d0 - 100 >= 0)>(%i) {
      affine.store %c1, %alloc[%i] : memref<200xi32>
    } else {
      affine.store %c2, %alloc[%i] : memref<200xi32>
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 7: Nussinov pattern — affine_map lower bound
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nussinov_pattern
// CHECK-NOT:     affine.if
// CHECK:         affine.store
#map_nuss = affine_map<(d0) -> (-d0 + 500)>
func.func @nussinov_pattern() {
  %alloc = memref.alloc() : memref<501xi32>
  %c1 = arith.constant 1 : i32
  affine.for %arg0 = 0 to 500 {
    // %arg1 ranges from (500 - arg0) to 500, so arg1 >= 500 - 499 = 1.
    affine.for %arg1 = #map_nuss(%arg0) to 500 {
      // arg1 >= 1 is always true.
      affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%arg1) {
        affine.store %c1, %alloc[%arg1] : memref<501xi32>
      }
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 8: Skip affine.if with results — preserved
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @skip_if_with_results
// CHECK:         affine.if
func.func @skip_if_with_results() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = affine.for %i = 1 to 10 iter_args(%acc = %c0) -> (i32) {
    %v = affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%i) -> i32 {
      affine.yield %c1 : i32
    } else {
      affine.yield %c0 : i32
    }
    %new = arith.addi %acc, %v : i32
    affine.yield %new : i32
  }
  return %r : i32
}

//===----------------------------------------------------------------------===//
// Test 9: No enclosing loop — preserved (nothing to derive bounds from)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_enclosing_loop
// CHECK:         affine.if
func.func @no_enclosing_loop(%arg0: index) {
  %alloc = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%arg0) {
    affine.store %c1, %alloc[%arg0] : memref<200xi32>
  }
  return
}
