// RUN: wasm-opt %s --affine-promote-accumulators | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 1: Basic invariant accumulator (no other loads from same memref)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_accumulator
// CHECK:         %[[INIT:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:         %[[RESULT:.*]] = affine.for %{{.*}} = 0 to 100 iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK-NOT:       affine.load
// CHECK-NOT:       affine.store
// CHECK:           %[[NEW:.*]] = arith.addi %[[ACC]], %{{.*}}
// CHECK:           affine.yield %[[NEW]]
// CHECK:         affine.store %[[RESULT]], %{{.*}}[%{{.*}}, %{{.*}}]
func.func @basic_accumulator(%arg0: index, %arg1: index) {
  %alloc = memref.alloc() : memref<200x200xi32>
  %c5 = arith.constant 5 : i32
  affine.for %i = 0 to 100 {
    %v = affine.load %alloc[%arg0, %arg1] : memref<200x200xi32>
    %new = arith.addi %v, %c5 : i32
    affine.store %new, %alloc[%arg0, %arg1] : memref<200x200xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: Non-aliasing reads from same memref (nussinov-like pattern)
// Loop bounds k ∈ [arg0+1, arg1) guarantee k ≠ arg1, so no aliasing.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @non_aliasing_reads
// CHECK:         affine.load
// CHECK:         affine.for {{.*}} iter_args
// CHECK-NOT:       affine.store
// CHECK:           affine.load
// CHECK:           affine.yield
// CHECK:         affine.store
#map_lb = affine_map<(d0) -> (d0 + 1)>
func.func @non_aliasing_reads(%arg0: index, %arg1: index) {
  %alloc = memref.alloc() : memref<200x200xi32>
  affine.for %k = #map_lb(%arg0) to %arg1 {
    %acc = affine.load %alloc[%arg0, %arg1] : memref<200x200xi32>
    %v1 = affine.load %alloc[%arg0, %k] : memref<200x200xi32>
    %sum = arith.addi %acc, %v1 : i32
    %cmp = arith.cmpi sge, %acc, %sum : i32
    %sel = arith.select %cmp, %acc, %sum : i32
    affine.store %sel, %alloc[%arg0, %arg1] : memref<200x200xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: Nested loop — inner accumulator promoted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nested_inner_promoted
// CHECK:         affine.for %{{.*}} = 0 to 100
// CHECK:           %[[INIT:.*]] = affine.load
// CHECK:           %[[RES:.*]] = affine.for %{{.*}} = 0 to 100 iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK:             arith.addi %[[ACC]]
// CHECK:             affine.yield
// CHECK:           affine.store %[[RES]]
func.func @nested_inner_promoted(%arg0: index) {
  %alloc = memref.alloc() : memref<200x200xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      %v = affine.load %alloc[%i, %arg0] : memref<200x200xi32>
      %new = arith.addi %v, %c1 : i32
      affine.store %new, %alloc[%i, %arg0] : memref<200x200xi32>
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Store-to-load forwarding (preceding store provides init value)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @store_to_load_forward
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// CHECK:         affine.store %[[C0]]
// CHECK:         %[[RESULT:.*]] = affine.for %{{.*}} = 0 to 100 iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32)
// CHECK-NOT:       affine.load
// CHECK:           arith.addi %[[ACC]]
// CHECK:           affine.yield
// CHECK:         affine.store %[[RESULT]]
func.func @store_to_load_forward(%arg0: index) {
  %alloc = memref.alloc() : memref<200xi32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  affine.store %c0, %alloc[%arg0] : memref<200xi32>
  affine.for %i = 0 to 100 {
    %v = affine.load %alloc[%arg0] : memref<200xi32>
    %new = arith.addi %v, %c1 : i32
    affine.store %new, %alloc[%arg0] : memref<200xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Multiple independent accumulators
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @multiple_accumulators
// CHECK-DAG:     affine.load %{{.*}}[%{{.*}}]
// CHECK-DAG:     affine.load %{{.*}}[%{{.*}}]
// CHECK:         affine.for {{.*}} iter_args
// CHECK-NOT:       affine.store
// CHECK:           affine.yield
// CHECK-DAG:     affine.store
// CHECK-DAG:     affine.store
func.func @multiple_accumulators(%arg0: index, %arg1: index) {
  %alloc1 = memref.alloc() : memref<200xi32>
  %alloc2 = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  affine.for %i = 0 to 100 {
    %v1 = affine.load %alloc1[%arg0] : memref<200xi32>
    %new1 = arith.addi %v1, %c1 : i32
    affine.store %new1, %alloc1[%arg0] : memref<200xi32>
    %v2 = affine.load %alloc2[%arg1] : memref<200xi32>
    %new2 = arith.addi %v2, %c2 : i32
    affine.store %new2, %alloc2[%arg1] : memref<200xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Negative — loop-variant address (skip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_variant_address
// CHECK:         affine.for
// CHECK:           affine.load
// CHECK:           affine.store
func.func @negative_variant_address() {
  %alloc = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 100 {
    %v = affine.load %alloc[%i] : memref<200xi32>
    %new = arith.addi %v, %c1 : i32
    affine.store %new, %alloc[%i] : memref<200xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 7: Negative — store value doesn't depend on load (skip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_no_dependence
// CHECK:         affine.for
// CHECK:           affine.load
// CHECK:           affine.store
func.func @negative_no_dependence(%arg0: index) {
  %alloc = memref.alloc() : memref<200xi32>
  %c42 = arith.constant 42 : i32
  affine.for %i = 0 to 100 {
    %v = affine.load %alloc[%arg0] : memref<200xi32>
    affine.store %c42, %alloc[%arg0] : memref<200xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 8: Negative — aliasing access (k ∈ [0, 200) could equal arg1)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_aliasing
// CHECK:         affine.for
// CHECK:           affine.load
// CHECK:           affine.load
// CHECK:           affine.store
func.func @negative_aliasing(%arg0: index, %arg1: index) {
  %alloc = memref.alloc() : memref<200x200xi32>
  affine.for %k = 0 to 200 {
    %acc = affine.load %alloc[%arg0, %arg1] : memref<200x200xi32>
    %v = affine.load %alloc[%arg0, %k] : memref<200x200xi32>
    %sum = arith.addi %acc, %v : i32
    affine.store %sum, %alloc[%arg0, %arg1] : memref<200x200xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 9: Negative — multiple stores to same address (skip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_multiple_stores
// CHECK:         affine.for
// CHECK:           affine.load
// CHECK:           affine.store
// CHECK:           affine.store
func.func @negative_multiple_stores(%arg0: index) {
  %alloc = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  affine.for %i = 0 to 100 {
    %v = affine.load %alloc[%arg0] : memref<200xi32>
    %new = arith.addi %v, %c1 : i32
    affine.store %new, %alloc[%arg0] : memref<200xi32>
    affine.store %c2, %alloc[%arg0] : memref<200xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 10: Negative — non-affine memory op (memref.load) causes bail
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_non_affine
// CHECK:         affine.for
// CHECK:           affine.load
// CHECK:           memref.load
// CHECK:           affine.store
func.func @negative_non_affine(%arg0: index, %m: memref<200xi32>) {
  %alloc = memref.alloc() : memref<200xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 100 {
    %v = affine.load %alloc[%arg0] : memref<200xi32>
    %x = memref.load %m[%arg0] : memref<200xi32>
    %sum = arith.addi %v, %x : i32
    affine.store %sum, %alloc[%arg0] : memref<200xi32>
  }
  return
}
