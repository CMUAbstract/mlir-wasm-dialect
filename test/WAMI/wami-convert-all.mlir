// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts | FileCheck %s

// This test verifies that the unified conversion pass correctly handles
// interleaved operations from different dialects, specifically scf.for
// with memref.load inside the loop body using the loop induction variable.

// Verify no unrealized conversion casts remain after the full pipeline
// CHECK-NOT: unrealized_conversion_cast

//===----------------------------------------------------------------------===//
// Test 1: scf.for with memref.load using induction variable
//===----------------------------------------------------------------------===//

// CHECK-DAG: wami.data @array_data = dense<[1, 2, 3, 4]> : tensor<4xi32> at 1024
// CHECK-DAG: wasmssa.global @array_base i32 mutable
memref.global @array : memref<4xi32> = dense<[1, 2, 3, 4]>

// CHECK-LABEL: wasmssa.func @sum_array_loop
// CHECK-SAME: () -> i32
func.func @sum_array_loop() -> i32 {
  // CHECK: wasmssa.global_get @array_base
  %ref = memref.get_global @array : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init = arith.constant 0 : i32

  // This is the key test: loop uses induction variable %i to index memref
  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (i32) {
    // CHECK: wami.load
    %elem = memref.load %ref[%i] : memref<4xi32>
    // CHECK: wasmssa.add
    %new_acc = arith.addi %acc, %elem : i32
    scf.yield %new_acc : i32
  }

  // CHECK: wasmssa.return
  return %sum : i32
}

//===----------------------------------------------------------------------===//
// Test 2: scf.while with memref operations
//===----------------------------------------------------------------------===//

// CHECK-DAG: wami.data @data_data
// CHECK-DAG: wasmssa.global @data_base
memref.global @data : memref<8xi32> = dense<[10, 20, 30, 40, 50, 60, 70, 80]>

// CHECK-LABEL: wasmssa.func @find_first_greater_than
func.func @find_first_greater_than(%threshold: i32) -> i32 {
  %ref = memref.get_global @data : memref<8xi32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %false = arith.constant false

  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %result:2 = scf.while (%i = %c0) : (i32) -> (i32, i32) {
    %i_idx = arith.index_cast %i : i32 to index
    // CHECK: wami.load
    %val = memref.load %ref[%i_idx] : memref<8xi32>
    // CHECK: wasmssa.gt_si
    %found = arith.cmpi sgt, %val, %threshold : i32
    // CHECK: wasmssa.lt_si
    %in_bounds = arith.cmpi slt, %i, %c8 : i32
    %not_found = arith.cmpi eq, %found, %false : i1
    // CHECK: wasmssa.and
    %continue = arith.andi %in_bounds, %not_found : i1
    scf.condition(%continue) %i, %val : i32, i32
  } do {
  ^bb0(%idx: i32, %value: i32):
    // CHECK: wasmssa.add
    %next_i = arith.addi %idx, %c1 : i32
    scf.yield %next_i : i32
  }

  // CHECK: wasmssa.return
  return %result#1 : i32
}

//===----------------------------------------------------------------------===//
// Test 3: Nested scf.if inside scf.for with memref
//===----------------------------------------------------------------------===//

// CHECK-DAG: wami.data @values_data
memref.global @values : memref<4xi32> = dense<[5, -3, 7, -2]>

// CHECK-LABEL: wasmssa.func @sum_positive
func.func @sum_positive() -> i32 {
  %ref = memref.get_global @values : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32

  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (i32) {
    // CHECK: wami.load
    %val = memref.load %ref[%i] : memref<4xi32>
    // CHECK: wasmssa.gt_si
    %is_positive = arith.cmpi sgt, %val, %zero : i32
    // CHECK: wasmssa.if
    %new_acc = scf.if %is_positive -> (i32) {
      // CHECK: wasmssa.add
      %sum_val = arith.addi %acc, %val : i32
      scf.yield %sum_val : i32
    } else {
      scf.yield %acc : i32
    }
    scf.yield %new_acc : i32
  }

  // CHECK: wasmssa.return
  return %sum : i32
}

//===----------------------------------------------------------------------===//
// Test 4: Memory store inside loop
//===----------------------------------------------------------------------===//

// CHECK-DAG: wami.data @output_data
memref.global @output : memref<4xi32> = dense<[0, 0, 0, 0]>

// CHECK-LABEL: wasmssa.func @fill_with_indices
func.func @fill_with_indices() {
  %ref = memref.get_global @output : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  scf.for %i = %c0 to %c4 step %c1 {
    %i_val = arith.index_cast %i : index to i32
    // CHECK: wami.store
    memref.store %i_val, %ref[%i] : memref<4xi32>
  }

  // CHECK: wasmssa.return
  return
}

//===----------------------------------------------------------------------===//
// Test 5: math.sqrt conversion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @sqrt_value
func.func @sqrt_value(%x: f64) -> f64 {
  // CHECK: wasmssa.sqrt
  %y = math.sqrt %x : f64
  // CHECK: wasmssa.return
  return %y : f64
}
