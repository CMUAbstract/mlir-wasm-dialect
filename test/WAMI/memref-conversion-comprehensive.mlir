// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts | FileCheck %s
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack -verify-wasmstack 2>&1 | FileCheck %s --check-prefix=VERIFY

// Comprehensive tests for memref dialect conversion through the full pipeline
// Tests memory operations, global arrays, and interactions with control flow

// Verify no unrealized conversion casts remain
// CHECK-NOT: unrealized_conversion_cast
// VERIFY-NOT: error

//===----------------------------------------------------------------------===//
// Global memory declarations
//===----------------------------------------------------------------------===//

// CHECK-DAG: wami.data @array_i32_data
memref.global @array_i32 : memref<4xi32> = dense<[1, 2, 3, 4]>

// CHECK-DAG: wami.data @array_i64_data
memref.global @array_i64 : memref<3xi64> = dense<[100, 200, 300]>

// CHECK-DAG: wami.data @array_f32_data
memref.global @array_f32 : memref<4xf32> = dense<[1.0, 2.0, 3.0, 4.0]>

// CHECK-DAG: wami.data @zeros_data
memref.global @zeros : memref<8xi32> = dense<0>

//===----------------------------------------------------------------------===//
// Basic load/store operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @basic_load_i32
func.func @basic_load_i32() -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>
  %c0 = arith.constant 0 : index
  // CHECK: wami.load
  %val = memref.load %ref[%c0] : memref<4xi32>
  return %val : i32
}

// CHECK-LABEL: wasmssa.func @basic_load_i64
func.func @basic_load_i64() -> i64 {
  %ref = memref.get_global @array_i64 : memref<3xi64>
  %c1 = arith.constant 1 : index
  // CHECK: wami.load
  %val = memref.load %ref[%c1] : memref<3xi64>
  return %val : i64
}

// CHECK-LABEL: wasmssa.func @basic_load_f32
func.func @basic_load_f32() -> f32 {
  %ref = memref.get_global @array_f32 : memref<4xf32>
  %c2 = arith.constant 2 : index
  // CHECK: wami.load
  %val = memref.load %ref[%c2] : memref<4xf32>
  return %val : f32
}

// CHECK-LABEL: wasmssa.func @basic_store_i32
func.func @basic_store_i32(%val: i32) {
  %ref = memref.get_global @zeros : memref<8xi32>
  %c0 = arith.constant 0 : index
  // CHECK: wami.store
  memref.store %val, %ref[%c0] : memref<8xi32>
  return
}

//===----------------------------------------------------------------------===//
// Variable index operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @load_variable_index
func.func @load_variable_index(%idx: index) -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>
  // CHECK: wami.load
  %val = memref.load %ref[%idx] : memref<4xi32>
  return %val : i32
}

// CHECK-LABEL: wasmssa.func @store_variable_index
func.func @store_variable_index(%idx: index, %val: i32) {
  %ref = memref.get_global @zeros : memref<8xi32>
  // CHECK: wami.store
  memref.store %val, %ref[%idx] : memref<8xi32>
  return
}

// CHECK-LABEL: wasmssa.func @computed_index
func.func @computed_index(%base: index, %offset: index) -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>
  %idx = arith.addi %base, %offset : index
  // CHECK: wami.load
  %val = memref.load %ref[%idx] : memref<4xi32>
  return %val : i32
}

//===----------------------------------------------------------------------===//
// Memory operations in loops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @sum_array
func.func @sum_array() -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init = arith.constant 0 : i32

  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (i32) {
    // CHECK: wami.load
    %elem = memref.load %ref[%i] : memref<4xi32>
    %new_acc = arith.addi %acc, %elem : i32
    scf.yield %new_acc : i32
  }
  return %sum : i32
}

// CHECK-LABEL: wasmssa.func @fill_array
func.func @fill_array(%val: i32) {
  %ref = memref.get_global @zeros : memref<8xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // CHECK: wasmssa.block
  // CHECK: wasmssa.loop
  scf.for %i = %c0 to %c8 step %c1 {
    // CHECK: wami.store
    memref.store %val, %ref[%i] : memref<8xi32>
  }
  return
}

// CHECK-LABEL: wasmssa.func @copy_array
func.func @copy_array() {
  %src = memref.get_global @array_i32 : memref<4xi32>
  %dst = memref.get_global @zeros : memref<8xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  scf.for %i = %c0 to %c4 step %c1 {
    // CHECK: wami.load
    %val = memref.load %src[%i] : memref<4xi32>
    // CHECK: wami.store
    memref.store %val, %dst[%i] : memref<8xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Memory operations in conditionals
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @conditional_load
func.func @conditional_load(%cond: i1, %idx1: index, %idx2: index) -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>

  %result = scf.if %cond -> (i32) {
    // CHECK: wami.load
    %v1 = memref.load %ref[%idx1] : memref<4xi32>
    scf.yield %v1 : i32
  } else {
    // CHECK: wami.load
    %v2 = memref.load %ref[%idx2] : memref<4xi32>
    scf.yield %v2 : i32
  }
  return %result : i32
}

// CHECK-LABEL: wasmssa.func @conditional_store
func.func @conditional_store(%cond: i1, %val1: i32, %val2: i32, %idx: index) {
  %ref = memref.get_global @zeros : memref<8xi32>

  scf.if %cond {
    // CHECK: wami.store
    memref.store %val1, %ref[%idx] : memref<8xi32>
  } else {
    // CHECK: wami.store
    memref.store %val2, %ref[%idx] : memref<8xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @running_sum
func.func @running_sum() {
  %ref = memref.get_global @zeros : memref<8xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %init = arith.constant 0 : i32

  // Store running sum at each index
  %_ = scf.for %i = %c0 to %c8 step %c1 iter_args(%sum = %init) -> (i32) {
    %i32 = arith.index_cast %i : index to i32
    %new_sum = arith.addi %sum, %i32 : i32
    memref.store %new_sum, %ref[%i] : memref<8xi32>
    scf.yield %new_sum : i32
  }
  return
}

// CHECK-LABEL: wasmssa.func @find_max
func.func @find_max() -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // Start with first element
  %first = memref.load %ref[%c0] : memref<4xi32>

  %max = scf.for %i = %c1 to %c4 step %c1 iter_args(%current_max = %first) -> (i32) {
    %elem = memref.load %ref[%i] : memref<4xi32>
    %is_greater = arith.cmpi sgt, %elem, %current_max : i32
    %new_max = arith.select %is_greater, %elem, %current_max : i32
    scf.yield %new_max : i32
  }
  return %max : i32
}

// CHECK-LABEL: wasmssa.func @dot_product
func.func @dot_product() -> i32 {
  %arr1 = memref.get_global @array_i32 : memref<4xi32>
  %arr2 = memref.get_global @array_i32 : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init = arith.constant 0 : i32

  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (i32) {
    %v1 = memref.load %arr1[%i] : memref<4xi32>
    %v2 = memref.load %arr2[%i] : memref<4xi32>
    %prod = arith.muli %v1, %v2 : i32
    %new_acc = arith.addi %acc, %prod : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Memory with floating point
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @sum_float_array
func.func @sum_float_array() -> f32 {
  %ref = memref.get_global @array_f32 : memref<4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init = arith.constant 0.0 : f32

  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (f32) {
    %elem = memref.load %ref[%i] : memref<4xf32>
    %new_acc = arith.addf %acc, %elem : f32
    scf.yield %new_acc : f32
  }
  return %sum : f32
}

// CHECK-LABEL: wasmssa.func @average_float
func.func @average_float() -> f32 {
  %ref = memref.get_global @array_f32 : memref<4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_f32 = arith.constant 4.0 : f32
  %init = arith.constant 0.0 : f32

  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (f32) {
    %elem = memref.load %ref[%i] : memref<4xf32>
    %new_acc = arith.addf %acc, %elem : f32
    scf.yield %new_acc : f32
  }

  %avg = arith.divf %sum, %c4_f32 : f32
  return %avg : f32
}

//===----------------------------------------------------------------------===//
// Nested loops with memory
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @nested_sum
func.func @nested_sum(%n: index, %m: index) -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %c4 = arith.constant 4 : index

  // Nested loop summing same array multiple times
  %outer_result = scf.for %i = %c0 to %n step %c1 iter_args(%outer_acc = %init) -> (i32) {
    %inner_result = scf.for %j = %c0 to %c4 step %c1 iter_args(%inner_acc = %outer_acc) -> (i32) {
      %elem = memref.load %ref[%j] : memref<4xi32>
      %new_inner = arith.addi %inner_acc, %elem : i32
      scf.yield %new_inner : i32
    }
    scf.yield %inner_result : i32
  }
  return %outer_result : i32
}

//===----------------------------------------------------------------------===//
// Load-modify-store patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @increment_element
func.func @increment_element(%idx: index) {
  %ref = memref.get_global @zeros : memref<8xi32>
  %c1 = arith.constant 1 : i32

  // Load, increment, store back
  %old = memref.load %ref[%idx] : memref<8xi32>
  %new = arith.addi %old, %c1 : i32
  memref.store %new, %ref[%idx] : memref<8xi32>
  return
}

// CHECK-LABEL: wasmssa.func @scale_element
func.func @scale_element(%idx: index, %factor: i32) {
  %ref = memref.get_global @zeros : memref<8xi32>

  %old = memref.load %ref[%idx] : memref<8xi32>
  %new = arith.muli %old, %factor : i32
  memref.store %new, %ref[%idx] : memref<8xi32>
  return
}

//===----------------------------------------------------------------------===//
// Multiple arrays
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @add_arrays
func.func @add_arrays(%idx: index) -> i32 {
  %arr1 = memref.get_global @array_i32 : memref<4xi32>
  %arr2 = memref.get_global @zeros : memref<8xi32>

  %v1 = memref.load %arr1[%idx] : memref<4xi32>
  %v2 = memref.load %arr2[%idx] : memref<8xi32>
  %sum = arith.addi %v1, %v2 : i32
  return %sum : i32
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @load_first_element
func.func @load_first_element() -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>
  %c0 = arith.constant 0 : index
  %val = memref.load %ref[%c0] : memref<4xi32>
  return %val : i32
}

// CHECK-LABEL: wasmssa.func @load_last_element
func.func @load_last_element() -> i32 {
  %ref = memref.get_global @array_i32 : memref<4xi32>
  %c3 = arith.constant 3 : index
  %val = memref.load %ref[%c3] : memref<4xi32>
  return %val : i32
}

// CHECK-LABEL: wasmssa.func @same_index_load_store
func.func @same_index_load_store(%idx: index) -> i32 {
  %ref = memref.get_global @zeros : memref<8xi32>

  // Load, compute, store to same location, return original
  %original = memref.load %ref[%idx] : memref<8xi32>
  %c1 = arith.constant 1 : i32
  %new = arith.addi %original, %c1 : i32
  memref.store %new, %ref[%idx] : memref<8xi32>
  return %original : i32
}
