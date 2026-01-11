// RUN: wasm-opt %s --wami-convert-memref --wami-convert-arith --wami-convert-func | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 1: Global Memory Declaration
//===----------------------------------------------------------------------===//

// CHECK-DAG: wami.data @arr_data = dense<[1, 2, 3, 4]> : tensor<4xi32> at 1024
// CHECK-DAG: wasmssa.global @arr_base i32 mutable
memref.global @arr : memref<4xi32> = dense<[1, 2, 3, 4]>

//===----------------------------------------------------------------------===//
// Test 2: Get Global and Load
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @read_global
func.func @read_global() -> i32 {
  // CHECK: wasmssa.global_get @arr_base
  %ref = memref.get_global @arr : memref<4xi32>
  %c0 = arith.constant 0 : index
  // CHECK: wami.load
  %val = memref.load %ref[%c0] : memref<4xi32>
  return %val : i32
}

//===----------------------------------------------------------------------===//
// Test 3: Store to Global Memory
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmssa.func @write_global
func.func @write_global(%val: i32) {
  %ref = memref.get_global @arr : memref<4xi32>
  %c1 = arith.constant 1 : index
  // CHECK: wami.store
  memref.store %val, %ref[%c1] : memref<4xi32>
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Multi-dimensional Array Access with constant indices
//===----------------------------------------------------------------------===//

// Second global - should get a different base address (1024 + 16 = 1040)
// CHECK-DAG: wami.data @matrix_data
// CHECK-DAG: wasmssa.global @matrix_base
memref.global @matrix : memref<2x3xi32> = dense<[[1, 2, 3], [4, 5, 6]]>

// CHECK-LABEL: wasmssa.func @matrix_const_access
func.func @matrix_const_access() -> i32 {
  %ref = memref.get_global @matrix : memref<2x3xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // Address should be: base + (1 * 3 + 2) * 4 = base + 20
  // CHECK: wasmssa.mul
  // CHECK: wasmssa.add
  // CHECK: wami.load
  %val = memref.load %ref[%c1, %c2] : memref<2x3xi32>
  return %val : i32
}

//===----------------------------------------------------------------------===//
// Test 5: Float Memory Operations
//===----------------------------------------------------------------------===//

// CHECK-DAG: wami.data @floats_data
memref.global @floats : memref<4xf32> = dense<[1.0, 2.0, 3.0, 4.0]>

// CHECK-LABEL: wasmssa.func @float_const_access
func.func @float_const_access(%val: f32) -> f32 {
  %ref = memref.get_global @floats : memref<4xf32>
  %c0 = arith.constant 0 : index
  // CHECK: wami.store
  memref.store %val, %ref[%c0] : memref<4xf32>
  // CHECK: wami.load
  %loaded = memref.load %ref[%c0] : memref<4xf32>
  return %loaded : f32
}

//===----------------------------------------------------------------------===//
// Test 6: MemRef with Non-Zero Offset
//===----------------------------------------------------------------------===//

// Test that memref with a static offset properly includes the offset in
// address computation. For strided<[1], offset: 2> with i32 elements,
// loading at index 0 should compute: base + 2*4 + 0*1*4 = base + 8
// The offset constant (8) should appear in the generated code.

// CHECK-LABEL: wasmssa.func @test_memref_with_offset
// CHECK-SAME: (%[[BASE:.*]]: !wasmssa<local ref to i32>)
func.func @test_memref_with_offset(%mem: memref<4xi32, strided<[1], offset: 2>>) -> i32 {
  %c0 = arith.constant 0 : index
  // The address computation should include offset: base + 8 (offset=2 * elemSize=4)
  // CHECK: %[[OFFSET:.*]] = wasmssa.const 8 : i32
  // CHECK: wasmssa.add %{{.*}} %[[OFFSET]] : i32
  // CHECK: wami.load
  %val = memref.load %mem[%c0] : memref<4xi32, strided<[1], offset: 2>>
  return %val : i32
}
