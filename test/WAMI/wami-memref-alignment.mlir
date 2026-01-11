// RUN: wasm-opt %s --wami-convert-memref | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Global Memory Alignment
//===----------------------------------------------------------------------===//
//
// This test verifies that global memrefs are placed at properly aligned
// addresses. When a global has alignment requirements, its base address
// should be aligned accordingly, not just placed immediately after the
// previous global.
//
// Layout:
//   @small_array: 3 bytes at 1024 (no alignment requirement)
//   @aligned_f64: 8 bytes, needs 8-byte alignment
//                 Should be at 1032 (next 8-byte aligned address after 1027)
//                 NOT at 1027 (1024 + 3)

// CHECK-DAG: wami.data @small_array_data = dense<[1, 2, 3]> : tensor<3xi8> at 1024
memref.global @small_array : memref<3xi8> = dense<[1, 2, 3]>

// With 8-byte alignment, this should be at 1032, not 1027
// 1027 rounded up to next multiple of 8 = 1032
// CHECK-DAG: wami.data @aligned_f64_data = dense<1.000000e+00> : tensor<1xf64> at 1032
memref.global @aligned_f64 : memref<1xf64> = dense<[1.0]> {alignment = 8}
