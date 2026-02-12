// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack 2>&1 | FileCheck %s

// Regression test: convert-to-wasmstack should fully lower module-level
// declarations, not leave wami/wasmssa top-level ops behind.

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: module {
// CHECK-NEXT:   wasmstack.module {
// CHECK: wasmstack.memory @__linear_memory min = 1 export "memory"
// CHECK: wasmstack.global @g_base : i32 mutable
// CHECK: wasmstack.i32.const 1024
// CHECK: wasmstack.data @__linear_memory offset = 1024
// CHECK-NOT: wami.
// CHECK-NOT: wasmssa.

module {
  memref.global @g : memref<1xi32> = dense<[42]>

  func.func @main() -> i32 {
    %m = memref.get_global @g : memref<1xi32>
    %c0 = arith.constant 0 : index
    %v = memref.load %m[%c0] : memref<1xi32>
    return %v : i32
  }
}
