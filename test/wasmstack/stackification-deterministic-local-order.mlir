// RUN: wasm-opt %s --convert-to-wasmstack 2>&1 | FileCheck %s

// Regression test: local allocation should be deterministic and follow
// stackification discovery order.

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: wasmstack.func @deterministic_local_order
// CHECK: wasmstack.local 0 : i32
// CHECK-NEXT: wasmstack.local 1 : i32
// CHECK-NEXT: wasmstack.local 2 : i32
// CHECK-NEXT: wasmstack.local 3 : i32
// CHECK: wasmstack.local.set 3 : i32
// CHECK: wasmstack.local.set 2 : i32
// CHECK: wasmstack.local.set 0 : i32
// CHECK: wasmstack.local.set 1 : i32

module {
  wasmssa.func @deterministic_local_order() -> i32 {
    %a = wasmssa.const 1 : i32
    %b = wasmssa.const 2 : i32
    %c = wasmssa.add %a %b : i32
    %d = wasmssa.sub %a %b : i32
    %e = wasmssa.mul %c %c : i32
    %f = wasmssa.mul %d %d : i32
    %g = wasmssa.add %e %f : i32
    wasmssa.return %g : i32
  }
}
