// RUN: wasm-opt %s --convert-to-wasmstack 2>&1 | FileCheck %s

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: module {
// CHECK:   wasmstack.module
// CHECK:     wasmstack.func @add
// CHECK:       wasmstack.i32.const 7
// CHECK:       wasmstack.i32.const 11
// CHECK:       wasmstack.add : i32
// CHECK:       wasmstack.return

module {
  wasmssa.func @add() -> i32 {
    %a = wasmssa.const 7 : i32
    %b = wasmssa.const 11 : i32
    %s = wasmssa.add %a %b : i32
    wasmssa.return %s : i32
  }
}
