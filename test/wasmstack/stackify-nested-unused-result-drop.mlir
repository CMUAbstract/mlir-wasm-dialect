// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// Regression test: nested control-flow lowering must drop unused stack
// results, otherwise stack height diverges at region exit.

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: wasmstack.func @nested_unused_result
// CHECK: wasmstack.block
// CHECK: wasmstack.div_s : i32
// CHECK-NEXT: wasmstack.drop : i32
// CHECK: wasmstack.return

module {
  wasmssa.func @nested_unused_result() -> i32 {
    %c1 = wasmssa.const 1 : i32

    wasmssa.block(%c1) : i32 : {
    ^body(%x: i32):
      %unused = wasmssa.div_si %c1 %x : i32
      wasmssa.block_return %x : i32
    }> ^exit

  ^exit(%r: i32):
    wasmssa.return %r : i32
  }
}

