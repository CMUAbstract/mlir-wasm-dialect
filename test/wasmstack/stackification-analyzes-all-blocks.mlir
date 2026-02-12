// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// Regression test: stackification analysis must include non-entry blocks.
// If analysis only visits the function entry block, `%sum` in `^next` is not
// planned for local/tee, and emitter fails when materializing `%sum, %sum`.

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: wasmstack.func @analyze_all_blocks
// CHECK: wasmstack.local.tee
// CHECK: wasmstack.mul : i32
// CHECK: wasmstack.return

module {
  wasmssa.func @analyze_all_blocks() -> i32 {
    %c1 = wasmssa.const 1 : i32
    wasmssa.block(%c1) : i32 : {
    ^body(%seed: i32):
      wasmssa.block_return %seed : i32
    }> ^next

  ^next(%v: i32):
    %sum = wasmssa.add %v %c1 : i32
    %doubled = wasmssa.mul %sum %sum : i32
    wasmssa.return %doubled : i32
  }
}
