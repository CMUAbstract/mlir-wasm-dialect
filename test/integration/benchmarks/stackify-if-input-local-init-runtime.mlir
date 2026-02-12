// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32=14 --quiet

// Runtime regression guard for if-input local initialization.
// The branch input is reused non-linearly (%a + %a), so stackification must
// initialize the local at branch entry before local.get-based reuse.

module {
  wasmssa.func exported @main() -> i32 {
    %cond = wasmssa.const 1 : i32
    %x = wasmssa.const 7 : i32

    wasmssa.if %cond(%x) : i32 : {
    ^bb0(%a: i32):
      %sum = wasmssa.add %a %a : i32
      wasmssa.block_return %sum : i32
    } "else" {
    ^bb1(%b: i32):
      %zero = wasmssa.const 0 : i32
      wasmssa.block_return %zero : i32
    }> ^bb2

  ^bb2(%r: i32):
    wasmssa.return %r : i32
  }
}
