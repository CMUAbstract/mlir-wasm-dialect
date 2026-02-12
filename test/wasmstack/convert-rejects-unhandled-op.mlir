// RUN: wasm-opt %s --convert-to-wasmstack -verify-diagnostics

module {
  wasmssa.func @reject_unhandled() -> i32 {
    %a = wasmssa.const 1 : i32
    %b = wasmssa.const 2 : i32
    // expected-error @+1 {{unhandled operation in stackification emitter}}
    %c = arith.addi %a, %b : i32
    wasmssa.return %c : i32
  }
}
