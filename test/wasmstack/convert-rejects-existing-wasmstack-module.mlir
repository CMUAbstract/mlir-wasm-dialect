// RUN: wasm-opt %s --convert-to-wasmstack -verify-diagnostics

// expected-error @below {{convert-to-wasmstack expects input without existing wasmstack.module}}
module {
  wasmstack.module @already {
    wasmstack.func @f : () -> i32 {
      wasmstack.i32.const 0
      wasmstack.return
    }
  }
}
