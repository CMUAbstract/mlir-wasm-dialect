// RUN: wasm-opt %s -verify-diagnostics

wasmstack.module {
  // expected-error @+1 {{unknown wasmstack.type.func symbol @missing_ft}}
  wasmstack.type.cont @ct = cont @missing_ft

  wasmstack.func @f : () -> () {
    wasmstack.return
  }
}
