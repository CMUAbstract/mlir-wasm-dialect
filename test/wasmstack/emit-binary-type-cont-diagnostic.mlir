// RUN: not wasm-emit %s --mlir-to-wasm -o %t.wasm 2>&1 | FileCheck %s

// This should be a clean diagnostic, not an assertion crash in wasm-emit.
// CHECK: unknown wasmstack.type.func symbol @missing_ft

wasmstack.module {
  wasmstack.type.cont @ct = cont @missing_ft

  wasmstack.func @f : () -> () {
    wasmstack.return
  }
}
