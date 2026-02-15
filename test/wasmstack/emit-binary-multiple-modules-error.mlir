// RUN: not wasm-emit %s --mlir-to-wasm -o %t.wasm 2>&1 | FileCheck %s

// CHECK: error: expected one top-level wasmstack.module, found 2

builtin.module {
  wasmstack.module @a {
    wasmstack.func @f : () -> () {
      wasmstack.return
    }
  }

  wasmstack.module @b {
    wasmstack.func @g : () -> () {
      wasmstack.return
    }
  }
}
