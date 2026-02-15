// RUN: not wasm-emit %s --mlir-to-wasm -o %t.wasm 2>&1 | FileCheck %s

// CHECK: error: 'wasmstack.br' op branch target 'missing' not found in enclosing block/loop/if

builtin.module {
  wasmstack.module @mod {
    wasmstack.func @main : () -> () {
      wasmstack.br @missing
      wasmstack.return
    }
  }
}
