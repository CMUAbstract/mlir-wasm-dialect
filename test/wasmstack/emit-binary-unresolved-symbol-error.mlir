// RUN: not wasm-emit %s --mlir-to-wasm -o %t.wasm 2>&1 | FileCheck %s

// CHECK: error: 'wasmstack.call' op unresolved function symbol 'missing' (no wasmstack.func or wasmstack.import_func)

builtin.module {
  wasmstack.module @mod {
    wasmstack.func @main : () -> i32 {
      wasmstack.i32.const 0
      wasmstack.call @missing : (i32) -> i32
      wasmstack.return
    }
  }
}
