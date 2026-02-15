// RUN: not wasm-emit %s --mlir-to-wasm -o %t.wasm 2>&1 | FileCheck %s

// CHECK: error: 'wasmstack.func' op unsupported local type for binary encoding

wasmstack.module {
  wasmstack.func @f : () -> () {
    wasmstack.local 0 : !wasmstack.contref_nonnull<@missing_ct>
    wasmstack.return
  }
}
