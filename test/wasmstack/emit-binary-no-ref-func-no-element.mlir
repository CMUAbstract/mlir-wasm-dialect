// REQUIRES: wabt
// RUN: wasm-opt %s --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: wasm-objdump -x %t.wasm | FileCheck %s --check-prefix=NOELEM

// NOELEM-NOT: Elem[

wasmstack.module @no_ref_func_decl {
  wasmstack.func @f0 : () -> () {
    wasmstack.return
  }

  wasmstack.func @f1 : () -> () {
    wasmstack.nop
    wasmstack.return
  }
}
