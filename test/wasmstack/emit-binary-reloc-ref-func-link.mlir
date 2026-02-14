// REQUIRES: wasm_ld
// RUN: wasm-opt %s --verify-wasmstack | wasm-emit --mlir-to-wasm --relocatable -o %t.o
// RUN: wasm-ld --no-entry --allow-undefined %t.o -o %t.linked.wasm
// RUN: od -An -tx1 -N4 -v %t.linked.wasm | FileCheck %s --check-prefix=MAGIC

// MAGIC: 00 61 73 6d

wasmstack.module @reloc_ref_func_link {
  wasmstack.func @f0 : () -> () {
    wasmstack.return
  }

  wasmstack.func @f1 : () -> () {
    wasmstack.ref.func @f0
    wasmstack.drop : !wasmstack.funcref<@f0>
    wasmstack.return
  }
}
