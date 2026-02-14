// RUN: wasm-opt %s --verify-wasmstack | wasm-emit --mlir-to-wasm --relocatable -o %t.o
// RUN: od -An -tx1 -v %t.o | tr -d ' \n' | FileCheck %s --check-prefix=HEX

// CI-safe indirect coverage for relocatable ref.func declaration encoding,
// without depending on an external linker.
// HEX: 0061736d01000000
// Table section payload for one synthetic funcref table.
// HEX: 040401700000
// Elem section payload for one declarative ref.func declaration.
// HEX: 09050103000100
// linking custom section name
// HEX: 6c696e6b696e67
// reloc.CODE custom section name
// HEX: 72656c6f632e434f4445

wasmstack.module @reloc_ref_func_sections {
  wasmstack.func @f0 : () -> () {
    wasmstack.return
  }

  wasmstack.func @f1 : () -> () {
    wasmstack.ref.func @f0
    wasmstack.drop : !wasmstack.funcref<@f0>
    wasmstack.return
  }
}
