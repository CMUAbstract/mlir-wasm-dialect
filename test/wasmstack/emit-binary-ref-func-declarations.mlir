// RUN: wasm-opt %s --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: od -An -tx1 -v %t.wasm | tr -d ' \n' | FileCheck %s --check-prefix=HEX

// HEX: 0061736d01000000
// Element section: id=0x09, size=0x05, payload=[count=1, flags=3, elemkind=0, vec_len=1, funcidx=0]
// HEX: 09050103000100
// ref.func opcode with function index 0 appears in code.
// HEX: d200

wasmstack.module @ref_func_decl {
  wasmstack.func @f0 : () -> () {
    wasmstack.return
  }

  wasmstack.func @f1 : () -> () {
    wasmstack.ref.func @f0
    wasmstack.drop : !wasmstack.funcref<@f0>
    wasmstack.return
  }
}
