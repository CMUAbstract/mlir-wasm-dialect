// REQUIRES: wabt
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: wasm-validate %t.wasm
// RUN: wasm-objdump -x %t.wasm | FileCheck %s --check-prefix=NONRELOC
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack | wasm-emit --mlir-to-wasm --relocatable -o %t.o
// RUN: wasm-validate %t.o
// RUN: wasm-objdump -x %t.o | FileCheck %s --check-prefix=RELOC

// NONRELOC: Type[1]:
// NONRELOC: Function[2]:
// NONRELOC: Code[2]:
// NONRELOC-NOT: name: "linking"

// RELOC: Type[1]:
// RELOC: Function[2]:
// RELOC: Code[2]:
// RELOC: name: "linking"
// RELOC: symbol table [count=2]
// RELOC: <callee>
// RELOC: <caller>
// RELOC: name: "reloc.CODE"
// RELOC: R_WASM_FUNCTION_INDEX_LEB

func.func @callee(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %y = arith.addi %x, %c1 : i32
  return %y : i32
}

func.func @caller(%x: i32) -> i32 {
  %y = func.call @callee(%x) : (i32) -> i32
  return %y : i32
}
