// REQUIRES: wabt
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack | wasm-emit --mlir-to-wasm --relocatable -o %t.o
// RUN: wasm-validate %t.o
// RUN: wasm-objdump -x %t.o | FileCheck %s

// CHECK: Import[1]:
// CHECK: <ext_add1>
// CHECK: <- env.ext_add1
// CHECK: Function[1]:
// CHECK: <caller>
// CHECK: symbol table [count=2]
// CHECK: <ext_add1> func=0 [ undefined
// CHECK: <caller> func=1

func.func private @ext_add1(%x: i32) -> i32

func.func @caller(%x: i32) -> i32 {
  %y = func.call @ext_add1(%x) : (i32) -> i32
  return %y : i32
}
