// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: test -s %t.wasm
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack | wasm-emit --mlir-to-wasm --relocatable -o %t.o
// RUN: test -s %t.o

// This test validates that wasm-emit accepts the top-level WasmStack form
// produced by convert-to-wasmstack (without requiring a wasmstack.module
// wrapper).

func.func @add(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  return %sum : i32
}
