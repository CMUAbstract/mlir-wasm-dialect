// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32 32 --quiet

func.func @main() -> i32 attributes { exported } {
  %c7 = arith.constant 7 : i32
  %c5 = arith.constant 5 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32

  %sum = arith.addi %c7, %c5 : i32
  %prod = arith.muli %sum, %c3 : i32
  %result = arith.subi %prod, %c4 : i32
  return %result : i32
}
