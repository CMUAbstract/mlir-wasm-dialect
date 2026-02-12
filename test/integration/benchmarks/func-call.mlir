// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32 42 --quiet

func.func @add3(%x: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %y = arith.addi %x, %c3 : i32
  return %y : i32
}

func.func @main() -> i32 attributes { exported } {
  %c39 = arith.constant 39 : i32
  %result = func.call @add3(%c39) : (i32) -> i32
  return %result : i32
}
