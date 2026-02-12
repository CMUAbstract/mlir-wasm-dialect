// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32=-13 --quiet

// Runtime regression guard for multi-result call ordering:
// swap(17, 4) returns (4, 17), so main must compute 4 - 17 = -13.

func.func @swap(%a: i32, %b: i32) -> (i32, i32) {
  return %b, %a : i32, i32
}

func.func @main() -> i32 attributes { exported } {
  %c17 = arith.constant 17 : i32
  %c4 = arith.constant 4 : i32
  %x, %y = func.call @swap(%c17, %c4) : (i32, i32) -> (i32, i32)
  %r = arith.subi %x, %y : i32
  return %r : i32
}
