// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32 55 --quiet

func.func @main() -> i32 attributes { exported } {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c11_idx = arith.constant 11 : index
  %c0_i32 = arith.constant 0 : i32

  %sum = scf.for %i = %c1_idx to %c11_idx step %c1_idx iter_args(%acc = %c0_i32) -> (i32) {
    %iv_i32 = arith.index_cast %i : index to i32
    %next = arith.addi %acc, %iv_i32 : i32
    scf.yield %next : i32
  }

  return %sum : i32
}
