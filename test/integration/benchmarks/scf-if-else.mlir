// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32 17 --quiet

func.func @main() -> i32 attributes { exported } {
  %c10 = arith.constant 10 : i32
  %c3 = arith.constant 3 : i32
  %cond = arith.cmpi sgt, %c10, %c3 : i32

  %result = scf.if %cond -> (i32) {
    %c17 = arith.constant 17 : i32
    scf.yield %c17 : i32
  } else {
    %c99 = arith.constant 99 : i32
    scf.yield %c99 : i32
  }

  return %result : i32
}
