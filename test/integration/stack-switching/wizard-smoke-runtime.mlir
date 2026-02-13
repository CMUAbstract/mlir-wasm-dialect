// REQUIRES: wizard_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wizard_bin --input %t.wasm --expect-i32 42 --quiet

module {
  func.func @main() -> i32 attributes { exported } {
    %c40 = arith.constant 40 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c40, %c2 : i32
    return %sum : i32
  }
}
