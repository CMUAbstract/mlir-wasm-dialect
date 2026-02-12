// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32 12 --quiet

func.func @main() -> i32 attributes { exported } {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c4_idx = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32

  %buf = memref.alloc() : memref<4xi32>

  scf.for %i = %c0_idx to %c4_idx step %c1_idx {
    %iv_i32 = arith.index_cast %i : index to i32
    %value = arith.muli %iv_i32, %c2_i32 : i32
    memref.store %value, %buf[%i] : memref<4xi32>
  }

  %sum = scf.for %j = %c0_idx to %c4_idx step %c1_idx iter_args(%acc = %c0_i32) -> (i32) {
    %loaded = memref.load %buf[%j] : memref<4xi32>
    %next = arith.addi %acc, %loaded : i32
    scf.yield %next : i32
  }

  memref.dealloc %buf : memref<4xi32>
  return %sum : i32
}
