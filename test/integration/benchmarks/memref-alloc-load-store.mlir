// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32 42 --quiet

func.func @main() -> i32 attributes { exported } {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32

  %buf = memref.alloc() : memref<1xi32>
  memref.store %c42, %buf[%c0] : memref<1xi32>
  %loaded = memref.load %buf[%c0] : memref<1xi32>
  memref.dealloc %buf : memref<1xi32>

  return %loaded : i32
}
