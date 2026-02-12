// REQUIRES: wasmtime_exec
// RUN: wasm-opt %s --wami-convert-all --reconcile-unrealized-casts --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wasm_bin --input %t.wasm --expect-i32=-2 --quiet

// Runtime regression guard for module-level global matrix lowering:
// det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2.

memref.global @gmat : memref<2x2xi32> = dense<[[1, 2], [3, 4]]>

func.func @main() -> i32 attributes { exported } {
  %mat = memref.get_global @gmat : memref<2x2xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %a00 = memref.load %mat[%c0, %c0] : memref<2x2xi32>
  %a01 = memref.load %mat[%c0, %c1] : memref<2x2xi32>
  %a10 = memref.load %mat[%c1, %c0] : memref<2x2xi32>
  %a11 = memref.load %mat[%c1, %c1] : memref<2x2xi32>

  %p0 = arith.muli %a00, %a11 : i32
  %p1 = arith.muli %a01, %a10 : i32
  %det = arith.subi %p0, %p1 : i32
  return %det : i32
}
