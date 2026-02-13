// RUN: not wasm-opt %s --wami-convert-all --reconcile-unrealized-casts -o /dev/null 2>&1 | FileCheck %s

// Minimal reproducer for the floyd-warshall legalization failure:
// an scf.if that returns i1 currently reaches wasmssa.block_return as i1.
module {
  func.func @main() -> i32 {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32

    %flag = scf.if %true -> (i1) {
      scf.yield %true : i1
    } else {
      scf.yield %false : i1
    }

    %r = arith.select %flag, %c0, %c0 : i32
    return %r : i32
  }
}

// CHECK: error: 'wasmssa.block_return' op operand #0
// CHECK-SAME: got 'i1'
