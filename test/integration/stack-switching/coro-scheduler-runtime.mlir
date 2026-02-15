// REQUIRES: wizard_exec, llvm_wasm_backend
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wami.wasm
// RUN: %run_wizard_bin --input %t.wami.wasm --expect-i32 42 --quiet
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --coro-to-llvm -o %t.coro.mlir
// RUN: mlir-opt %t.coro.mlir --lower-affine --convert-scf-to-cf --convert-arith-to-llvm="index-bitwidth=32" --convert-func-to-llvm="index-bitwidth=32" --memref-expand --expand-strided-metadata --finalize-memref-to-llvm="index-bitwidth=32" --convert-cf-to-llvm="index-bitwidth=32" --convert-to-llvm --reconcile-unrealized-casts -o %t.llvm.mlir
// RUN: mlir-translate %t.llvm.mlir --mlir-to-llvmir -o %t.llvm.ll
// RUN: llc -filetype=obj -mtriple=wasm32-wasi %t.llvm.ll -o %t.llvm.o
// RUN: wasm-ld --no-entry --allow-undefined --export=main --export-memory -o %t.llvm.wasm %t.llvm.o
// RUN: %run_wizard_bin --input %t.llvm.wasm --expect-i32 42 --quiet

module {
  func.func private @coro.spawn.task_a() -> i64
  func.func private @coro.resume.task_a(%h: i64, %x: i32)
      -> (i64, i1, i32)
  func.func private @coro.spawn.task_b() -> i64
  func.func private @coro.resume.task_b(%h: i64, %x: i32)
      -> (i64, i1, i32)

  func.func @coro.impl.task_a(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %r = arith.muli %x, %c2 : i32
    return %r : i32
  }

  func.func @coro.impl.task_b(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %r = arith.muli %x, %c2 : i32
    return %r : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %a = arith.constant 10 : i32
    %b = arith.constant 11 : i32

    %h10 = func.call @coro.spawn.task_a() : () -> i64
    %h20 = func.call @coro.spawn.task_b() : () -> i64

    %h11, %d1, %r1 = func.call @coro.resume.task_a(%h10, %a)
        : (i64, i32) -> (i64, i1, i32)
    %h21, %d2, %r2 = func.call @coro.resume.task_b(%h20, %b)
        : (i64, i32) -> (i64, i1, i32)

    %sum = arith.addi %r1, %r2 : i32
    return %sum : i32
  }
}
