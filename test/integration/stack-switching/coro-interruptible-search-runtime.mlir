// REQUIRES: wizard_exec, llvm_wasm_backend
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wami.wasm
// RUN: %run_wizard_bin --input %t.wami.wasm --expect-i32 42 --quiet
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize -o %t.norm.mlir
// RUN: mlir-opt %t.norm.mlir --lower-affine --convert-scf-to-cf --convert-arith-to-llvm="index-bitwidth=32" --convert-func-to-llvm="index-bitwidth=32" --memref-expand --expand-strided-metadata --finalize-memref-to-llvm="index-bitwidth=32" --convert-cf-to-llvm="index-bitwidth=32" --convert-to-llvm --reconcile-unrealized-casts -o %t.prellvm.mlir
// RUN: wasm-opt %t.prellvm.mlir --coro-to-llvm -o %t.coro.llvm.mlir
// RUN: mlir-translate %t.coro.llvm.mlir --mlir-to-llvmir -o %t.llvm.ll
// RUN: opt -passes='coro-early,coro-split,coro-elide,coro-cleanup' %t.llvm.ll -o %t.llvm.opt.ll
// RUN: llc -filetype=obj -mtriple=wasm32-wasi %t.llvm.opt.ll -o %t.llvm.o
// RUN: wasm-ld --no-entry --allow-undefined --export=main --export-memory -o %t.llvm.wasm %t.llvm.o
// RUN: %run_wizard_bin --input %t.llvm.wasm --expect-i32 42 --quiet

module {
  func.func private @coro.spawn.search() -> i64
  func.func private @coro.resume.search(%h: i64, %candidate: i32)
      -> (i64, i1, i32)

  func.func @coro.impl.search(%candidate: i32) -> i32 {
    %threshold = arith.constant 42 : i32
    %zero = arith.constant 0 : i32
    %is_valid = arith.cmpi eq, %candidate, %threshold : i32
    %r = scf.if %is_valid -> i32 {
      scf.yield %candidate : i32
    } else {
      scf.yield %zero : i32
    }
    return %r : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %h0 = func.call @coro.spawn.search() : () -> i64
    %p0 = arith.constant 10 : i32
    %p1 = arith.constant 42 : i32

    %h1, %d1, %r1 = func.call @coro.resume.search(%h0, %p1)
        : (i64, i32) -> (i64, i1, i32)

    return %r1 : i32
  }
}
