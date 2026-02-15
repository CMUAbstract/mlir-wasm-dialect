// REQUIRES: wizard_exec, llvm_wasm_backend
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wami.wasm
// RUN: %run_wizard_bin --input %t.wami.wasm --expect-i32 42 --quiet
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --coro-to-llvm -o %t.coro.mlir
// RUN: mlir-opt %t.coro.mlir --lower-affine --convert-scf-to-cf --convert-arith-to-llvm="index-bitwidth=32" --convert-func-to-llvm="index-bitwidth=32" --memref-expand --expand-strided-metadata --finalize-memref-to-llvm="index-bitwidth=32" --convert-cf-to-llvm="index-bitwidth=32" --convert-to-llvm --reconcile-unrealized-casts -o %t.llvm.mlir
// RUN: mlir-translate %t.llvm.mlir --mlir-to-llvmir -o %t.llvm.ll
// RUN: llc -filetype=obj -mtriple=wasm32-wasi %t.llvm.ll -o %t.llvm.o
// RUN: wasm-ld --no-entry --allow-undefined --export=main --export-memory -o %t.llvm.wasm %t.llvm.o
// RUN: %run_wizard_bin --input %t.llvm.wasm --expect-i32 42 --quiet
//
// Trace output is printed without separators by Wizard.
// In order, the observed values are:
//   100, 41, 200, 41, 201, 42, 101, 42, 42
// Where:
//   100: main before spawn/resume
//   41:  main input argument
//   200: coro.impl entry marker
//   41:  coro.impl observed %x
//   201: coro.impl after compute marker
//   42:  coro.impl computed result
//   101: main after resume marker
//   42:  main observed resumed value
//   42:  final @main result printed by run_wizard_bin (--print-result)
//
// To view trace output (without --quiet):
//   RUN_WIZARD_STACK_SWITCHING=1 WIZARD_ENGINE_DIR=/Users/byeongjee/wasm/wizard-engine \
//   wasm-opt test/integration/stack-switching/coro-oneshot-trace-runtime.mlir \
//     --coro-verify-intrinsics --coro-normalize --wami-convert-all \
//     --reconcile-unrealized-casts --coro-to-wami --convert-to-wasmstack \
//     --verify-wasmstack | wasm-emit --mlir-to-wasm -o /tmp/coro-oneshot-trace.wasm
//   RUN_WIZARD_STACK_SWITCHING=1 WIZARD_ENGINE_DIR=/Users/byeongjee/wasm/wizard-engine \
//   python3 test/integration/stack-switching/run_wizard_bin.py \
//     --input /tmp/coro-oneshot-trace.wasm --expect-i32 42

module {
  func.func private @print_i32(i32) attributes {
    wasm.import_module = "wizeng",
    wasm.import_name = "puti"
  }

  func.func private @coro.spawn.oneshot() -> i64
  func.func private @coro.resume.oneshot(%h: i64, %x: i32)
      -> (i64, i1, i32)

  func.func @coro.impl.oneshot(%x: i32) -> i32 {
    %m200 = arith.constant 200 : i32
    func.call @print_i32(%m200) : (i32) -> ()
    func.call @print_i32(%x) : (i32) -> ()

    %c1 = arith.constant 1 : i32
    %r = arith.addi %x, %c1 : i32

    %m201 = arith.constant 201 : i32
    func.call @print_i32(%m201) : (i32) -> ()
    func.call @print_i32(%r) : (i32) -> ()
    return %r : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %m100 = arith.constant 100 : i32
    func.call @print_i32(%m100) : (i32) -> ()

    %h0 = func.call @coro.spawn.oneshot() : () -> i64
    %x = arith.constant 41 : i32
    func.call @print_i32(%x) : (i32) -> ()

    %h1, %done, %r = func.call @coro.resume.oneshot(%h0, %x)
        : (i64, i32) -> (i64, i1, i32)

    %m101 = arith.constant 101 : i32
    func.call @print_i32(%m101) : (i32) -> ()
    func.call @print_i32(%r) : (i32) -> ()
    return %r : i32
  }
}
