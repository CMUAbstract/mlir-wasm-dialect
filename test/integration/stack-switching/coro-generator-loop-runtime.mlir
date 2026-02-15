// REQUIRES: wizard_exec
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wizard_bin --input %t.wasm --expect-i32 0 --quiet
//
// Trace intent (conceptual): 0 1 0 2 0 3 0 4 0
// - main prints 0 before each resume
// - coroutine prints current value (1,2,3 on yield; 4 on final return)
// - runner prints final main result (0)

module {
  func.func private @print_i32(i32) attributes {
    wasm.import_module = "wizeng",
    wasm.import_name = "puti"
  }

  func.func private @coro.spawn.generator() -> i64
  func.func private @coro.resume.generator(%h: i64, %x: i32)
      -> (i64, i1, i32)
  func.func private @coro.yield.generator(%v: i32) -> i32

  func.func @coro.impl.generator(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1_step = arith.constant 1 : index
    %c3 = arith.constant 3 : index

    %final = scf.for %i = %c0 to %c3 step %c1_step iter_args(%cur = %x) -> (i32) {
      func.call @print_i32(%cur) : (i32) -> ()
      %resume_in = func.call @coro.yield.generator(%cur) : (i32) -> i32
      scf.yield %resume_in : i32
    }

    func.call @print_i32(%final) : (i32) -> ()
    return %final : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %h0 = func.call @coro.spawn.generator() : () -> i64

    %z = arith.constant 0 : i32
    %a = arith.constant 1 : i32
    %b = arith.constant 2 : i32
    %c = arith.constant 3 : i32
    %d = arith.constant 4 : i32

    func.call @print_i32(%z) : (i32) -> ()
    %h1, %done1, %r1 = func.call @coro.resume.generator(%h0, %a)
        : (i64, i32) -> (i64, i1, i32)
    func.call @print_i32(%z) : (i32) -> ()
    %h2, %done2, %r2 = func.call @coro.resume.generator(%h1, %b)
        : (i64, i32) -> (i64, i1, i32)
    func.call @print_i32(%z) : (i32) -> ()
    %h3, %done3, %r3 = func.call @coro.resume.generator(%h2, %c)
        : (i64, i32) -> (i64, i1, i32)
    func.call @print_i32(%z) : (i32) -> ()
    %h4, %done4, %r4 = func.call @coro.resume.generator(%h3, %d)
        : (i64, i32) -> (i64, i1, i32)

    %d1 = arith.extui %done1 : i1 to i32
    %d2 = arith.extui %done2 : i1 to i32
    %d3 = arith.extui %done3 : i1 to i32
    %d4 = arith.extui %done4 : i1 to i32

    %s12 = arith.addi %r1, %r2 : i32
    %s34 = arith.addi %r3, %r4 : i32
    %sr = arith.addi %s12, %s34 : i32

    %sd12 = arith.addi %d1, %d2 : i32
    %sd34 = arith.addi %d3, %d4 : i32
    %sd = arith.addi %sd12, %sd34 : i32
    %sink = arith.addi %sr, %sd : i32
    %ret0 = arith.subi %sink, %sink : i32
    return %ret0 : i32
  }
}
