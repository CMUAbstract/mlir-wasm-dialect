// REQUIRES: wizard_exec
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wizard_bin --input %t.wasm --expect-i32 39 --quiet
//
// Cooperative scheduler:
// - Two tasks (`task_a`, `task_b`) each yield three times and then return.
// - Main uses a loop, checks `coro.is_done.*`, and resumes the next available
//   task until both are done.
// - Result is the accumulated outputs across all resumes (= 39).
// - For trace viewing, scheduler prints marker 0 before each dispatch.

module {
  func.func private @print_i32(i32) attributes {
    wasm.import_module = "wizeng",
    wasm.import_name = "puti"
  }

  func.func private @coro.spawn.task_a() -> i64
  func.func private @coro.resume.task_a(%h: i64, %x: i32)
      -> (i64, i1, i32)
  func.func private @coro.yield.task_a(%v: i32) -> i32
  func.func private @coro.is_done.task_a(%h: i64) -> i1
  func.func private @coro.spawn.task_b() -> i64
  func.func private @coro.resume.task_b(%h: i64, %x: i32)
      -> (i64, i1, i32)
  func.func private @coro.yield.task_b(%v: i32) -> i32
  func.func private @coro.is_done.task_b(%h: i64) -> i1

  func.func @coro.impl.task_a(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1_step = arith.constant 1 : index
    %c1 = arith.constant 1 : i32

    %final = scf.for %i = %c0 to %c3 step %c1_step iter_args(%cur = %x) -> (i32) {
      %out = arith.addi %cur, %c1 : i32
      func.call @print_i32(%out) : (i32) -> ()
      %next = func.call @coro.yield.task_a(%out) : (i32) -> i32
      scf.yield %next : i32
    }
    func.call @print_i32(%final) : (i32) -> ()
    return %final : i32
  }

  func.func @coro.impl.task_b(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1_step = arith.constant 1 : index

    %final = scf.for %i = %c0 to %c3 step %c1_step iter_args(%cur = %x) -> (i32) {
      %out = arith.addi %cur, %c2 : i32
      func.call @print_i32(%out) : (i32) -> ()
      %next = func.call @coro.yield.task_b(%out) : (i32) -> i32
      scf.yield %next : i32
    }
    func.call @print_i32(%final) : (i32) -> ()
    return %final : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %false = arith.constant false
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %a0 = arith.constant 1 : i32
    %b0 = arith.constant 2 : i32

    %hA0 = func.call @coro.spawn.task_a() : () -> i64
    %hB0 = func.call @coro.spawn.task_b() : () -> i64

    %loop:8 = scf.while (%hA = %hA0, %hB = %hB0, %argA = %a0, %argB = %b0,
                          %doneA = %false, %doneB = %false, %turn = %zero,
                          %acc = %zero) :
        (i64, i64, i32, i32, i1, i1, i32, i32) ->
        (i64, i64, i32, i32, i1, i1, i32, i32) {
      %qA = func.call @coro.is_done.task_a(%hA) : (i64) -> i1
      %qB = func.call @coro.is_done.task_b(%hB) : (i64) -> i1
      %doneA_now = arith.ori %doneA, %qA : i1
      %doneB_now = arith.ori %doneB, %qB : i1
      %aActive = arith.cmpi eq, %doneA_now, %false : i1
      %bActive = arith.cmpi eq, %doneB_now, %false : i1
      %keepRunning = arith.ori %aActive, %bActive : i1
      scf.condition(%keepRunning) %hA, %hB, %argA, %argB, %doneA_now,
          %doneB_now, %turn, %acc : i64, i64, i32, i32, i1, i1, i32, i32
    } do {
    ^bb0(%hA: i64, %hB: i64, %argA: i32, %argB: i32, %doneA: i1, %doneB: i1,
         %turn: i32, %acc: i32):
      func.call @print_i32(%zero) : (i32) -> ()
      %turnIsA = arith.cmpi eq, %turn, %zero : i32
      %aActive = arith.cmpi eq, %doneA, %false : i1
      %preferA = arith.andi %turnIsA, %aActive : i1
      %runA = arith.ori %preferA, %doneB : i1
      %nextTurn = arith.xori %turn, %one : i32

      %hA_next, %hB_next, %argA_next, %argB_next, %doneA_next, %doneB_next, %acc_next = scf.if %runA -> (i64, i64, i32, i32, i1, i1, i32) {
        %hA1, %doneA_res, %outA = func.call @coro.resume.task_a(%hA, %argA)
            : (i64, i32) -> (i64, i1, i32)
        %doneA1 = arith.ori %doneA, %doneA_res : i1
        %acc1 = arith.addi %acc, %outA : i32
        scf.yield %hA1, %hB, %outA, %argB, %doneA1, %doneB, %acc1
            : i64, i64, i32, i32, i1, i1, i32
      } else {
        %hB1, %doneB_res, %outB = func.call @coro.resume.task_b(%hB, %argB)
            : (i64, i32) -> (i64, i1, i32)
        %doneB1 = arith.ori %doneB, %doneB_res : i1
        %acc1 = arith.addi %acc, %outB : i32
        scf.yield %hA, %hB1, %argA, %outB, %doneA, %doneB1, %acc1
            : i64, i64, i32, i32, i1, i1, i32
      }

      scf.yield %hA_next, %hB_next, %argA_next, %argB_next, %doneA_next,
          %doneB_next, %nextTurn, %acc_next : i64, i64, i32, i32, i1, i1, i32,
          i32
    }

    return %loop#7 : i32
  }
}
