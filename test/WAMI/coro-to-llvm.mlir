// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --coro-to-llvm | FileCheck %s

module {
  func.func private @coro.spawn.generator() -> i64
  func.func private @coro.resume.generator(%h: i64, %x: i32) -> i32
  func.func private @coro.is_done.generator(%h: i64) -> i1
  func.func private @coro.cancel.generator(%h: i64)
  func.func private @coro.yield.tick(%x: i32) -> i32

  // CHECK-LABEL: func.func @coro.impl.generator
  // CHECK-NOT: @coro.yield.tick
  func.func @coro.impl.generator(%x: i32) -> i32 {
    %y = func.call @coro.yield.tick(%x) : (i32) -> i32
    return %y : i32
  }

// CHECK-LABEL: func.func @main
// CHECK: %[[H:.*]] = arith.constant 1 : i64
// CHECK: %[[D:.*]] = arith.constant true
// CHECK: %[[OUT:.*]] = scf.if
// CHECK: %[[R:.*]] = func.call @coro.impl.generator(
// CHECK-NOT: @coro.resume.generator
// CHECK-NOT: @coro.cancel.generator
// CHECK: return %[[OUT]] : i32
  func.func @main() -> i32 {
    %h = func.call @coro.spawn.generator() : () -> i64
    %d = func.call @coro.is_done.generator(%h) : (i64) -> i1
    %x = arith.constant 41 : i32
    %r = scf.if %d -> i32 {
      %v = func.call @coro.resume.generator(%h, %x) : (i64, i32) -> i32
      func.call @coro.cancel.generator(%h) : (i64) -> ()
      scf.yield %v : i32
    } else {
      %z = arith.constant 0 : i32
      scf.yield %z : i32
    }
    return %r : i32
  }
}
