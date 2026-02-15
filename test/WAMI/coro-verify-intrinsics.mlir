// RUN: wasm-opt %s --coro-verify-intrinsics | FileCheck %s

module {
  func.func private @coro.spawn.generator() -> i64
  func.func private @coro.resume.generator(%h: i64, %x: i32) -> i32
  func.func private @coro.is_done.generator(%h: i64) -> i1
  func.func private @coro.cancel.generator(%h: i64)
  func.func private @coro.yield.tick(%x: i32) -> i32

  // CHECK-LABEL: func.func @coro.impl.generator
  func.func @coro.impl.generator(%x: i32) -> i32 {
    %y = func.call @coro.yield.tick(%x) : (i32) -> i32
    return %y : i32
  }

  // CHECK-LABEL: func.func @main
  func.func @main() -> i32 {
    %h = func.call @coro.spawn.generator() : () -> i64
    %x = arith.constant 41 : i32
    %r = func.call @coro.resume.generator(%h, %x) : (i64, i32) -> i32
    return %r : i32
  }
}
