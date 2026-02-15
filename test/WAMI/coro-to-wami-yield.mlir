// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami | FileCheck %s

module {
  func.func private @coro.spawn.yielder() -> i64
  func.func private @coro.resume.yielder(%h: i64, %x: i32) -> i32
  func.func private @coro.yield.tick(%x: i32) -> i32

  func.func @coro.impl.yielder(%x: i32) -> i32 {
    %y = func.call @coro.yield.tick(%x) : (i32) -> i32
    return %y : i32
  }

  func.func @main() -> i32 {
    %h = func.call @coro.spawn.yielder() : () -> i64
    %x = arith.constant 7 : i32
    %r = func.call @coro.resume.yielder(%h, %x) : (i64, i32) -> i32
    return %r : i32
  }
}

// CHECK: wami.tag @coro_tag_tick : (i32) -> i32
// CHECK-LABEL: wasmssa.func @coro.impl.yielder
// CHECK: "wami.suspend"
