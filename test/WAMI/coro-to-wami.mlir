// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami | FileCheck %s

module {
  func.func private @coro.spawn.generator() -> i64
  func.func private @coro.resume.generator(%h: i64, %x: i32) -> i32

  func.func @coro.impl.generator(%x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %r = arith.addi %x, %c1 : i32
    return %r : i32
  }

  func.func @main() -> i32 {
    %h = func.call @coro.spawn.generator() : () -> i64
    %x = arith.constant 41 : i32
    %r = func.call @coro.resume.generator(%h, %x) : (i64, i32) -> i32
    return %r : i32
  }
}

// CHECK: wami.type.func @coro_ft_generator_b0 = (i32) -> i32
// CHECK: wami.type.cont @coro_ct_generator_b0 = cont @coro_ft_generator_b0
// CHECK-LABEL: wasmssa.func @main
// CHECK: wami.ref.func @coro.impl.generator
// CHECK: wami.cont.new
// CHECK: "wami.resume"
// CHECK-NOT: wasmssa.call @coro.spawn.generator
// CHECK-NOT: wasmssa.call @coro.resume.generator
