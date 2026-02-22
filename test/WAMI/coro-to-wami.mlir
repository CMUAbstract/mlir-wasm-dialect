// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami | FileCheck %s

module {
  func.func private @coro.spawn.generator() -> i64
  func.func private @coro.resume.generator(%h: i64, %x: i32)
      -> (i64, i1, i32)
  func.func private @coro.is_done.generator(%h: i64) -> i1
  func.func private @coro.cancel.generator(%h: i64)

  func.func @coro.impl.generator(%x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %r = arith.addi %x, %c1 : i32
    return %r : i32
  }

  func.func @main() -> i32 {
    %h = func.call @coro.spawn.generator() : () -> i64
    %x0 = arith.constant 20 : i32
    %h1, %d0, %r0 = func.call @coro.resume.generator(%h, %x0)
        : (i64, i32) -> (i64, i1, i32)
    %done = func.call @coro.is_done.generator(%h1) : (i64) -> i1
    func.call @coro.cancel.generator(%h1) : (i64) -> ()
    %x1 = arith.constant 21 : i32
    %h2, %d1, %r1 = func.call @coro.resume.generator(%h1, %x1)
        : (i64, i32) -> (i64, i1, i32)
    %sum = arith.addi %r0, %r1 : i32
    return %sum : i32
  }
}

// CHECK: wami.type.func @coro_ft_generator_b0 = (i32) -> i32
// CHECK: wami.type.cont @coro_ct_generator_b0 = cont @coro_ft_generator_b0
// CHECK: wasmssa.global @coro_slot_generator_1_state i32 mutable
// CHECK: wasmssa.global @coro_slot_generator_1_cont !wami.cont<@coro_ct_generator_b0, true> mutable
// CHECK: wami.tag @coro_tag_generator : (i32) -> i32
// CHECK-LABEL: wasmssa.func @coro.rt.resume.generator
// CHECK: wami.resume %{{.*}}(%{{.*}}) @coro_ct_generator_b0 [#wami.on_label<tag = @coro_tag_generator, level = 0>]
// CHECK-LABEL: wasmssa.func @main
// CHECK: wami.ref.func @coro.impl.generator
// CHECK: wami.cont.new
// CHECK: wasmssa.global_set @coro_slot_generator_1_cont
// CHECK: wasmssa.call @coro.rt.resume.generator
// CHECK: wasmssa.call @coro.rt.resume.generator
// CHECK-NOT: wasmssa.call @coro.spawn.generator
// CHECK-NOT: wasmssa.call @coro.resume.generator
// CHECK-NOT: wasmssa.call @coro.is_done.generator
// CHECK-NOT: wasmssa.call @coro.cancel.generator
