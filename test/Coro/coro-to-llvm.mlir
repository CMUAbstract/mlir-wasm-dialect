// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize | mlir-opt --lower-affine --convert-scf-to-cf --convert-arith-to-llvm="index-bitwidth=32" --convert-func-to-llvm="index-bitwidth=32" --convert-cf-to-llvm="index-bitwidth=32" --convert-to-llvm --reconcile-unrealized-casts | wasm-opt --coro-to-llvm | FileCheck %s

module {
  func.func private @coro.spawn.generator() -> i64
  func.func private @coro.resume.generator(%h: i64, %x: i32)
      -> (i64, i1, i32)
  func.func private @coro.is_done.generator(%h: i64) -> i1
  func.func private @coro.cancel.generator(%h: i64)
  func.func private @coro.yield.generator(%x: i32) -> i32

  // CHECK-LABEL: llvm.func @coro.rt.llvm.spawn.generator
  // CHECK-LABEL: llvm.func @coro.rt.llvm.resume.generator
  // CHECK: llvm.call @llvm.trap
  // CHECK-LABEL: llvm.func @coro.rt.llvm.is_done.generator
  // CHECK-LABEL: llvm.func @coro.rt.llvm.cancel.generator
  // CHECK-LABEL: llvm.func @coro.impl.generator
  // CHECK-SAME: (%[[X:arg[0-9]+]]: i32, %[[RT:arg[0-9]+]]: !llvm.ptr)
  // CHECK-DAG: llvm.call @llvm.coro.suspend
  // CHECK-DAG: llvm.call @llvm.coro.end
  // CHECK-DAG: llvm.call @llvm.coro.free
  func.func @coro.impl.generator(%x: i32) -> i32 {
    %y = func.call @coro.yield.generator(%x) : (i32) -> i32
    return %y : i32
  }

// CHECK-LABEL: llvm.func @main
// CHECK: llvm.call @coro.rt.llvm.spawn.generator
// CHECK: llvm.call @coro.rt.llvm.is_done.generator
// CHECK: llvm.call @coro.rt.llvm.resume.generator
// CHECK: llvm.call @coro.rt.llvm.cancel.generator
// CHECK-NOT: @coro.spawn.generator
// CHECK-NOT: @coro.resume.generator
// CHECK-NOT: @coro.is_done.generator
// CHECK-NOT: @coro.cancel.generator
  func.func @main() -> i32 {
    %h = func.call @coro.spawn.generator() : () -> i64
    %d = func.call @coro.is_done.generator(%h) : (i64) -> i1
    %x = arith.constant 41 : i32
    %r = scf.if %d -> i32 {
      %h2, %done2, %v = func.call @coro.resume.generator(%h, %x)
          : (i64, i32) -> (i64, i1, i32)
      func.call @coro.cancel.generator(%h) : (i64) -> ()
      scf.yield %v : i32
    } else {
      %z = arith.constant 0 : i32
      scf.yield %z : i32
    }
    return %r : i32
  }
}
