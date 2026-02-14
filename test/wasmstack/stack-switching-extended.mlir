// RUN: wasm-opt %s --verify-wasmstack | FileCheck %s

// CHECK-LABEL: wasmstack.module
wasmstack.module {
  wasmstack.type.func @f = (i32) -> i32
  wasmstack.type.cont @c = cont @f
  wasmstack.tag @yield : () -> i32

  wasmstack.func @worker : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @resume_ok
  // CHECK: wasmstack.resume @c (@yield -> switch)
  wasmstack.func @resume_ok : () -> i32 {
    wasmstack.i32.const 7
    wasmstack.ref.func @worker
    wasmstack.cont.new @c
    wasmstack.resume @c (@yield -> switch)
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @resume_throw_ok
  // CHECK: wasmstack.resume_throw @c (@yield -> switch)
  wasmstack.func @resume_throw_ok : () -> () {
    wasmstack.i32.const 9
    wasmstack.ref.func @worker
    wasmstack.cont.new @c
    wasmstack.resume_throw @c (@yield -> switch)
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @barrier_ok
  // CHECK: wasmstack.barrier : (i32) -> i32
  wasmstack.func @barrier_ok : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.barrier : (i32) -> i32
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @nonnull_contref_ok
  // CHECK: wasmstack.local 0 : !wasmstack.contref_nonnull<@c>
  wasmstack.func @nonnull_contref_ok : () -> () {
    wasmstack.local 0 : !wasmstack.contref_nonnull<@c>
    wasmstack.ref.func @worker
    wasmstack.cont.new @c
    wasmstack.local.set 0 : !wasmstack.contref_nonnull<@c>
    wasmstack.return
  }
}
