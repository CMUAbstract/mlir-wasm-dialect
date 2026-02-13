// RUN: wasm-opt %s --verify-wasmstack | FileCheck %s

// CHECK-LABEL: wasmstack.module
wasmstack.module {
  wasmstack.type.func @f = (i32) -> i32
  wasmstack.type.cont @c = cont @f
  wasmstack.tag @yield : (i32) -> i32

  wasmstack.func @worker : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @resume_ok
  // CHECK: wasmstack.resume @c (@yield -> @h)
  wasmstack.func @resume_ok : () -> i32 {
    wasmstack.ref.func @worker
    wasmstack.cont.new @c
    wasmstack.i32.const 7
    wasmstack.resume @c (@yield -> @h)
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @resume_throw_ok
  // CHECK: wasmstack.resume_throw @c (@yield -> @h)
  wasmstack.func @resume_throw_ok : () -> () {
    wasmstack.ref.func @worker
    wasmstack.cont.new @c
    wasmstack.i32.const 9
    wasmstack.resume_throw @c (@yield -> @h)
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @barrier_ok
  // CHECK: wasmstack.barrier : (i32) -> i32
  wasmstack.func @barrier_ok : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.barrier : (i32) -> i32
    wasmstack.return
  }
}
