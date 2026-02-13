// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: wasmstack.module
// CHECK: wasmstack.type.func @gen_ft = (i32) -> i32
// CHECK: wasmstack.type.cont @gen_ct = cont @gen_ft
// CHECK: wasmstack.tag @yield : (i32) -> i32
// CHECK-LABEL: wasmstack.func @driver
// CHECK: wasmstack.ref.func @worker
// CHECK: wasmstack.cont.new @gen_ct
// CHECK: wasmstack.resume @gen_ct (@yield -> @__wami_resume_handler_0)

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @worker(%x: !wasmssa<local ref to i32>) -> i32 {
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    wasmssa.return %arg : i32
  }

  wasmssa.func @driver() -> i32 {
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.const 7 : i32
    %r = "wami.resume"(%c, %arg) <{cont_type = @gen_ct, handler_tags = [@yield]}> ({
    ^bb0(%payload: i32):
      wami.handler.yield %payload : i32
    }) : (!wami.cont<@gen_ct>, i32) -> i32
    wasmssa.return %r : i32
  }
}
