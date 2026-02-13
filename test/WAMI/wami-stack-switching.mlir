// RUN: wasm-opt %s | FileCheck %s

module {
  // CHECK: wami.type.func @gen_ft = (i32) -> i32
  wami.type.func @gen_ft = (i32) -> i32
  // CHECK: wami.type.cont @gen_ct = cont @gen_ft
  wami.type.cont @gen_ct = cont @gen_ft
  // CHECK: wami.tag @yield : (i32) -> i32
  wami.tag @yield : (i32) -> i32

  // CHECK-LABEL: wasmssa.func @driver
  wasmssa.func @driver(%x: !wasmssa<local ref to i32>) -> i32 {
    // CHECK: wami.ref.func @driver
    %f = wami.ref.func @driver : !wami.funcref<@driver>
    // CHECK: wami.cont.new
    %c = wami.cont.new %f : !wami.funcref<@driver> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    // CHECK: "wami.resume"
    // CHECK: wami.handler.yield
    %r = "wami.resume"(%c, %arg) <{cont_type = @gen_ct, handler_tags = [@yield]}> ({
    ^bb0(%payload: i32):
      wami.handler.yield %payload : i32
    }) : (!wami.cont<@gen_ct>, i32) -> i32
    wasmssa.return %r : i32
  }
}
