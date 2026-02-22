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
    // CHECK: wami.resume {{.*}} @gen_ct [#wami.on_label<tag = @yield, level = 0>]
    wasmssa.block : {
    ^bb0:
      %r = wami.resume %c(%arg) @gen_ct [#wami.on_label<tag = @yield, level = 0>] : (!wami.cont<@gen_ct>, i32) -> i32
      wasmssa.return %r : i32
    }> ^on_yield

  ^on_yield(%payload: i32, %k: !wami.cont<@gen_ct>):
    wasmssa.return %payload : i32
  }
}
