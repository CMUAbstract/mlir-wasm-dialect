// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: wasmstack.module
// CHECK: wasmstack.type.func @gen_ft = (i32) -> i32
// CHECK: wasmstack.type.cont @gen_ct = cont @gen_ft
// CHECK: wasmstack.tag @yield : (i32) -> i32

// CHECK-LABEL: wasmstack.func @driver_label
// CHECK: wasmstack.block @[[ON:[A-Za-z0-9_]+]]
// CHECK: wasmstack.resume @gen_ct (@yield -> @[[ON]])

// CHECK-LABEL: wasmstack.func @driver_switch
// CHECK: wasmstack.resume @gen_ct (@yield -> @switch)

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @worker(%x: !wasmssa<local ref to i32>) -> i32 {
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    wasmssa.return %arg : i32
  }

  wasmssa.func @driver_label(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>

    wasmssa.block : {
    ^bb0:
      %r = "wami.resume"(%c, %arg) <{cont_type = @gen_ct, handlers = [#wami.on_label<tag = @yield, level = 0>]}> : (!wami.cont<@gen_ct>, i32) -> i32
      wasmssa.return %r : i32
    }> ^on_yield

  ^on_yield(%payload: i32, %k: !wami.cont<@gen_ct>):
    wasmssa.return %payload : i32
  }

  wasmssa.func @driver_switch(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    %r = "wami.resume"(%c, %arg) <{cont_type = @gen_ct, handlers = [#wami.on_switch<tag = @yield>]}> : (!wami.cont<@gen_ct>, i32) -> i32
    wasmssa.return %r : i32
  }
}
