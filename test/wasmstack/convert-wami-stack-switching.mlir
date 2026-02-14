// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: wasmstack.module
// CHECK-DAG: wasmstack.type.func @gen_ft = (i32) -> i32
// CHECK-DAG: wasmstack.type.cont @gen_ct = cont @gen_ft
// CHECK-DAG: wasmstack.tag @yield : (i32) -> i32
// CHECK-DAG: wasmstack.tag @switch_yield : () -> i32
// CHECK-DAG: wasmstack.type.func @src_ft = (i32, i32) -> i32
// CHECK-DAG: wasmstack.type.func @dst_ft = (i32) -> i32
// CHECK-DAG: wasmstack.type.cont @src_ct = cont @src_ft
// CHECK-DAG: wasmstack.type.cont @dst_ct = cont @dst_ft

// CHECK-LABEL: wasmstack.func @driver_label
// CHECK: wasmstack.block @[[ON:[A-Za-z0-9_]+]]
// CHECK: wasmstack.local.get [[ARG_LABEL:[0-9]+]] : i32
// CHECK-NEXT: wasmstack.local.get [[CONT_LABEL:[0-9]+]] : !wasmstack.contref_nonnull<@gen_ct>
// CHECK-NEXT: wasmstack.resume @gen_ct (@yield -> @[[ON]])

// CHECK-LABEL: wasmstack.func @driver_switch
// CHECK: wasmstack.local.get [[ARG_SWITCH:[0-9]+]] : i32
// CHECK: wasmstack.local.get [[CONT_SWITCH:[0-9]+]] : !wasmstack.contref_nonnull<@gen_ct>
// CHECK-NEXT: wasmstack.resume @gen_ct (@switch_yield -> @switch)

// CHECK-LABEL: wasmstack.func @driver_bind
// CHECK: wasmstack.cont.new @src_ct
// CHECK: wasmstack.i32.const 7
// CHECK: wasmstack.cont.bind @src_ct -> @dst_ct
// CHECK: wasmstack.local.get [[ARG_BIND:[0-9]+]] : i32
// CHECK: wasmstack.local.get [[CONT_BIND:[0-9]+]] : !wasmstack.contref_nonnull<@dst_ct>
// CHECK-NEXT: wasmstack.resume @dst_ct (@switch_yield -> @switch)

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32
  wami.tag @switch_yield : () -> i32
  wami.type.func @src_ft = (i32, i32) -> i32
  wami.type.func @dst_ft = (i32) -> i32
  wami.type.cont @src_ct = cont @src_ft
  wami.type.cont @dst_ct = cont @dst_ft

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
    %r = "wami.resume"(%c, %arg) <{cont_type = @gen_ct, handlers = [#wami.on_switch<tag = @switch_yield>]}> : (!wami.cont<@gen_ct>, i32) -> i32
    wasmssa.return %r : i32
  }

  wasmssa.func @worker2(%x: !wasmssa<local ref to i32>, %y: !wasmssa<local ref to i32>) -> i32 {
    %lx = wasmssa.local_get %x : !wasmssa<local ref to i32>
    %ly = wasmssa.local_get %y : !wasmssa<local ref to i32>
    %sum = wasmssa.add %lx %ly : i32
    wasmssa.return %sum : i32
  }

  wasmssa.func @driver_bind(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @worker2 : !wami.funcref<@worker2>
    %c = wami.cont.new %f : !wami.funcref<@worker2> as @src_ct -> !wami.cont<@src_ct>
    %bound = wasmssa.const 7 : i32
    %cb = "wami.cont.bind"(%c, %bound) <{src_cont_type = @src_ct, dst_cont_type = @dst_ct}> : (!wami.cont<@src_ct>, i32) -> !wami.cont<@dst_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    %r = "wami.resume"(%cb, %arg) <{cont_type = @dst_ct, handlers = [#wami.on_switch<tag = @switch_yield>]}> : (!wami.cont<@dst_ct>, i32) -> i32
    wasmssa.return %r : i32
  }
}
