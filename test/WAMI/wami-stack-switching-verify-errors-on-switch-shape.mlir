// RUN: wasm-opt %s -verify-diagnostics

module {
  wami.type.func @ct_ft = (i32) -> i32
  wami.type.cont @ct = cont @ct_ft

  // Invalid for (on ... switch): non-empty tag inputs.
  wami.tag @bad_switch : (i32) -> i64

  wasmssa.func @worker(%x: !wasmssa<local ref to i32>) -> i32 {
    %v = wasmssa.local_get %x : !wasmssa<local ref to i32>
    wasmssa.return %v : i32
  }

  wasmssa.func @driver(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @ct -> !wami.cont<@ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    // expected-error @+1 {{on_switch handler tag must have empty inputs}}
    %r = "wami.resume"(%c, %arg) <{cont_type = @ct, handlers = [#wami.on_switch<tag = @bad_switch>]}> : (!wami.cont<@ct>, i32) -> i32
    wasmssa.return %r : i32
  }

  wasmssa.func @driver_throw(%x: !wasmssa<local ref to i32>) {
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @ct -> !wami.cont<@ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    // expected-error @+1 {{on_switch handler tag must have empty inputs}}
    "wami.resume_throw"(%c, %arg) <{cont_type = @ct, handlers = [#wami.on_switch<tag = @bad_switch>]}> : (!wami.cont<@ct>, i32) -> ()
    wasmssa.return
  }
}
