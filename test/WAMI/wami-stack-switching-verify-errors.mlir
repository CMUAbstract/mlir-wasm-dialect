// RUN: wasm-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error @+1 {{unknown wami.type.func symbol}}
  wami.type.cont @missing_type = cont @does_not_exist
}

// -----

module {
  wami.tag @yield : (i32) -> i32
  wasmssa.func @payload_mismatch(%x: !wasmssa<local ref to i64>) -> i32 {
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i64>
    // expected-error @+1 {{suspend payload types type mismatch at index 0}}
    %r = wami.suspend @yield(%arg) : (i64) -> i32
    wasmssa.return %r : i32
  }
}

// -----

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @bad_handler_attr(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @bad_handler_attr : !wami.funcref<@bad_handler_attr>
    %c = wami.cont.new %f : !wami.funcref<@bad_handler_attr> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    // expected-error @+1 {{handlers must contain #wami.on_label or #wami.on_switch attributes}}
    %r = wami.resume %c(%arg) @gen_ct [@yield] : (!wami.cont<@gen_ct>, i32) -> i32
    wasmssa.return %r : i32
  }
}

// -----

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @dup_handler_tag(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @dup_handler_tag : !wami.funcref<@dup_handler_tag>
    %c = wami.cont.new %f : !wami.funcref<@dup_handler_tag> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>

    wasmssa.block : {
    ^bb0:
      // expected-error @+1 {{duplicate handler tag @yield}}
      %r = wami.resume %c(%arg) @gen_ct [#wami.on_label<tag = @yield, level = 0>, #wami.on_switch<tag = @yield>] : (!wami.cont<@gen_ct>, i32) -> i32
      wasmssa.return %r : i32
    }> ^on_yield

  ^on_yield(%payload: i32, %k: !wami.cont<@gen_ct>):
    wasmssa.return %payload : i32
  }
}

// -----

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @negative_level(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @negative_level : !wami.funcref<@negative_level>
    %c = wami.cont.new %f : !wami.funcref<@negative_level> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>

    wasmssa.block : {
    ^bb0:
      // expected-error @+1 {{on_label level must be non-negative}}
      %r = wami.resume %c(%arg) @gen_ct [#wami.on_label<tag = @yield, level = -1>] : (!wami.cont<@gen_ct>, i32) -> i32
      wasmssa.return %r : i32
    }> ^on_yield

  ^on_yield(%payload: i32, %k: !wami.cont<@gen_ct>):
    wasmssa.return %payload : i32
  }
}

// -----

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @level_oob(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @level_oob : !wami.funcref<@level_oob>
    %c = wami.cont.new %f : !wami.funcref<@level_oob> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>

    wasmssa.block : {
    ^bb0:
      // expected-error @+1 {{on_label level 1 exceeds enclosing structured label depth 1}}
      %r = wami.resume %c(%arg) @gen_ct [#wami.on_label<tag = @yield, level = 1>] : (!wami.cont<@gen_ct>, i32) -> i32
      wasmssa.return %r : i32
    }> ^on_yield

  ^on_yield(%payload: i32, %k: !wami.cont<@gen_ct>):
    wasmssa.return %payload : i32
  }
}

// -----

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @bad_resume_throw(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @bad_resume_throw : !wami.funcref<@bad_resume_throw>
    %c = wami.cont.new %f : !wami.funcref<@bad_resume_throw> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>

    wasmssa.block : {
    ^bb0:
      // expected-error @+1 {{'wami.resume_throw' op requires zero results}}
      %r = "wami.resume_throw"(%c, %arg) <{cont_type = @gen_ct, handlers = [#wami.on_label<tag = @yield, level = 0>]}> : (!wami.cont<@gen_ct>, i32) -> i32
      wasmssa.return %r : i32
    }> ^on_yield

  ^on_yield(%payload: i32, %k: !wami.cont<@gen_ct>):
    wasmssa.return %payload : i32
  }
}
