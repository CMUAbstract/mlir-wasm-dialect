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
    %r = "wami.suspend"(%arg) <{tag = @yield}> : (i64) -> i32
    wasmssa.return %r : i32
  }
}

// -----

module {
  wami.type.func @gen_ft = (i32) -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @yield : (i32) -> i32
  wami.tag @other : (i32) -> i32

  wasmssa.func @bad_resume_handler_arity(%x: !wasmssa<local ref to i32>) -> i32 {
    %f = wami.ref.func @bad_resume_handler_arity : !wami.funcref<@bad_resume_handler_arity>
    %c = wami.cont.new %f : !wami.funcref<@bad_resume_handler_arity> as @gen_ct -> !wami.cont<@gen_ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    // expected-error @+1 {{handler region block count must match handler_tags count}}
    %r = "wami.resume"(%c, %arg) <{cont_type = @gen_ct, handler_tags = [@yield, @other]}> ({
    ^bb0(%payload: i32):
      wami.handler.yield %payload : i32
    }) : (!wami.cont<@gen_ct>, i32) -> i32
    wasmssa.return %r : i32
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
    // expected-error @+1 {{resume_throw handlers must not terminate with wami.handler.yield}}
    "wami.resume_throw"(%c, %arg) <{cont_type = @gen_ct, handler_tags = [@yield]}> ({
    ^bb0(%payload: i32):
      wami.handler.yield %payload : i32
    }) : (!wami.cont<@gen_ct>, i32) -> ()
    %zero = wasmssa.const 0 : i32
    wasmssa.return %zero : i32
  }
}
