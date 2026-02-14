// RUN: wasm-opt %s -verify-diagnostics

module {
  wami.type.func @expect_ft = (i32) -> i32
  wami.type.cont @ct = cont @expect_ft

  wasmssa.func @worker_noarg() -> i32 {
    %z = wasmssa.const 0 : i32
    wasmssa.return %z : i32
  }

  wasmssa.func @driver() -> i32 {
    %f = wami.ref.func @worker_noarg : !wami.funcref<@worker_noarg>
    // expected-error @+1 {{funcref signature does not match continuation type}}
    %c = wami.cont.new %f : !wami.funcref<@worker_noarg> as @ct -> !wami.cont<@ct>
    %z = wasmssa.const 0 : i32
    wasmssa.return %z : i32
  }
}
