// RUN: wasm-opt %s -verify-diagnostics

module {
  wami.type.func @ft = () -> i32
  wami.type.cont @ct = cont @ft

  wasmssa.func @driver() -> i32 {
    // expected-error @+1 {{ref.null for continuation requires nullable continuation type}}
    %c = wami.ref.null : !wami.cont<@ct>
    %r = wasmssa.const 0 : i32
    wasmssa.return %r : i32
  }
}
