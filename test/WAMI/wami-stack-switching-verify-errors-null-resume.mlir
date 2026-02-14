// RUN: wasm-opt %s -verify-diagnostics

module {
  wami.type.func @ft = () -> i32
  wami.type.cont @ct = cont @ft

  wasmssa.func @driver() -> i32 {
    %c = wami.ref.null : !wami.cont<@ct, true>
    // expected-error @+1 {{resume requires non-null continuation}}
    %r = "wami.resume"(%c) <{cont_type = @ct, handlers = []}> : (!wami.cont<@ct, true>) -> i32
    wasmssa.return %r : i32
  }
}
