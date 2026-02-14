// RUN: wasm-opt %s -verify-diagnostics

module {
  wami.type.func @ft = () -> i32
  wami.type.cont @ct = cont @ft

  wasmssa.func @driver() {
    %c = wami.ref.null : !wami.cont<@ct, true>
    // expected-error @+1 {{resume_throw requires non-null continuation}}
    "wami.resume_throw"(%c) <{cont_type = @ct, handlers = []}> : (!wami.cont<@ct, true>) -> ()
    wasmssa.return
  }
}
