// RUN: wasm-opt %s --verify-wasmstack -split-input-file -verify-diagnostics

wasmstack.module {
  wasmstack.type.func @ft = (i32) -> i32
  wasmstack.type.cont @ct = cont @ft

  // Invalid for (on ... switch): non-empty tag inputs.
  wasmstack.tag @bad_switch : (i32) -> i64

  wasmstack.func @worker : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.return
  }

  wasmstack.func @driver : () -> i32 {
    wasmstack.i32.const 7
    wasmstack.ref.func @worker
    wasmstack.cont.new @ct
    // expected-error @+1 {{switch handler tag must have empty inputs}}
    wasmstack.resume @ct (@bad_switch -> switch)
    wasmstack.return
  }
}

// -----

wasmstack.module {
  wasmstack.type.func @ft = (i32) -> i32
  wasmstack.type.cont @ct = cont @ft

  // Invalid for (on ... switch): non-empty tag inputs.
  wasmstack.tag @bad_switch : (i32) -> i64

  wasmstack.func @worker : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.return
  }

  wasmstack.func @driver : () -> () {
    wasmstack.i32.const 7
    wasmstack.ref.func @worker
    wasmstack.cont.new @ct
    // expected-error @+1 {{switch handler tag must have empty inputs}}
    wasmstack.resume_throw @ct (@bad_switch -> switch)
    wasmstack.return
  }
}

// -----

wasmstack.module {
  wasmstack.type.func @ft = (i32) -> i32
  wasmstack.type.cont @ct = cont @ft

  // Invalid for switch: non-empty tag inputs.
  wasmstack.tag @bad_switch : (i32) -> ()

  wasmstack.func @worker : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.return
  }

  wasmstack.func @driver : () -> () {
    // cont argument
    wasmstack.i32.const 7
    // tag payload (invalid shape for switch)
    wasmstack.i32.const 11
    wasmstack.ref.func @worker
    wasmstack.cont.new @ct
    // expected-error @+1 {{switch tag must have empty inputs}}
    wasmstack.switch @ct (tag: @bad_switch)
    wasmstack.return
  }
}
