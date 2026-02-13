// RUN: wasm-opt %s --verify-wasmstack -split-input-file -verify-diagnostics

wasmstack.module {
  wasmstack.type.func @takes_i32 = (i32) -> i32
  wasmstack.type.func @takes_i64 = (i64) -> i32
  wasmstack.type.cont @cont_i32 = cont @takes_i32

  wasmstack.func @worker_i64 : (i64) -> i32 {
    wasmstack.i32.const 0
    wasmstack.return
  }

  wasmstack.func @bad_cont_new_signature : () -> i32 {
    wasmstack.ref.func @worker_i64
    // expected-error @+1 {{funcref signature does not match continuation type}}
    wasmstack.cont.new @cont_i32
    wasmstack.i32.const 0
    wasmstack.return
  }
}

// -----

wasmstack.module {
  wasmstack.type.func @takes_i64 = (i64) -> i32
  wasmstack.type.cont @cont_i64 = cont @takes_i64
  wasmstack.tag @yield_i32 : (i32) -> i32

  wasmstack.func @worker_i64 : (i64) -> i32 {
    wasmstack.i32.const 0
    wasmstack.return
  }

  wasmstack.func @bad_resume_missing_arg : () -> i32 {
    wasmstack.ref.func @worker_i64
    wasmstack.cont.new @cont_i64
    // expected-error @+1 {{type mismatch}}
    wasmstack.resume @cont_i64 (@yield_i32 -> @h)
    wasmstack.i32.const 0
    wasmstack.return
  }
}

// -----

wasmstack.module {
  wasmstack.type.func @takes_i64 = (i64) -> i32
  wasmstack.type.cont @cont_i64 = cont @takes_i64

  wasmstack.func @worker_i64 : (i64) -> i32 {
    wasmstack.i32.const 0
    wasmstack.return
  }

  wasmstack.func @bad_resume_unknown_tag : () -> i32 {
    wasmstack.ref.func @worker_i64
    wasmstack.cont.new @cont_i64
    wasmstack.i64.const 1
    // expected-error @+1 {{unknown wasmstack.tag symbol}}
    wasmstack.resume @cont_i64 (@missing_tag -> @h)
    wasmstack.return
  }
}
