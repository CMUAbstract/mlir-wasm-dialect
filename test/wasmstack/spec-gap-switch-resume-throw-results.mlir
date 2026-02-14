// XFAIL: *
// RUN: wasm-opt %s --verify-wasmstack -split-input-file

// TODO(issue-draft): .omc/issue-drafts/spec-gap-switch-resume-throw-result-typing.md
// Desired future surface: switch/resume_throw should support typed result flow
// per stack-switching spec typing rules.

wasmstack.module {
  wasmstack.type.func @ft = () -> i32
  wasmstack.type.cont @ct = cont @ft
  wasmstack.tag @yield : () -> i32

  wasmstack.func @worker : () -> i32 {
    wasmstack.i32.const 1
    wasmstack.return
  }

  wasmstack.func @driver : () -> i32 {
    wasmstack.ref.func @worker
    wasmstack.cont.new @ct
    %r = wasmstack.resume_throw @ct (@yield -> @switch)
    wasmstack.return
  }
}

// -----

wasmstack.module {
  wasmstack.type.func @ft = () -> i32
  wasmstack.type.cont @ct = cont @ft
  wasmstack.tag @yield : () -> ()

  wasmstack.func @worker : () -> i32 {
    wasmstack.i32.const 2
    wasmstack.return
  }

  wasmstack.func @driver : () -> i32 {
    wasmstack.ref.func @worker
    wasmstack.cont.new @ct
    %r = wasmstack.switch @ct (tag: @yield)
    wasmstack.return
  }
}
