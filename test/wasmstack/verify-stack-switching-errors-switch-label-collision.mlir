// RUN: wasm-opt %s --verify-wasmstack -verify-diagnostics

wasmstack.module {
  wasmstack.type.func @ft = () -> i32
  wasmstack.type.cont @ct = cont @ft
  wasmstack.tag @yield : () -> ()

  wasmstack.func @worker : () -> i32 {
    wasmstack.i32.const 0
    wasmstack.return
  }

  wasmstack.func @driver : () -> i32 {
    // Label name intentionally collides with current @switch sentinel string.
    wasmstack.block @switch : ([]) -> [] {
      wasmstack.ref.func @worker
      wasmstack.cont.new @ct
      // expected-error @+1 {{handler label @switch expects 0 values but handler passes 1}}
      wasmstack.resume @ct (@yield -> @switch)
      wasmstack.drop : i32
    }

    wasmstack.i32.const 0
    wasmstack.return
  }
}
