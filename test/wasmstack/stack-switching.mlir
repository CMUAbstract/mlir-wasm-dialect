// RUN: wasm-opt %s | wasm-opt | FileCheck %s

// Tests for stack switching (typed continuations) operations.

// CHECK-LABEL: wasmstack.module @stack_switching
wasmstack.module @stack_switching {

  // Define function type for generators
  // CHECK: wasmstack.type.func @gen_func = (i32) -> i32
  wasmstack.type.func @gen_func = (i32) -> i32

  // Define continuation type
  // CHECK: wasmstack.type.cont @gen_cont = cont @gen_func
  wasmstack.type.cont @gen_cont = cont @gen_func

  // Define a tag for yielding values
  // CHECK: wasmstack.tag @yield : (i32) -> i32
  wasmstack.tag @yield : (i32) -> i32

  // A generator function that yields values
  // CHECK-LABEL: wasmstack.func @generator
  wasmstack.func @generator : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.set 1 : i32  // result = init

    // Yield the current value
    wasmstack.local.get 1 : i32
    wasmstack.suspend @yield

    // After resume, add 1 and yield again
    wasmstack.i32.const 1
    wasmstack.add : i32
    wasmstack.local.tee 1 : i32
    wasmstack.suspend @yield

    // Final value
    wasmstack.local.get 1 : i32
    wasmstack.return
  }

  // Consumer that creates and resumes a continuation
  // CHECK-LABEL: wasmstack.func @run_generator
  wasmstack.func @run_generator : () -> i32 {
    // Push continuation argument first; resume consumes cont on top.
    wasmstack.i32.const 10

    // Get reference to generator function
    wasmstack.ref.func @generator

    // Create a new continuation
    wasmstack.cont.new @gen_cont

    // Resume with handler for yield tag
    wasmstack.block @handle_yield : ([]) -> [i32] {
      wasmstack.resume @gen_cont (@yield -> @handle_yield)
      // If generator returns normally, push result
    }

    // After suspension/completion
    wasmstack.return
  }

  // Example with cont.bind
  // CHECK-LABEL: wasmstack.func @partial_application
  wasmstack.func @partial_application : () -> i32 {
    // Get function reference
    wasmstack.ref.func @generator

    // Create continuation
    wasmstack.cont.new @gen_cont

    // Bind the first argument
    wasmstack.i32.const 42
    wasmstack.cont.bind @gen_cont -> @gen_cont

    // Now we have a continuation that needs no arguments
    wasmstack.i32.const 0  // dummy
    wasmstack.resume @gen_cont
    wasmstack.return
  }
}
