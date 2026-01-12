// RUN: wasm-opt %s | wasm-opt | FileCheck %s

// CHECK-LABEL: wasmstack.module @test_module
wasmstack.module @test_module {

  // CHECK-LABEL: wasmstack.memory @mem
  wasmstack.memory @mem min = 1

  // CHECK-LABEL: wasmstack.global @counter
  wasmstack.global @counter : i32 mutable {
    wasmstack.i32.const 0
  }

  // CHECK-LABEL: wasmstack.func @add
  wasmstack.func @add : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @factorial
  wasmstack.func @factorial : (i32) -> i32 {
    wasmstack.block @done : () -> [i32] {
      // Check if n <= 1
      wasmstack.local.get 0 : i32
      wasmstack.i32.const 1
      wasmstack.le_s : i32
      wasmstack.br_if @done

      // Compute n * factorial(n-1)
      wasmstack.local.get 0 : i32
      wasmstack.local.get 0 : i32
      wasmstack.i32.const 1
      wasmstack.sub : i32
      wasmstack.call @factorial : (i32) -> i32
      wasmstack.mul : i32
      wasmstack.br @done
    }
    wasmstack.i32.const 1
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @loop_example
  wasmstack.func @loop_example : (i32) -> i32 {
    // Initialize sum to 0
    wasmstack.i32.const 0
    wasmstack.local.set 1 : i32

    wasmstack.block @exit : () -> [i32] {
      wasmstack.loop @continue : () -> [] {
        // Decrement counter
        wasmstack.local.get 0 : i32
        wasmstack.i32.const 1
        wasmstack.sub : i32
        wasmstack.local.tee 0 : i32

        // If counter is 0, exit
        wasmstack.eqz : i32
        wasmstack.br_if @exit

        // Add counter to sum
        wasmstack.local.get 1 : i32
        wasmstack.local.get 0 : i32
        wasmstack.add : i32
        wasmstack.local.set 1 : i32

        // Continue loop
        wasmstack.br @continue
      }
      wasmstack.local.get 1 : i32
    }
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @memory_ops
  wasmstack.func @memory_ops : (i32, i32) -> i32 {
    // Store value
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.i32.store offset = 0 align = 4 : i32

    // Load it back
    wasmstack.local.get 0 : i32
    wasmstack.i32.load offset = 0 align = 4 : i32
    wasmstack.return
  }

  // CHECK-LABEL: wasmstack.func @conditional
  wasmstack.func @conditional : (i32, i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.if : () -> [i32] then {
      wasmstack.local.get 1 : i32
    } else {
      wasmstack.local.get 2 : i32
    }
    wasmstack.return
  }
}
