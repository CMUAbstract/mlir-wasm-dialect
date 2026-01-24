// RUN: wasm-opt %s -verify-wasmstack 2>&1 | FileCheck %s

// Comprehensive tests for the verify-wasmstack pass - valid programs only
// This file tests a wide range of stack patterns to ensure the verification
// pass correctly validates WebAssembly stack semantics.

//===----------------------------------------------------------------------===//
// Basic Arithmetic Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @arithmetic_ops
wasmstack.module @arithmetic_ops {

  // CHECK: wasmstack.func @all_binary_i32_ops
  wasmstack.func @all_binary_i32_ops : (i32, i32) -> i32 {
    // Test all i32 binary operations
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    wasmstack.local.get 1 : i32
    wasmstack.sub : i32
    wasmstack.local.get 1 : i32
    wasmstack.mul : i32
    wasmstack.local.get 1 : i32
    wasmstack.div_s : i32
    wasmstack.local.get 1 : i32
    wasmstack.div_u : i32
    wasmstack.local.get 1 : i32
    wasmstack.rem_s : i32
    wasmstack.local.get 1 : i32
    wasmstack.rem_u : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @all_bitwise_i32_ops
  wasmstack.func @all_bitwise_i32_ops : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.and : i32
    wasmstack.local.get 1 : i32
    wasmstack.or : i32
    wasmstack.local.get 1 : i32
    wasmstack.xor : i32
    wasmstack.local.get 1 : i32
    wasmstack.shl : i32
    wasmstack.local.get 1 : i32
    wasmstack.shr_s : i32
    wasmstack.local.get 1 : i32
    wasmstack.shr_u : i32
    wasmstack.local.get 1 : i32
    wasmstack.rotl : i32
    wasmstack.local.get 1 : i32
    wasmstack.rotr : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @all_unary_i32_ops
  wasmstack.func @all_unary_i32_ops : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.clz : i32
    wasmstack.ctz : i32
    wasmstack.popcnt : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @i64_arithmetic
  wasmstack.func @i64_arithmetic : (i64, i64) -> i64 {
    wasmstack.local.get 0 : i64
    wasmstack.local.get 1 : i64
    wasmstack.add : i64
    wasmstack.local.get 1 : i64
    wasmstack.mul : i64
    wasmstack.local.get 1 : i64
    wasmstack.sub : i64
    wasmstack.return
  }

  // CHECK: wasmstack.func @f32_arithmetic
  wasmstack.func @f32_arithmetic : (f32, f32) -> f32 {
    wasmstack.local.get 0 : f32
    wasmstack.local.get 1 : f32
    wasmstack.add : f32
    wasmstack.local.get 1 : f32
    wasmstack.sub : f32
    wasmstack.local.get 1 : f32
    wasmstack.mul : f32
    wasmstack.local.get 1 : f32
    wasmstack.div : f32
    wasmstack.local.get 1 : f32
    wasmstack.min : f32
    wasmstack.local.get 1 : f32
    wasmstack.max : f32
    wasmstack.local.get 1 : f32
    wasmstack.copysign : f32
    wasmstack.return
  }

  // CHECK: wasmstack.func @f32_unary
  wasmstack.func @f32_unary : (f32) -> f32 {
    wasmstack.local.get 0 : f32
    wasmstack.abs : f32
    wasmstack.neg : f32
    wasmstack.ceil : f32
    wasmstack.floor : f32
    wasmstack.trunc : f32
    wasmstack.nearest : f32
    wasmstack.sqrt : f32
    wasmstack.return
  }

  // CHECK: wasmstack.func @f64_arithmetic
  wasmstack.func @f64_arithmetic : (f64, f64) -> f64 {
    wasmstack.local.get 0 : f64
    wasmstack.local.get 1 : f64
    wasmstack.add : f64
    wasmstack.local.get 1 : f64
    wasmstack.mul : f64
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @comparison_ops
wasmstack.module @comparison_ops {

  // CHECK: wasmstack.func @i32_comparisons
  wasmstack.func @i32_comparisons : (i32, i32) -> i32 {
    // eqz test
    wasmstack.local.get 0 : i32
    wasmstack.eqz : i32
    wasmstack.drop : i32

    // All comparison operations
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.eq : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.ne : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.lt_s : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.lt_u : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.gt_s : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.gt_u : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.le_s : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.le_u : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.ge_s : i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.ge_u : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @i64_comparisons
  wasmstack.func @i64_comparisons : (i64, i64) -> i32 {
    wasmstack.local.get 0 : i64
    wasmstack.eqz : i64
    wasmstack.drop : i32

    wasmstack.local.get 0 : i64
    wasmstack.local.get 1 : i64
    wasmstack.lt_s : i64
    wasmstack.return
  }

  // CHECK: wasmstack.func @f32_comparisons
  wasmstack.func @f32_comparisons : (f32, f32) -> i32 {
    wasmstack.local.get 0 : f32
    wasmstack.local.get 1 : f32
    wasmstack.eq : f32
    wasmstack.drop : i32

    wasmstack.local.get 0 : f32
    wasmstack.local.get 1 : f32
    wasmstack.ne : f32
    wasmstack.drop : i32

    wasmstack.local.get 0 : f32
    wasmstack.local.get 1 : f32
    wasmstack.lt : f32
    wasmstack.drop : i32

    wasmstack.local.get 0 : f32
    wasmstack.local.get 1 : f32
    wasmstack.gt : f32
    wasmstack.drop : i32

    wasmstack.local.get 0 : f32
    wasmstack.local.get 1 : f32
    wasmstack.le : f32
    wasmstack.drop : i32

    wasmstack.local.get 0 : f32
    wasmstack.local.get 1 : f32
    wasmstack.ge : f32
    wasmstack.return
  }

  // CHECK: wasmstack.func @f64_comparisons
  wasmstack.func @f64_comparisons : (f64, f64) -> i32 {
    wasmstack.local.get 0 : f64
    wasmstack.local.get 1 : f64
    wasmstack.lt : f64
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Type Conversion Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @conversion_ops
wasmstack.module @conversion_ops {

  // CHECK: wasmstack.func @int_conversions
  wasmstack.func @int_conversions : (i32, i64) -> i64 {
    // i32 to i64
    wasmstack.local.get 0 : i32
    wasmstack.i64.extend_i32_s : i32 -> i64
    wasmstack.drop : i64

    wasmstack.local.get 0 : i32
    wasmstack.i64.extend_i32_u : i32 -> i64
    wasmstack.drop : i64

    // i64 to i32
    wasmstack.local.get 1 : i64
    wasmstack.i32.wrap_i64 : i64 -> i32
    wasmstack.drop : i32

    wasmstack.local.get 1 : i64
    wasmstack.return
  }

  // CHECK: wasmstack.func @float_int_conversions
  wasmstack.func @float_int_conversions : (f32, f64, i32, i64) -> f64 {
    // f32 to int
    wasmstack.local.get 0 : f32
    wasmstack.i32.trunc_f32_s : f32 -> i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : f32
    wasmstack.i32.trunc_f32_u : f32 -> i32
    wasmstack.drop : i32

    wasmstack.local.get 0 : f32
    wasmstack.i64.trunc_f32_s : f32 -> i64
    wasmstack.drop : i64

    // f64 to int
    wasmstack.local.get 1 : f64
    wasmstack.i32.trunc_f64_s : f64 -> i32
    wasmstack.drop : i32

    wasmstack.local.get 1 : f64
    wasmstack.i64.trunc_f64_s : f64 -> i64
    wasmstack.drop : i64

    // int to float
    wasmstack.local.get 2 : i32
    wasmstack.f32.convert_i32_s : i32 -> f32
    wasmstack.drop : f32

    wasmstack.local.get 2 : i32
    wasmstack.f64.convert_i32_s : i32 -> f64
    wasmstack.drop : f64

    wasmstack.local.get 3 : i64
    wasmstack.f64.convert_i64_s : i64 -> f64
    wasmstack.return
  }

  // CHECK: wasmstack.func @float_conversions
  wasmstack.func @float_conversions : (f32, f64) -> f64 {
    wasmstack.local.get 0 : f32
    wasmstack.f64.promote_f32 : f32 -> f64
    wasmstack.drop : f64

    wasmstack.local.get 1 : f64
    wasmstack.f32.demote_f64 : f64 -> f32
    wasmstack.drop : f32

    wasmstack.local.get 1 : f64
    wasmstack.return
  }

  // CHECK: wasmstack.func @reinterpret_conversions
  wasmstack.func @reinterpret_conversions : (i32, i64, f32, f64) -> i64 {
    wasmstack.local.get 0 : i32
    wasmstack.f32.reinterpret_i32 : i32 -> f32
    wasmstack.i32.reinterpret_f32 : f32 -> i32
    wasmstack.drop : i32

    wasmstack.local.get 1 : i64
    wasmstack.f64.reinterpret_i64 : i64 -> f64
    wasmstack.i64.reinterpret_f64 : f64 -> i64
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Local Variable Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @local_ops
wasmstack.module @local_ops {

  // CHECK: wasmstack.func @local_get_set_tee
  wasmstack.func @local_get_set_tee : (i32, i64) -> i32 {
    wasmstack.local 2 : i32
    wasmstack.local 3 : i64

    // local.set and local.get
    wasmstack.local.get 0 : i32
    wasmstack.i32.const 10
    wasmstack.add : i32
    wasmstack.local.set 2 : i32

    wasmstack.local.get 1 : i64
    wasmstack.i64.const 20
    wasmstack.add : i64
    wasmstack.local.set 3 : i64

    // local.tee - sets and keeps value on stack
    wasmstack.local.get 2 : i32
    wasmstack.local.tee 0 : i32
    wasmstack.local.get 0 : i32
    wasmstack.add : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @all_local_types
  wasmstack.func @all_local_types : () -> () {
    wasmstack.local 0 : i32
    wasmstack.local 1 : i64
    wasmstack.local 2 : f32
    wasmstack.local 3 : f64

    wasmstack.i32.const 1
    wasmstack.local.set 0 : i32

    wasmstack.i64.const 2
    wasmstack.local.set 1 : i64

    wasmstack.f32.const 3.0
    wasmstack.local.set 2 : f32

    wasmstack.f64.const 4.0
    wasmstack.local.set 3 : f64

    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @memory_ops
wasmstack.module @memory_ops {
  wasmstack.memory @mem min = 1

  // CHECK: wasmstack.func @basic_load_store
  wasmstack.func @basic_load_store : (i32, i32) -> i32 {
    // Store value
    wasmstack.local.get 0 : i32  // address
    wasmstack.local.get 1 : i32  // value
    wasmstack.i32.store offset = 0 align = 4 : i32

    // Load value back
    wasmstack.local.get 0 : i32  // address
    wasmstack.i32.load offset = 0 align = 4 : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @load_store_all_types
  wasmstack.func @load_store_all_types : (i32) -> i64 {
    wasmstack.local 1 : i64
    wasmstack.local 2 : f32
    wasmstack.local 3 : f64

    // i32 load/store
    wasmstack.local.get 0 : i32
    wasmstack.i32.const 42
    wasmstack.i32.store offset = 0 align = 4 : i32

    // i64 load/store
    wasmstack.local.get 0 : i32
    wasmstack.i64.const 100
    wasmstack.i64.store offset = 8 align = 8 : i64

    // f32 load/store
    wasmstack.local.get 0 : i32
    wasmstack.f32.const 3.14
    wasmstack.f32.store offset = 16 align = 4 : f32

    // f64 load/store
    wasmstack.local.get 0 : i32
    wasmstack.f64.const 2.718
    wasmstack.f64.store offset = 24 align = 8 : f64

    // Load i64 back
    wasmstack.local.get 0 : i32
    wasmstack.i64.load offset = 8 align = 8 : i64
    wasmstack.return
  }

  // CHECK: wasmstack.func @partial_loads
  wasmstack.func @partial_loads : (i32) -> i32 {
    // i32.load8_s
    wasmstack.local.get 0 : i32
    wasmstack.i32.load8_s offset = 0 align = 1 : i32
    wasmstack.drop : i32

    // i32.load8_u
    wasmstack.local.get 0 : i32
    wasmstack.i32.load8_u offset = 0 align = 1 : i32
    wasmstack.drop : i32

    // i32.load16_s
    wasmstack.local.get 0 : i32
    wasmstack.i32.load16_s offset = 0 align = 2 : i32
    wasmstack.drop : i32

    // i32.load16_u
    wasmstack.local.get 0 : i32
    wasmstack.i32.load16_u offset = 0 align = 2 : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @partial_stores
  wasmstack.func @partial_stores : (i32, i32) -> () {
    // i32.store8
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.i32.store8 offset = 0 align = 1 : i32

    // i32.store16
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.i32.store16 offset = 2 align = 2 : i32

    wasmstack.return
  }

  // CHECK: wasmstack.func @memory_size_grow
  wasmstack.func @memory_size_grow : () -> i32 {
    wasmstack.memory.size
    wasmstack.drop : i32

    wasmstack.i32.const 1
    wasmstack.memory.grow
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Control Flow - Blocks
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @block_control_flow
wasmstack.module @block_control_flow {

  // CHECK: wasmstack.func @simple_block
  wasmstack.func @simple_block : () -> i32 {
    wasmstack.block @b0 : ([]) -> [i32] {
      wasmstack.i32.const 42
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @block_with_params
  wasmstack.func @block_with_params : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.block @b0 : ([i32, i32]) -> [i32] {
      wasmstack.add : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @block_multi_value_result
  wasmstack.func @block_multi_value_result : () -> i32 {
    wasmstack.block @b0 : ([]) -> [i32, i32] {
      wasmstack.i32.const 10
      wasmstack.i32.const 20
    }
    wasmstack.add : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @nested_blocks_deep
  wasmstack.func @nested_blocks_deep : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.block @outer : ([i32]) -> [i32] {
      wasmstack.block @middle : ([i32]) -> [i32] {
        wasmstack.block @inner : ([i32]) -> [i32] {
          wasmstack.i32.const 1
          wasmstack.add : i32
        }
        wasmstack.i32.const 2
        wasmstack.add : i32
      }
      wasmstack.i32.const 3
      wasmstack.add : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @block_early_exit
  wasmstack.func @block_early_exit : (i32) -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.local.get 0 : i32
      wasmstack.eqz : i32
      wasmstack.if : ([]) -> [] then {
        wasmstack.i32.const 0
        wasmstack.br @exit
      }
      wasmstack.local.get 0 : i32
    }
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Control Flow - Loops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @loop_control_flow
wasmstack.module @loop_control_flow {

  // CHECK: wasmstack.func @simple_loop
  wasmstack.func @simple_loop : () -> i32 {
    wasmstack.i32.const 0
    wasmstack.loop @loop : ([i32]) -> [i32] {
      wasmstack.i32.const 1
      wasmstack.add : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @loop_with_exit
  wasmstack.func @loop_with_exit : (i32) -> i32 {
    wasmstack.local 1 : i32
    wasmstack.i32.const 0
    wasmstack.local.set 1 : i32

    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.loop @continue : ([]) -> [] {
        // Increment counter
        wasmstack.local.get 1 : i32
        wasmstack.i32.const 1
        wasmstack.add : i32
        wasmstack.local.set 1 : i32

        // Check if done
        wasmstack.local.get 1 : i32
        wasmstack.local.get 0 : i32
        wasmstack.ge_s : i32
        wasmstack.if : ([]) -> [] then {
          wasmstack.local.get 1 : i32
          wasmstack.br @exit
        }
        wasmstack.br @continue
      }
      wasmstack.i32.const 0
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @nested_loops
  wasmstack.func @nested_loops : (i32, i32) -> i32 {
    wasmstack.local 2 : i32
    wasmstack.i32.const 0
    wasmstack.local.set 2 : i32

    wasmstack.block @done : ([]) -> [i32] {
      wasmstack.loop @outer : ([]) -> [] {
        wasmstack.loop @inner : ([]) -> [] {
          wasmstack.local.get 2 : i32
          wasmstack.i32.const 1
          wasmstack.add : i32
          wasmstack.local.set 2 : i32

          wasmstack.local.get 2 : i32
          wasmstack.local.get 1 : i32
          wasmstack.lt_s : i32
          wasmstack.br_if @inner
        }

        wasmstack.local.get 2 : i32
        wasmstack.local.get 0 : i32
        wasmstack.ge_s : i32
        wasmstack.if : ([]) -> [] then {
          wasmstack.local.get 2 : i32
          wasmstack.br @done
        }
        wasmstack.br @outer
      }
      wasmstack.i32.const 0
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @loop_multivalue_params
  wasmstack.func @loop_multivalue_params : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.loop @loop : ([i32, i32]) -> [i32] {
      // On stack: [a, b]
      wasmstack.add : i32
      // Result on stack
    }
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Control Flow - If/Else
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @if_control_flow
wasmstack.module @if_control_flow {

  // CHECK: wasmstack.func @simple_if_else
  wasmstack.func @simple_if_else : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.i32.const 1
    } else {
      wasmstack.i32.const 0
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @if_no_result
  wasmstack.func @if_no_result : (i32) -> () {
    wasmstack.local.get 0 : i32
    wasmstack.if : ([]) -> [] then {
      wasmstack.nop
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @if_with_params
  wasmstack.func @if_with_params : (i32, i32, i32) -> i32 {
    // Push params first
    wasmstack.local.get 1 : i32
    wasmstack.local.get 2 : i32
    // Then condition
    wasmstack.local.get 0 : i32
    wasmstack.if : ([i32, i32]) -> [i32] then {
      wasmstack.add : i32
    } else {
      wasmstack.sub : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @nested_if
  wasmstack.func @nested_if : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.local.get 1 : i32
      wasmstack.if : ([]) -> [i32] then {
        wasmstack.i32.const 1
      } else {
        wasmstack.i32.const 2
      }
    } else {
      wasmstack.i32.const 3
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @if_multi_value_result
  wasmstack.func @if_multi_value_result : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.if : ([]) -> [i32, i32] then {
      wasmstack.i32.const 10
      wasmstack.i32.const 20
    } else {
      wasmstack.i32.const 30
      wasmstack.i32.const 40
    }
    wasmstack.add : i32
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Control Flow - Branches
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @branch_control_flow
wasmstack.module @branch_control_flow {

  // CHECK: wasmstack.func @br_unconditional
  wasmstack.func @br_unconditional : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.i32.const 42
      wasmstack.br @exit
      // This code is unreachable but valid
      wasmstack.i32.const 0
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @br_if_conditional
  wasmstack.func @br_if_conditional : (i32) -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      // Push result value
      wasmstack.i32.const 100
      // Push condition
      wasmstack.local.get 0 : i32
      wasmstack.br_if @exit
      // Fallthrough - drop the 100, produce different value
      wasmstack.drop : i32
      wasmstack.i32.const 200
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @br_table
  // All br_table targets must have the same arity
  wasmstack.func @br_table : (i32) -> i32 {
    wasmstack.local 1 : i32
    wasmstack.i32.const -1
    wasmstack.local.set 1 : i32

    // All blocks have void arity so br_table targets are consistent
    wasmstack.block @exit : ([]) -> [] {
      wasmstack.block @case2 : ([]) -> [] {
        wasmstack.block @case1 : ([]) -> [] {
          wasmstack.block @case0 : ([]) -> [] {
            wasmstack.local.get 0 : i32
            wasmstack.br_table [@case0, @case1, @case2] default @exit
          }
          // case 0
          wasmstack.i32.const 0
          wasmstack.local.set 1 : i32
          wasmstack.br @exit
        }
        // case 1
        wasmstack.i32.const 1
        wasmstack.local.set 1 : i32
        wasmstack.br @exit
      }
      // case 2
      wasmstack.i32.const 2
      wasmstack.local.set 1 : i32
      wasmstack.br @exit
    }
    // default (exit) - return the result
    wasmstack.local.get 1 : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @br_to_loop
  wasmstack.func @br_to_loop : (i32) -> i32 {
    wasmstack.local 1 : i32
    wasmstack.local.get 0 : i32
    wasmstack.local.set 1 : i32

    wasmstack.block @done : ([]) -> [i32] {
      wasmstack.loop @again : ([]) -> [] {
        wasmstack.local.get 1 : i32
        wasmstack.i32.const 1
        wasmstack.sub : i32
        wasmstack.local.tee 1 : i32
        wasmstack.eqz : i32
        wasmstack.if : ([]) -> [] then {
          wasmstack.local.get 1 : i32
          wasmstack.br @done
        }
        wasmstack.br @again
      }
      wasmstack.i32.const -1
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @br_with_values
  wasmstack.func @br_with_values : (i32) -> i32 {
    wasmstack.block @exit : ([]) -> [i32, i32] {
      wasmstack.local.get 0 : i32
      wasmstack.eqz : i32
      wasmstack.if : ([]) -> [] then {
        wasmstack.i32.const 1
        wasmstack.i32.const 2
        wasmstack.br @exit
      }
      wasmstack.i32.const 3
      wasmstack.i32.const 4
    }
    wasmstack.add : i32
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Polymorphic Stack (Unreachable Code)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @polymorphic_stack
wasmstack.module @polymorphic_stack {

  // CHECK: wasmstack.func @after_unreachable
  wasmstack.func @after_unreachable : () -> i32 {
    wasmstack.unreachable
    // Stack is polymorphic after unreachable - any operations are valid
    wasmstack.add : i32
    wasmstack.mul : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @after_br
  wasmstack.func @after_br : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.i32.const 42
      wasmstack.br @exit
      // After unconditional branch, stack is polymorphic
      wasmstack.sub : i32
      wasmstack.div_s : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @after_return
  wasmstack.func @after_return : () -> i32 {
    wasmstack.i32.const 0
    wasmstack.return
    // After return, stack is polymorphic
    wasmstack.add : i64
    wasmstack.f32.const 1.0
    wasmstack.return
  }

  // CHECK: wasmstack.func @after_br_table
  wasmstack.func @after_br_table : (i32) -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.i32.const 99
      wasmstack.br @exit
      wasmstack.local.get 0 : i32
      wasmstack.br_table [@exit] default @exit
      // Polymorphic after br_table
      wasmstack.add : f64
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @unreachable_if_branches
  wasmstack.func @unreachable_if_branches : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.unreachable
      // Polymorphic - can "produce" i32 without actually doing so
    } else {
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @function_calls
wasmstack.module @function_calls {

  // CHECK: wasmstack.func @callee_void
  wasmstack.func @callee_void : () -> () {
    wasmstack.return
  }

  // CHECK: wasmstack.func @callee_i32
  wasmstack.func @callee_i32 : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.i32.const 1
    wasmstack.add : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @callee_multi
  wasmstack.func @callee_multi : (i32, i32) -> (i32, i32) {
    wasmstack.local.get 1 : i32
    wasmstack.local.get 0 : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @caller
  wasmstack.func @caller : () -> i32 {
    // Call void function
    wasmstack.call @callee_void : () -> ()

    // Call with one arg, one result
    wasmstack.i32.const 10
    wasmstack.call @callee_i32 : (i32) -> i32
    wasmstack.drop : i32

    // Call with multiple args and results
    wasmstack.i32.const 1
    wasmstack.i32.const 2
    wasmstack.call @callee_multi : (i32, i32) -> (i32, i32)
    wasmstack.add : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @recursive_call
  wasmstack.func @recursive_call : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.eqz : i32
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.i32.const 1
    } else {
      wasmstack.local.get 0 : i32
      wasmstack.local.get 0 : i32
      wasmstack.i32.const 1
      wasmstack.sub : i32
      wasmstack.call @recursive_call : (i32) -> i32
      wasmstack.mul : i32
    }
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Stack Manipulation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @stack_manipulation
wasmstack.module @stack_manipulation {

  // CHECK: wasmstack.func @drop_values
  wasmstack.func @drop_values : () -> () {
    wasmstack.i32.const 1
    wasmstack.drop : i32

    wasmstack.i64.const 2
    wasmstack.drop : i64

    wasmstack.f32.const 3.0
    wasmstack.drop : f32

    wasmstack.f64.const 4.0
    wasmstack.drop : f64

    wasmstack.return
  }

  // CHECK: wasmstack.func @select_op
  wasmstack.func @select_op : (i32, i32, i32) -> i32 {
    wasmstack.local.get 0 : i32  // val1
    wasmstack.local.get 1 : i32  // val2
    wasmstack.local.get 2 : i32  // condition
    wasmstack.select : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @select_all_types
  wasmstack.func @select_all_types : (i32) -> f64 {
    // i32 select
    wasmstack.i32.const 1
    wasmstack.i32.const 2
    wasmstack.local.get 0 : i32
    wasmstack.select : i32
    wasmstack.drop : i32

    // i64 select
    wasmstack.i64.const 1
    wasmstack.i64.const 2
    wasmstack.local.get 0 : i32
    wasmstack.select : i64
    wasmstack.drop : i64

    // f32 select
    wasmstack.f32.const 1.0
    wasmstack.f32.const 2.0
    wasmstack.local.get 0 : i32
    wasmstack.select : f32
    wasmstack.drop : f32

    // f64 select
    wasmstack.f64.const 1.0
    wasmstack.f64.const 2.0
    wasmstack.local.get 0 : i32
    wasmstack.select : f64
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Global Variables
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @global_vars
wasmstack.module @global_vars {
  wasmstack.global @g_const : i32 {
    wasmstack.i32.const 42
  }

  wasmstack.global @g_mut : i32 mutable {
    wasmstack.i32.const 0
  }

  wasmstack.global @g_f64 : f64 mutable {
    wasmstack.f64.const 3.14159
  }

  // CHECK: wasmstack.func @use_globals
  wasmstack.func @use_globals : () -> i32 {
    // Read immutable global
    wasmstack.global.get @g_const : i32
    wasmstack.drop : i32

    // Read and write mutable global
    wasmstack.global.get @g_mut : i32
    wasmstack.i32.const 1
    wasmstack.add : i32
    wasmstack.global.set @g_mut : i32

    // Float global
    wasmstack.global.get @g_f64 : f64
    wasmstack.drop : f64

    wasmstack.global.get @g_mut : i32
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Complex Patterns from Stackification
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @stackification_patterns
wasmstack.module @stackification_patterns {

  // Pattern: Expression tree (optimal stackification)
  // CHECK: wasmstack.func @expression_tree
  wasmstack.func @expression_tree : (i32, i32, i32) -> i32 {
    // Computes: (a + b) * (a - c)
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    wasmstack.local.get 0 : i32
    wasmstack.local.get 2 : i32
    wasmstack.sub : i32
    wasmstack.mul : i32
    wasmstack.return
  }

  // Pattern: Multi-use value with tee
  // CHECK: wasmstack.func @tee_pattern
  wasmstack.func @tee_pattern : (i32, i32) -> i32 {
    wasmstack.local 2 : i32

    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    // Multi-use: tee saves to local, keeps value on stack
    wasmstack.local.tee 2 : i32
    wasmstack.local.get 2 : i32
    wasmstack.mul : i32
    wasmstack.return
  }

  // Pattern: Constant rematerialization
  // CHECK: wasmstack.func @const_remat
  wasmstack.func @const_remat : (i32) -> i32 {
    // Same constant used twice - rematerialized instead of using local
    wasmstack.local.get 0 : i32
    wasmstack.i32.const 10
    wasmstack.add : i32
    wasmstack.i32.const 10
    wasmstack.mul : i32
    wasmstack.return
  }

  // Pattern: Complex control flow with values
  // CHECK: wasmstack.func @complex_cf_values
  wasmstack.func @complex_cf_values : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.block @outer : ([i32, i32]) -> [i32] {
      // On stack: [a, b]
      wasmstack.local.get 0 : i32
      wasmstack.eqz : i32
      wasmstack.if : ([i32, i32]) -> [i32, i32] then {
        // If takes [a, b] as params, returns [a, b] unchanged
        // Just return the params as-is
      } else {
        // Else branch also returns [a, b] unchanged
      }
      // Stack: [a, b] - add them
      wasmstack.add : i32
    }
    wasmstack.return
  }

  // Pattern: Deeply nested expression in control flow
  // CHECK: wasmstack.func @nested_expr_in_cf
  wasmstack.func @nested_expr_in_cf : (i32, i32, i32) -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.local.get 0 : i32
      wasmstack.if : ([]) -> [] then {
        // Complex expression in then branch
        wasmstack.local.get 1 : i32
        wasmstack.local.get 2 : i32
        wasmstack.add : i32
        wasmstack.local.get 1 : i32
        wasmstack.mul : i32
        wasmstack.br @exit
      }
      wasmstack.local.get 2 : i32
    }
    wasmstack.return
  }
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @edge_cases
wasmstack.module @edge_cases {

  // Empty function
  // CHECK: wasmstack.func @empty_void
  wasmstack.func @empty_void : () -> () {
    wasmstack.return
  }

  // Single constant return
  // CHECK: wasmstack.func @const_return
  wasmstack.func @const_return : () -> i32 {
    wasmstack.i32.const 0
    wasmstack.return
  }

  // Identity function
  // CHECK: wasmstack.func @identity
  wasmstack.func @identity : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.return
  }

  // Multiple returns in function
  // CHECK: wasmstack.func @multi_return_paths
  wasmstack.func @multi_return_paths : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.eqz : i32
    wasmstack.if : ([]) -> [] then {
      wasmstack.i32.const 0
      wasmstack.return
    }
    wasmstack.local.get 0 : i32
    wasmstack.i32.const 1
    wasmstack.sub : i32
    wasmstack.return
  }

  // Block that consumes all params and produces different count
  // CHECK: wasmstack.func @block_different_arity
  wasmstack.func @block_different_arity : (i32, i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.local.get 2 : i32
    wasmstack.block @b : ([i32, i32, i32]) -> [i32] {
      // Consumes 3, produces 1
      wasmstack.add : i32
      wasmstack.add : i32
    }
    wasmstack.return
  }

  // Deeply nested blocks (stress test)
  // CHECK: wasmstack.func @deep_nesting
  wasmstack.func @deep_nesting : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.block @b1 : ([i32]) -> [i32] {
      wasmstack.block @b2 : ([i32]) -> [i32] {
        wasmstack.block @b3 : ([i32]) -> [i32] {
          wasmstack.block @b4 : ([i32]) -> [i32] {
            wasmstack.block @b5 : ([i32]) -> [i32] {
              wasmstack.i32.const 1
              wasmstack.add : i32
            }
          }
        }
      }
    }
    wasmstack.return
  }

  // If without else but unreachable then
  // CHECK: wasmstack.func @if_unreachable_then
  wasmstack.func @if_unreachable_then : (i32) -> () {
    wasmstack.local.get 0 : i32
    wasmstack.if : ([]) -> [] then {
      wasmstack.unreachable
    }
    wasmstack.return
  }

  // Nop operations
  // CHECK: wasmstack.func @with_nops
  wasmstack.func @with_nops : () -> i32 {
    wasmstack.nop
    wasmstack.nop
    wasmstack.i32.const 42
    wasmstack.nop
    wasmstack.return
  }
}
