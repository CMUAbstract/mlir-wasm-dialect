// RUN: wasm-opt %s -verify-wasmstack 2>&1 | FileCheck %s

// Tests for the verify-wasmstack pass - valid programs only
// Invalid programs should be tested with -verify-diagnostics in separate files

//===----------------------------------------------------------------------===//
// Valid Programs - Should Pass Verification
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.module @valid_programs
wasmstack.module @valid_programs {

  // CHECK: wasmstack.func @simple_add
  wasmstack.func @simple_add : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @void_function
  wasmstack.func @void_function : () -> () {
    wasmstack.return
  }

  // CHECK: wasmstack.func @arithmetic_chain
  wasmstack.func @arithmetic_chain : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    wasmstack.local.get 0 : i32
    wasmstack.mul : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @block_with_result
  wasmstack.func @block_with_result : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.block @b0 : ([i32]) -> [i32] {
      wasmstack.i32.const 10
      wasmstack.add : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @simple_loop
  wasmstack.func @simple_loop : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.loop @loop : ([i32]) -> [i32] {
      wasmstack.i32.const 1
      wasmstack.sub : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @if_else
  wasmstack.func @if_else : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.i32.const 1
    } else {
      wasmstack.i32.const 0
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @unconditional_branch
  wasmstack.func @unconditional_branch : (i32) -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.local.get 0 : i32
      wasmstack.br @exit
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @conditional_branch_pattern
  // WebAssembly br_if: needs [result, condition] on stack
  wasmstack.func @conditional_branch_pattern : (i32) -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      // Push result value first
      wasmstack.i32.const 42
      // Then push condition
      wasmstack.local.get 0 : i32
      wasmstack.br_if @exit
      // If not taken, drop the result and compute new one
      wasmstack.drop : i32
      wasmstack.i32.const 0
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @polymorphic_after_unreachable
  wasmstack.func @polymorphic_after_unreachable : () -> i32 {
    wasmstack.unreachable
    // After unreachable, stack is polymorphic - this is valid
    wasmstack.add : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @polymorphic_after_br
  wasmstack.func @polymorphic_after_br : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.i32.const 42
      wasmstack.br @exit
      // After unconditional br, stack is polymorphic
      wasmstack.add : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @drop_operation
  wasmstack.func @drop_operation : (i32) -> () {
    wasmstack.local.get 0 : i32
    wasmstack.drop : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @nested_blocks
  wasmstack.func @nested_blocks : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.block @outer : ([i32]) -> [i32] {
      wasmstack.block @inner : ([i32]) -> [i32] {
        wasmstack.i32.const 1
        wasmstack.add : i32
      }
      wasmstack.i32.const 2
      wasmstack.add : i32
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @loop_with_br_continue
  wasmstack.func @loop_with_br_continue : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.set 0 : i32
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.loop @continue : ([]) -> [] {
        wasmstack.local.get 0 : i32
        wasmstack.i32.const 1
        wasmstack.sub : i32
        wasmstack.local.tee 0 : i32
        wasmstack.eqz : i32
        wasmstack.if : ([]) -> [] then {
          wasmstack.local.get 0 : i32
          wasmstack.br @exit
        }
        wasmstack.br @continue
      }
      wasmstack.i32.const 0
    }
    wasmstack.return
  }

  // CHECK: wasmstack.func @comparison_ops
  wasmstack.func @comparison_ops : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.lt_s : i32
    wasmstack.return
  }

  // CHECK: wasmstack.func @float_operations
  wasmstack.func @float_operations : (f32, f32) -> f32 {
    wasmstack.local.get 0 : f32
    wasmstack.local.get 1 : f32
    wasmstack.add : f32
    wasmstack.return
  }
}
