// RUN: wasm-opt %s -split-input-file -verify-wasmstack -verify-diagnostics

// Comprehensive error detection tests for verify-wasmstack pass
// Each test case is separated by ----- and tests a specific error condition

//===----------------------------------------------------------------------===//
// Stack Underflow Errors
//===----------------------------------------------------------------------===//

wasmstack.module @underflow_binary_op {
  wasmstack.func @empty_stack_binary : () -> i32 {
    // expected-error @+1 {{stack underflow}}
    wasmstack.add : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_one_operand {
  wasmstack.func @one_operand_binary : () -> i32 {
    wasmstack.i32.const 10
    // expected-error @+1 {{stack underflow}}
    wasmstack.add : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_unary {
  wasmstack.func @empty_stack_unary : () -> i32 {
    // expected-error @+1 {{stack underflow}}
    wasmstack.clz : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_comparison {
  wasmstack.func @one_operand_compare : () -> i32 {
    wasmstack.i32.const 5
    // expected-error @+1 {{stack underflow}}
    wasmstack.lt_s : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_store {
  wasmstack.memory @mem min = 1
  wasmstack.func @store_missing_value : () -> () {
    wasmstack.i32.const 0  // address only
    // expected-error @+1 {{stack underflow}}
    wasmstack.i32.store offset = 0 align = 4 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_store_missing_both {
  wasmstack.memory @mem min = 1
  wasmstack.func @store_empty : () -> () {
    // expected-error @+1 {{stack underflow}}
    wasmstack.i32.store offset = 0 align = 4 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_load {
  wasmstack.memory @mem min = 1
  wasmstack.func @load_no_address : () -> i32 {
    // expected-error @+1 {{stack underflow}}
    wasmstack.i32.load offset = 0 align = 4 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_select {
  wasmstack.func @select_missing_ops : () -> i32 {
    wasmstack.i32.const 1
    wasmstack.i32.const 0
    // Only 2 operands, need 3
    // expected-error @+1 {{stack underflow}}
    wasmstack.select : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_drop {
  wasmstack.func @drop_empty : () -> () {
    // expected-error @+1 {{stack underflow}}
    wasmstack.drop : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_local_set {
  wasmstack.func @local_set_empty : () -> () {
    wasmstack.local 0 : i32
    // expected-error @+1 {{stack underflow}}
    wasmstack.local.set 0 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_local_tee {
  wasmstack.func @local_tee_empty : () -> i32 {
    wasmstack.local 0 : i32
    // expected-error @+1 {{stack underflow}}
    wasmstack.local.tee 0 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_conversion {
  wasmstack.func @convert_empty : () -> i64 {
    // expected-error @+1 {{stack underflow}}
    wasmstack.i64.extend_i32_s : i32 -> i64
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_return {
  wasmstack.func @return_missing_value : () -> i32 {
    // expected-error @+1 {{stack underflow}}
    wasmstack.return
  }
}

// -----

wasmstack.module @underflow_call {
  wasmstack.func @callee : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.return
  }
  wasmstack.func @caller_missing_args : () -> i32 {
    wasmstack.i32.const 1
    // Only 1 arg, need 2
    // expected-error @+1 {{stack underflow}}
    wasmstack.call @callee : (i32, i32) -> i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Type Mismatch Errors
//===----------------------------------------------------------------------===//

wasmstack.module @type_mismatch_binary {
  wasmstack.func @wrong_type_binary : () -> i32 {
    wasmstack.f32.const 1.0
    wasmstack.i32.const 10
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.add : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_both_operands {
  wasmstack.func @both_wrong : () -> i32 {
    wasmstack.f64.const 1.0
    wasmstack.f32.const 2.0
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.add : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_unary {
  wasmstack.func @wrong_unary : () -> i32 {
    wasmstack.i64.const 100
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.clz : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_return {
  wasmstack.func @wrong_return_type : () -> i32 {
    wasmstack.f32.const 1.0
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_multi_return {
  wasmstack.func @wrong_second_return : () -> (i32, i64) {
    wasmstack.i32.const 1
    wasmstack.i32.const 2  // Should be i64
    // expected-error @+1 {{type mismatch: expected 'i64' but got 'i32'}}
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_local_set {
  wasmstack.func @wrong_local_type : () -> () {
    wasmstack.local 0 : i32
    wasmstack.i64.const 100
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.local.set 0 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_store_value {
  wasmstack.memory @mem min = 1
  wasmstack.func @wrong_store_value : () -> () {
    wasmstack.i32.const 0  // address
    wasmstack.f32.const 1.0  // wrong type for i32 store
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.i32.store offset = 0 align = 4 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_store_address {
  wasmstack.memory @mem min = 1
  wasmstack.func @wrong_store_addr : () -> () {
    wasmstack.i64.const 0  // wrong address type (should be i32)
    wasmstack.i32.const 42
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.i32.store offset = 0 align = 4 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_load_address {
  wasmstack.memory @mem min = 1
  wasmstack.func @wrong_load_addr : () -> i32 {
    wasmstack.f32.const 0.0  // wrong address type
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.i32.load offset = 0 align = 4 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_conversion {
  wasmstack.func @wrong_conversion_input : () -> i64 {
    wasmstack.i64.const 100  // i64 instead of i32
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.i64.extend_i32_s : i32 -> i64
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_select_operands {
  wasmstack.func @mixed_select : () -> i32 {
    wasmstack.i32.const 1
    wasmstack.i64.const 2  // Wrong type
    wasmstack.i32.const 0  // condition
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.select : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_select_condition {
  wasmstack.func @wrong_condition : () -> i32 {
    wasmstack.i32.const 1
    wasmstack.i32.const 2
    wasmstack.f32.const 1.0  // condition should be i32
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.select : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_drop {
  wasmstack.func @wrong_drop : () -> () {
    wasmstack.i64.const 100
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.drop : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @type_mismatch_call {
  wasmstack.func @callee : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.return
  }
  wasmstack.func @caller_wrong_arg : () -> i32 {
    wasmstack.f64.const 1.0  // wrong type
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f64'}}
    wasmstack.call @callee : (i32) -> i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Block/Loop Stack Balance Errors
//===----------------------------------------------------------------------===//

wasmstack.module @unbalanced_block_extra {
  wasmstack.func @extra_values : () -> i32 {
    // expected-error @below {{stack height mismatch}}
    wasmstack.block @b0 : ([]) -> [i32] {
      wasmstack.i32.const 10
      wasmstack.i32.const 20
      // Two values but block produces one
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @unbalanced_block_missing {
  wasmstack.func @missing_result : () -> i32 {
    // expected-error @below {{stack height mismatch}}
    wasmstack.block @b0 : ([]) -> [i32] {
      // No values but block should produce i32
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @unbalanced_block_wrong_count {
  wasmstack.func @wrong_result_count : () -> i32 {
    // expected-error @below {{stack height mismatch}}
    wasmstack.block @b0 : ([]) -> [i32, i32] {
      wasmstack.i32.const 10
      // Should produce 2 i32s but only produces 1
    }
    wasmstack.add : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @unbalanced_loop {
  wasmstack.func @loop_wrong_result : () -> i32 {
    // expected-error @below {{stack height mismatch}}
    wasmstack.loop @l0 : ([]) -> [i32] {
      // Loop should produce i32 but produces nothing
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @block_result_type_mismatch {
  wasmstack.func @wrong_block_result : () -> i32 {
    // expected-error @below {{frame result type mismatch}}
    wasmstack.block @b0 : ([]) -> [i32] {
      wasmstack.f32.const 1.0
      // Produces f32 but block expects i32
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @block_param_underflow {
  wasmstack.func @not_enough_params : () -> i32 {
    wasmstack.i32.const 10
    // Block wants 2 params but only 1 on stack
    // expected-error @+1 {{stack underflow}}
    wasmstack.block @b0 : ([i32, i32]) -> [i32] {
      wasmstack.add : i32
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @block_param_type_mismatch {
  wasmstack.func @wrong_param_types : () -> i32 {
    wasmstack.f32.const 1.0
    wasmstack.i32.const 10
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.block @b0 : ([i32, i32]) -> [i32] {
      wasmstack.add : i32
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Branch Target Errors
//===----------------------------------------------------------------------===//

wasmstack.module @invalid_branch_target {
  wasmstack.func @bad_target : () -> i32 {
    wasmstack.block @b0 : ([]) -> [i32] {
      wasmstack.i32.const 42
      // expected-error @+1 {{branch target 'nonexistent' not found}}
      wasmstack.br @nonexistent
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @invalid_br_if_target {
  wasmstack.func @bad_br_if : () -> i32 {
    wasmstack.block @b0 : ([]) -> [i32] {
      wasmstack.i32.const 42
      wasmstack.i32.const 1
      // expected-error @+1 {{branch target 'missing' not found}}
      wasmstack.br_if @missing
      wasmstack.drop : i32
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @branch_type_mismatch {
  wasmstack.func @wrong_branch_type : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.f32.const 1.0
      // Block expects i32 but stack has f32
      // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
      wasmstack.br @exit
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @branch_missing_values {
  wasmstack.func @br_no_values : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      // expected-error @+1 {{stack underflow}}
      wasmstack.br @exit
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @br_if_wrong_condition {
  wasmstack.func @br_if_bad_cond : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.i32.const 42
      wasmstack.f32.const 1.0  // condition should be i32
      // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
      wasmstack.br_if @exit
      wasmstack.drop : i32
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @br_if_insufficient_values {
  wasmstack.func @need_result : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      // Only condition, no result value for branch
      wasmstack.i32.const 1
      // expected-error @+1 {{insufficient values for conditional branch}}
      wasmstack.br_if @exit
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @br_if_type_mismatch {
  wasmstack.func @br_if_wrong_type : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.f64.const 1.0  // Wrong type for block result
      wasmstack.i32.const 1    // condition
      // expected-error @+1 {{conditional branch type mismatch}}
      wasmstack.br_if @exit
      wasmstack.drop : f64
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @br_table_invalid_target {
  wasmstack.func @br_table_bad : () -> i32 {
    wasmstack.block @valid : ([]) -> [i32] {
      wasmstack.i32.const 42
      wasmstack.i32.const 0
      // expected-error @+1 {{branch target 'invalid' not found}}
      wasmstack.br_table [@invalid] default @valid
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @br_table_invalid_default {
  wasmstack.func @br_table_bad_default : () -> i32 {
    wasmstack.block @valid : ([]) -> [i32] {
      wasmstack.i32.const 42
      wasmstack.i32.const 0
      // expected-error @+1 {{default branch target 'missing' not found}}
      wasmstack.br_table [@valid] default @missing
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @br_table_inconsistent_arity {
  wasmstack.func @br_table_arity : (i32) -> i32 {
    wasmstack.block @one_result : ([]) -> [i32] {
      wasmstack.block @no_result : ([]) -> [] {
        wasmstack.i32.const 42
        wasmstack.local.get 0 : i32
        // expected-error @+1 {{branch table targets have inconsistent arities}}
        wasmstack.br_table [@no_result] default @one_result
      }
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// If/Else Errors
//===----------------------------------------------------------------------===//

wasmstack.module @if_wrong_condition {
  wasmstack.func @if_bad_cond : () -> i32 {
    wasmstack.f32.const 1.0  // condition should be i32
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.i32.const 1
    } else {
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @if_then_wrong_result {
  wasmstack.func @then_wrong : () -> i32 {
    wasmstack.i32.const 1
    // expected-error @below {{frame result type mismatch}}
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.f32.const 1.0  // Wrong type
    } else {
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @if_else_wrong_result {
  wasmstack.func @else_wrong : () -> i32 {
    wasmstack.i32.const 1
    // expected-error @+1 {{frame result type mismatch}}
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.i32.const 1
    } else {
      wasmstack.f64.const 1.0  // Wrong type
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @if_missing_else_with_result {
  wasmstack.func @no_else_result : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    // expected-error @+1 {{if without else cannot produce results}}
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.i32.const 1
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @if_param_type_mismatch {
  wasmstack.func @wrong_if_param : () -> i32 {
    wasmstack.f32.const 1.0  // Param should be i32
    wasmstack.i32.const 1    // condition
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.if : ([i32]) -> [i32] then {
      wasmstack.i32.const 1
      wasmstack.add : i32
    } else {
      wasmstack.i32.const 1
      wasmstack.sub : i32
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @if_then_unbalanced {
  wasmstack.func @then_extra : () -> i32 {
    wasmstack.i32.const 1
    // expected-error @below {{stack height mismatch}}
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.i32.const 1
      wasmstack.i32.const 2
      // Two values but should produce one
    } else {
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Return Errors
//===----------------------------------------------------------------------===//

// Note: Having extra values on the stack at return is allowed in WebAssembly
// (extra values are discarded), so @return_extra_values is actually valid.

wasmstack.module @return_missing_multi {
  wasmstack.func @need_two : () -> (i32, i64) {
    wasmstack.i32.const 1
    // expected-error @+1 {{type mismatch: expected 'i64' but got 'i32'}}
    wasmstack.return
  }
}

// -----

wasmstack.module @return_wrong_order {
  wasmstack.func @swapped_types : () -> (i32, i64) {
    wasmstack.i64.const 1
    wasmstack.i32.const 2
    // expected-error @+1 {{type mismatch: expected 'i64' but got 'i32'}}
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Global Variable Errors
//===----------------------------------------------------------------------===//

wasmstack.module @global_set_type_mismatch {
  wasmstack.global @g : i32 mutable {
    wasmstack.i32.const 0
  }
  wasmstack.func @wrong_global_type : () -> () {
    wasmstack.f32.const 1.0
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.global.set @g : i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Memory Grow Errors
//===----------------------------------------------------------------------===//

wasmstack.module @memory_grow_wrong_type {
  wasmstack.memory @mem min = 1
  wasmstack.func @grow_bad_type : () -> i32 {
    wasmstack.i64.const 1  // Should be i32
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.memory.grow
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Complex Error Scenarios
//===----------------------------------------------------------------------===//

wasmstack.module @nested_error_propagation {
  wasmstack.func @error_in_nested : () -> i32 {
    wasmstack.block @outer : ([]) -> [i32] {
      // expected-error @+1 {{frame result type mismatch}}
      wasmstack.block @inner : ([]) -> [i32] {
        wasmstack.f32.const 1.0  // Wrong type - block expects i32
      }
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @cross_frame_underflow {
  wasmstack.func @cant_pop_across_frame : () -> i32 {
    wasmstack.i32.const 100  // Value before block
    wasmstack.block @b : ([]) -> [i32] {
      // Can't access the 100 from parent frame
      // expected-error @+1 {{stack underflow}}
      wasmstack.add : i32
    }
    wasmstack.return
  }
}
