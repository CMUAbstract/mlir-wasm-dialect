// RUN: wasm-opt %s -split-input-file -verify-wasmstack -verify-diagnostics

// Extended error detection tests for verify-wasmstack pass
// Tests additional edge cases that could expose verification bugs

//===----------------------------------------------------------------------===//
// Block Parameter Errors
//===----------------------------------------------------------------------===//

wasmstack.module @block_param_underflow {
  wasmstack.func @insufficient_params : () -> i32 {
    // Block expects 2 params but only 1 provided
    wasmstack.i32.const 1
    // expected-error @+1 {{stack underflow}}
    wasmstack.block @b0 : ([i32, i32]) -> [i32] {
      wasmstack.add : i32
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @block_param_type_mismatch {
  wasmstack.func @wrong_param_type : () -> i32 {
    wasmstack.f32.const 1.0
    wasmstack.i32.const 2
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.block @b0 : ([i32, i32]) -> [i32] {
      wasmstack.add : i32
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Loop Parameter Errors
//===----------------------------------------------------------------------===//

wasmstack.module @loop_param_underflow {
  wasmstack.func @loop_needs_params : () -> i32 {
    // Loop expects i32 param but none provided
    // expected-error @+1 {{stack underflow}}
    wasmstack.loop @loop : ([i32]) -> [i32] {
      wasmstack.i32.const 1
      wasmstack.add : i32
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @loop_param_type_wrong {
  wasmstack.func @loop_wrong_type : () -> i32 {
    wasmstack.i64.const 1
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.loop @loop : ([i32]) -> [i32] {
      wasmstack.i32.const 1
      wasmstack.add : i32
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// If/Else Parameter Errors
//===----------------------------------------------------------------------===//

wasmstack.module @if_param_underflow {
  wasmstack.func @if_needs_params : (i32) -> i32 {
    // If expects i32 param
    wasmstack.local.get 0 : i32  // condition
    // expected-error @+1 {{stack underflow}}
    wasmstack.if : ([i32]) -> [i32] then {
      wasmstack.i32.const 1
      wasmstack.add : i32
    } else {
      wasmstack.i32.const 2
      wasmstack.add : i32
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @if_branch_result_mismatch {
  wasmstack.func @then_wrong_type : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    // expected-error @below {{type mismatch}}
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.f32.const 1.0  // wrong type
    } else {
      wasmstack.i32.const 2
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @if_else_arity_mismatch {
  wasmstack.func @different_result_counts : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    // expected-error @below {{stack height mismatch}}
    wasmstack.if : ([]) -> [i32] then {
      wasmstack.i32.const 1
      wasmstack.i32.const 2  // extra value
    } else {
      wasmstack.i32.const 3
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Conditional Branch Errors
//===----------------------------------------------------------------------===//

wasmstack.module @br_if_no_condition {
  wasmstack.func @missing_condition : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.i32.const 42  // value for branch
      // expected-error @+1 {{insufficient values for conditional branch: need 1 but only have 0}}
      wasmstack.br_if @exit
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @br_if_wrong_value_type {
  wasmstack.func @value_type_error : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.f32.const 1.0  // wrong type for i32 block result
      wasmstack.i32.const 1    // condition
      // expected-error @+1 {{conditional branch type mismatch at index 0: expected 'i32' but got 'f32'}}
      wasmstack.br_if @exit
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Function Return Type Errors
//===----------------------------------------------------------------------===//

wasmstack.module @return_underflow {
  wasmstack.func @missing_return_value : () -> i32 {
    // Function expects i32 return but nothing on stack
    // expected-error @+1 {{stack underflow}}
    wasmstack.return
  }
}

// -----

// Per WebAssembly spec: extra values before unconditional branch (return)
// are valid - they get discarded when the branch is taken.
wasmstack.module @return_extra_values_valid {
  wasmstack.func @extra_values_before_return : () -> i32 {
    wasmstack.i32.const 1
    wasmstack.i32.const 2  // extra value - valid, discarded by return
    wasmstack.return
  }
}

// -----

wasmstack.module @return_wrong_type {
  wasmstack.func @wrong_return_type : () -> i32 {
    wasmstack.i64.const 42  // i64 instead of i32
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.return
  }
}

// -----

wasmstack.module @multi_return_partial {
  wasmstack.func @partial_return : () -> (i32, i32) {
    wasmstack.i32.const 1
    // Only one value but two expected
    // expected-error @+1 {{stack underflow}}
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Memory Operation Errors
//===----------------------------------------------------------------------===//

wasmstack.module @load_no_address {
  wasmstack.memory @mem min = 1

  wasmstack.func @load_underflow : () -> i32 {
    // Load needs address but stack is empty
    // expected-error @+1 {{stack underflow}}
    wasmstack.i32.load offset = 0 align = 4 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @store_missing_value {
  wasmstack.memory @mem min = 1

  wasmstack.func @store_underflow : () -> () {
    wasmstack.i32.const 0  // address only
    // expected-error @+1 {{stack underflow}}
    wasmstack.i32.store offset = 0 align = 4 : i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Local Variable Errors
//===----------------------------------------------------------------------===//

wasmstack.module @local_set_underflow {
  wasmstack.func @set_nothing : () -> () {
    wasmstack.local 0 : i32
    // Nothing on stack to set
    // expected-error @+1 {{stack underflow}}
    wasmstack.local.set 0 : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @local_tee_underflow {
  wasmstack.func @tee_nothing : () -> i32 {
    wasmstack.local 0 : i32
    // Nothing on stack to tee
    // expected-error @+1 {{stack underflow}}
    wasmstack.local.tee 0 : i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Select Operation Errors
//===----------------------------------------------------------------------===//

wasmstack.module @select_underflow {
  wasmstack.func @select_missing_ops : () -> i32 {
    wasmstack.i32.const 1  // only one value
    wasmstack.i32.const 0  // condition
    // expected-error @+1 {{stack underflow}}
    wasmstack.select : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @select_type_mismatch {
  wasmstack.func @select_different_types : () -> i32 {
    wasmstack.i32.const 1
    wasmstack.i64.const 2  // different type
    wasmstack.i32.const 1  // condition
    // expected-error @+1 {{type mismatch}}
    wasmstack.select : i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Drop Operation Errors
//===----------------------------------------------------------------------===//

wasmstack.module @drop_underflow {
  wasmstack.func @drop_empty : () -> () {
    // Nothing to drop
    // expected-error @+1 {{stack underflow}}
    wasmstack.drop : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @drop_type_mismatch {
  wasmstack.func @drop_wrong_type : () -> () {
    wasmstack.i64.const 42
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'i64'}}
    wasmstack.drop : i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Unary Operation Errors
//===----------------------------------------------------------------------===//

wasmstack.module @unary_underflow {
  wasmstack.func @clz_empty : () -> i32 {
    // expected-error @+1 {{stack underflow}}
    wasmstack.clz : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @unary_type_error {
  wasmstack.func @clz_wrong_type : () -> i32 {
    wasmstack.f32.const 1.0
    // expected-error @+1 {{type mismatch}}
    wasmstack.clz : i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Conversion Operation Errors
//===----------------------------------------------------------------------===//

wasmstack.module @conversion_underflow {
  wasmstack.func @extend_empty : () -> i64 {
    // expected-error @+1 {{stack underflow}}
    wasmstack.i64.extend_i32_s : i32 -> i64
    wasmstack.return
  }
}

// -----

wasmstack.module @conversion_wrong_input {
  wasmstack.func @extend_wrong_type : () -> i64 {
    wasmstack.i64.const 1  // wrong input type
    // expected-error @+1 {{type mismatch}}
    wasmstack.i64.extend_i32_s : i32 -> i64
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Nested Block Errors
//===----------------------------------------------------------------------===//

wasmstack.module @nested_inner_error {
  wasmstack.func @inner_block_underflow : () -> i32 {
    wasmstack.block @outer : ([]) -> [i32] {
      // expected-error @below {{stack height mismatch at frame exit}}
      wasmstack.block @inner : ([]) -> [i32] {
        // No value produced
      }
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @nested_branch_to_wrong_level {
  wasmstack.func @branch_type_error : () -> i32 {
    wasmstack.block @outer : ([]) -> [i32] {
      wasmstack.block @inner : ([]) -> [] {
        wasmstack.f32.const 1.0  // wrong type for outer block
        // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
        wasmstack.br @outer
      }
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Call Errors
//===----------------------------------------------------------------------===//

wasmstack.module @call_underflow {
  wasmstack.func @callee : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    wasmstack.return
  }

  wasmstack.func @caller_missing_args : () -> i32 {
    wasmstack.i32.const 1  // only one arg
    // expected-error @+1 {{stack underflow}}
    wasmstack.call @callee : (i32, i32) -> i32
    wasmstack.return
  }
}

// -----

wasmstack.module @call_type_mismatch {
  wasmstack.func @callee : (i32, i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.local.get 1 : i32
    wasmstack.add : i32
    wasmstack.return
  }

  wasmstack.func @caller_wrong_type : () -> i32 {
    wasmstack.i64.const 1  // wrong type
    wasmstack.i32.const 2
    // expected-error @+1 {{type mismatch}}
    wasmstack.call @callee : (i32, i32) -> i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Global Operation Errors
//===----------------------------------------------------------------------===//

wasmstack.module @global_set_underflow {
  wasmstack.global @g : i32 mutable {
    wasmstack.i32.const 0
  }

  wasmstack.func @set_empty : () -> () {
    // expected-error @+1 {{stack underflow}}
    wasmstack.global.set @g : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @global_set_type_error {
  wasmstack.global @g : i32 mutable {
    wasmstack.i32.const 0
  }

  wasmstack.func @set_wrong_type : () -> () {
    wasmstack.i64.const 1
    // expected-error @+1 {{type mismatch}}
    wasmstack.global.set @g : i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// br_table Errors
//===----------------------------------------------------------------------===//

wasmstack.module @br_table_no_index {
  wasmstack.func @missing_index : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      // No index on stack
      // expected-error @+1 {{stack underflow}}
      wasmstack.br_table [@exit] default @exit
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Complex Control Flow Errors
//===----------------------------------------------------------------------===//

// Per WebAssembly spec: extra values before unconditional branch (br)
// are valid - they get discarded when the branch is taken.
wasmstack.module @loop_exit_extra_values_valid {
  wasmstack.func @extra_values_before_br : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.loop @loop : ([]) -> [] {
        wasmstack.i32.const 1
        wasmstack.i32.const 2  // extra value - valid, discarded by br
        wasmstack.br @exit
      }
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}
