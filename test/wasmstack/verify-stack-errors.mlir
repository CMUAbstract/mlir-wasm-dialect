// RUN: wasm-opt %s -split-input-file -verify-wasmstack -verify-diagnostics

// Tests for verify-wasmstack pass error detection

//===----------------------------------------------------------------------===//
// Stack Underflow Errors
//===----------------------------------------------------------------------===//

wasmstack.module @stack_underflow {
  wasmstack.func @underflow : () -> i32 {
    // expected-error @+1 {{stack underflow}}
    wasmstack.add : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @insufficient_operands {
  wasmstack.func @one_operand : () -> i32 {
    wasmstack.i32.const 10
    // expected-error @+1 {{stack underflow}}
    wasmstack.add : i32
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Type Mismatch Errors
//===----------------------------------------------------------------------===//

wasmstack.module @type_mismatch {
  wasmstack.func @wrong_type : () -> i32 {
    wasmstack.f32.const 1.0
    wasmstack.i32.const 10
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.add : i32
    wasmstack.return
  }
}

// -----

wasmstack.module @return_type_mismatch {
  wasmstack.func @wrong_return : () -> i32 {
    wasmstack.f32.const 1.0
    // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Block/Loop Stack Balance Errors
//===----------------------------------------------------------------------===//

wasmstack.module @unbalanced_block {
  wasmstack.func @extra_value : () -> i32 {
    // expected-error @below {{stack height mismatch}}
    wasmstack.block @b0 : ([]) -> [i32] {
      wasmstack.i32.const 10
      wasmstack.i32.const 20
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @missing_result {
  wasmstack.func @no_result : () -> i32 {
    // expected-error @below {{stack height mismatch}}
    wasmstack.block @b0 : ([]) -> [i32] {
      // Block expects to produce i32 but produces nothing
    }
    wasmstack.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Branch Target Errors
//===----------------------------------------------------------------------===//

wasmstack.module @invalid_branch {
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

wasmstack.module @branch_type_mismatch {
  wasmstack.func @wrong_branch_type : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      wasmstack.f32.const 1.0
      // expected-error @+1 {{type mismatch: expected 'i32' but got 'f32'}}
      wasmstack.br @exit
    }
    wasmstack.return
  }
}

// -----

wasmstack.module @br_if_missing_values {
  wasmstack.func @need_result : () -> i32 {
    wasmstack.block @exit : ([]) -> [i32] {
      // Only have condition, no result value
      wasmstack.i32.const 1
      // expected-error @+1 {{insufficient values for conditional branch}}
      wasmstack.br_if @exit
      wasmstack.i32.const 0
    }
    wasmstack.return
  }
}
