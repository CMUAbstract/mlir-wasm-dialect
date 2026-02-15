// RUN: wasm-opt %s --convert-to-wasmstack 2>&1 | FileCheck %s

// Test deep nesting with multiple exit levels to verify label resolution.
// This verifies that getLabelForExitLevel correctly resolves exit levels
// to the appropriate enclosing block/loop labels.


module {
  //===--------------------------------------------------------------------===//
  // Test 1: Triple nested blocks with different exit levels
  //===--------------------------------------------------------------------===//

  // Creates nested blocks: outer block -> middle block -> inner block
  // Tests that branch_if correctly resolves exit levels:
  // - Exit level 0 = innermost (loop)
  // - Exit level 1 = next outer (middle block)
  // - Exit level 2 = outermost (outer block)
  //
  // CHECK-LABEL: wasmstack.func @triple_nested
  // CHECK: wasmstack.block @[[OUTER:block_[0-9]+]]
  // CHECK: wasmstack.block @[[MIDDLE:block_[0-9]+]]
  // CHECK: wasmstack.loop @[[INNER:loop_[0-9]+]]
  // Exit level 0 should go to inner loop
  // CHECK: wasmstack.br_if @[[INNER]]
  // Exit level 1 should go to middle block
  // CHECK: wasmstack.br_if @[[MIDDLE]]
  // Exit level 2 should go to outer block
  // CHECK: wasmstack.br_if @[[OUTER]]
  wasmssa.func @triple_nested() -> i32 {
    %c0 = wasmssa.const 0 : i32
    %c1 = wasmssa.const 1 : i32
    %result = wasmssa.const 42 : i32

    // Outer block
    wasmssa.block(%c0) : i32 : {
    ^outer_entry(%outer_arg: i32):
      // Middle block
      wasmssa.block(%outer_arg) : i32 : {
      ^middle_entry(%middle_arg: i32):
        // Inner loop
        wasmssa.loop(%middle_arg) : i32 : {
        ^inner_entry(%inner_arg: i32):
          // Exit level 0 -> inner loop (continue)
          wasmssa.branch_if %c1 to level 0 with args(%inner_arg : i32) else ^test1

        ^test1:
          // Exit level 1 -> middle block (exit to middle)
          wasmssa.branch_if %c1 to level 1 with args(%inner_arg : i32) else ^test2

        ^test2:
          // Exit level 2 -> outer block (exit to outer)
          wasmssa.branch_if %c1 to level 2 with args(%inner_arg : i32) else ^continue

        ^continue:
          wasmssa.block_return %inner_arg : i32
        }> ^loop_exit

      ^loop_exit(%loop_result: i32):
        wasmssa.block_return %loop_result : i32
      }> ^middle_exit

    ^middle_exit(%middle_result: i32):
      wasmssa.block_return %middle_result : i32
    }> ^outer_exit

  ^outer_exit(%outer_result: i32):
    wasmssa.return %outer_result : i32
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Mixed block and loop nesting
  //===--------------------------------------------------------------------===//

  // Tests exit level resolution with mixed block/loop:
  // block -> loop -> block
  //
  // CHECK-LABEL: wasmstack.func @mixed_nesting
  // CHECK: wasmstack.block @[[OUTER2:block_[0-9]+]]
  // CHECK: wasmstack.loop @[[LOOP2:loop_[0-9]+]]
  // CHECK: wasmstack.block @[[INNER2:block_[0-9]+]]
  wasmssa.func @mixed_nesting() -> i32 {
    %c0 = wasmssa.const 0 : i32
    %c1 = wasmssa.const 1 : i32

    // Outer block
    wasmssa.block(%c0) : i32 : {
    ^bb0(%arg0: i32):
      // Loop
      wasmssa.loop(%arg0) : i32 : {
      ^bb1(%arg1: i32):
        // Inner block
        wasmssa.block(%arg1) : i32 : {
        ^bb2(%arg2: i32):
          // Exit level 0 -> inner block
          wasmssa.branch_if %c1 to level 0 with args(%arg2 : i32) else ^cont
        ^cont:
          wasmssa.block_return %arg2 : i32
        }> ^after_inner

      ^after_inner(%inner_result: i32):
        // Exit level 0 from here -> loop (continue)
        // Exit level 1 -> outer block
        wasmssa.branch_if %c1 to level 1 with args(%inner_result : i32) else ^loop_cont

      ^loop_cont:
        wasmssa.block_return %inner_result : i32
      }> ^loop_exit

    ^loop_exit(%loop_result: i32):
      wasmssa.block_return %loop_result : i32
    }> ^func_exit

  ^func_exit(%final: i32):
    wasmssa.return %final : i32
  }
}
