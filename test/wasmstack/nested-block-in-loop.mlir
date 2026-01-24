// RUN: wasm-opt %s --convert-to-wasmstack 2>&1 | FileCheck %s

// Test for bug: BlockReturnOp inside a multi-block loop region should
// branch to the loop label, but when there's a nested wasmssa.block operation
// in the CFG, labelStack.back() returns the inner block's label instead.
//
// Bug location: lib/wasmstack/WasmStackPasses.cpp lines 934-937
// When block_return is in a loop body that contains nested wasmssa.block ops,
// labelStack.back() may return the wrong label.
//
// This test demonstrates the scenario where:
// 1. A loop contains multiple basic blocks (^loop_start, ^has_nested_block, ^after_nested)
// 2. One of those blocks (^has_nested_block) contains a nested wasmssa.block operation
// 3. After the nested wasmssa.block completes, the next block (^after_nested) has a block_return
// 4. That block_return should branch to the loop label to continue iteration
//
// If labelStack is not properly managed (i.e., if the nested block's label isn't popped),
// then labelStack.back() would return @block_2 instead of @loop_1, causing incorrect behavior.

// CHECK: ConvertToWasmStack pass running on module

module {
  // Test case: Loop with multi-block CFG where one block contains a nested
  // wasmssa.block operation. The block_return after the nested block should
  // continue the loop, but labelStack.back() may have been corrupted by the
  // nested block's label being pushed.
  //
  // CHECK-LABEL: wasmstack.func @nested_block_in_loop
  // Verify the structure: outer block containing loop containing nested block
  // Block params get saved to locals, then loaded for the loop entry
  // CHECK: wasmstack.block @[[OUTER_BLOCK:block_[0-9]+]]
  // CHECK: wasmstack.local.set
  // CHECK: wasmstack.local.get
  // CHECK: wasmstack.loop @[[LOOP:loop_[0-9]+]]
  // CHECK: wasmstack.block @[[INNER_BLOCK:block_[0-9]+]]
  // After the inner block completes, the next br should target the loop, not any block
  // CHECK: wasmstack.br @[[LOOP]]
  // CHECK-NOT: wasmstack.br @[[INNER_BLOCK]]
  // CHECK-NOT: wasmstack.br @[[OUTER_BLOCK]]
  wasmssa.func @nested_block_in_loop() -> i32 {
    %n = wasmssa.const 10 : i32
    %c0 = wasmssa.const 0 : i32
    %c1 = wasmssa.const 1 : i32

    // Outer block to catch loop exit
    wasmssa.block(%c0) : i32 : {
    ^bb0(%init: i32):
      // Loop with multi-block body
      wasmssa.loop(%init) : i32 : {
      ^loop_start(%counter: i32):
        // Check exit condition
        %should_exit = wasmssa.ge_si %counter %n : i32 -> i32
        wasmssa.branch_if %should_exit to level 1 else ^has_nested_block

      ^has_nested_block:
        // This block contains a nested wasmssa.block operation
        // The nested block is a structured control flow operation
        wasmssa.block(%counter) : i32 : {
        ^inner:
          %incremented = wasmssa.add %counter %c1 : i32
          wasmssa.block_return %incremented : i32
        }> ^after_nested

      ^after_nested(%new_counter: i32):
        // BUG: This block_return should emit "br @loop_N" to continue the loop.
        // But after the nested wasmssa.block above, if labelStack still has
        // the block's label on it, labelStack.back() will return the wrong label.
        //
        // Expected: wasmstack.br @loop_1 (the loop label)
        // Buggy behavior would be: wasmstack.br @block_2 (the inner block label)
        //
        // The CHECK patterns at the function level verify this behavior
        wasmssa.block_return %new_counter : i32
      }> ^exit

    ^exit(%result: i32):
      wasmssa.block_return %result : i32
    }> ^final

  ^final(%final_result: i32):
    wasmssa.return %final_result : i32
  }
}
