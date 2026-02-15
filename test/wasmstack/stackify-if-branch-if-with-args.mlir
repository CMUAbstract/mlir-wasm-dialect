// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// Regression test: branch_if terminators inside wasmssa.if regions must use
// terminator lowering so branch operands are preserved and else-successor CFG
// is emitted correctly.

// CHECK-LABEL: wasmstack.func @if_branch_if_with_args
// CHECK: wasmstack.if : ([i32]) -> [i32] then {
// CHECK: wasmstack.br_if @[[IF_EXIT:if_[0-9]+]]
// CHECK: wasmstack.add : i32
// CHECK: } else {

module {
  wasmssa.func @if_branch_if_with_args() -> i32 {
    %c0 = wasmssa.const 0 : i32
    %c1 = wasmssa.const 1 : i32

    wasmssa.if %c1(%c0) : i32 : {
    ^then(%x: i32):
      wasmssa.branch_if %c1 to level 0 with args(%x : i32) else ^cont

    ^cont(%carry: i32):
      %inc = wasmssa.add %carry %c1 : i32
      wasmssa.block_return %inc : i32
    } "else" {
    ^else(%y: i32):
      wasmssa.block_return %y : i32
    }> ^exit

  ^exit(%r: i32):
    wasmssa.return %r : i32
  }
}
