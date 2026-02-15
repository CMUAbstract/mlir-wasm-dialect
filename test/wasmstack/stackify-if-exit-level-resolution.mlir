// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// Regression test: branch exit level resolution must treat wasmstack.if as a
// valid control-frame label target.

// CHECK-LABEL: wasmstack.func @if_exit_level_zero
// CHECK: wasmstack.if : ([]) -> [] then {
// CHECK: wasmstack.br_if @[[IF_EXIT:if_[0-9]+]]
// CHECK: } else {
// CHECK: wasmstack.return

module {
  wasmssa.func @if_exit_level_zero() -> i32 {
    %c0 = wasmssa.const 0 : i32
    %c1 = wasmssa.const 1 : i32

    wasmssa.if %c1 : {
    ^then:
      wasmssa.branch_if %c1 to level 0 else ^cont

    ^cont:
      wasmssa.block_return
    } "else" {
    ^else:
      wasmssa.block_return
    }> ^after

  ^after:
    wasmssa.return %c0 : i32
  }
}

