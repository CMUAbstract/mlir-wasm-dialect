// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// Regression test: if-branch block arguments that are local-backed must be
// initialized at branch entry before local.get-based reuses.

// CHECK: ConvertToWasmStack pass running on module
// CHECK-LABEL: wasmstack.func @if_inputs
// CHECK: wasmstack.if : ([i32]) -> [i32] then {
// CHECK-NEXT: wasmstack.local.set [[THEN:[0-9]+]] : i32
// CHECK-NEXT: wasmstack.local.get [[THEN]] : i32
// CHECK-NEXT: wasmstack.local.get [[THEN]] : i32
// CHECK-NEXT: wasmstack.add : i32
// CHECK: } else {
// CHECK-NEXT: wasmstack.local.set [[ELSE:[0-9]+]] : i32
// CHECK-NEXT: wasmstack.local.get [[ELSE]] : i32
// CHECK-NEXT: wasmstack.local.get [[ELSE]] : i32
// CHECK-NEXT: wasmstack.add : i32

module {
  wasmssa.func @if_inputs(%cond_ref: !wasmssa<local ref to i32>, %x_ref: !wasmssa<local ref to i32>) -> i32 {
    %cond = wasmssa.local_get %cond_ref :  ref to i32
    %x = wasmssa.local_get %x_ref :  ref to i32

    wasmssa.if %cond(%x) : i32 : {
    ^bb0(%a: i32):
      %sum = wasmssa.add %a %a : i32
      wasmssa.block_return %sum : i32
    } "else" {
    ^bb1(%b: i32):
      %sum2 = wasmssa.add %b %b : i32
      wasmssa.block_return %sum2 : i32
    }> ^bb2

  ^bb2(%r: i32):
    wasmssa.return %r : i32
  }
}
