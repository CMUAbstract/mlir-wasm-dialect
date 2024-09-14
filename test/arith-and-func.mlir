// RUN: mlir-opt %s --load-dialect-plugin=%wasm_libs/WasmPlugin%shlibext --pass-pipeline="builtin.module(convert-to-wasm)" | FileCheck %s

module {
  func.func @foo(%i: i32) -> (i32) {
    %a = arith.constant 2 : i32
    %b = arith.constant 4 : i32
    %c = arith.addi %a, %b : i32
    %d = arith.addi %c, %i : i32
    return %d : i32
  }
}

// CHECK: module {
// CHECK:   wasm.func @foo(%[[ARG0:.*]]: !wasm.local<i32>) -> i32 {
// CHECK:     %[[ZERO:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !wasm.local<i32> to i32
// CHECK:     %[[ONE:.*]] = wasm.temp.local<i32> : !wasm.local<i32>
// CHECK:     wasm.constant 2 : i32
// CHECK:     wasm.temp.local.set : i32
// CHECK:     %[[TWO:.*]] = wasm.temp.local<i32> : !wasm.local<i32>
// CHECK:     wasm.constant 4 : i32
// CHECK:     wasm.temp.local.set : i32
// CHECK:     %[[THREE:.*]] = wasm.temp.local<i32> : !wasm.local<i32>
// CHECK:     wasm.constant 6 : i32
// CHECK:     wasm.temp.local.set : i32
// CHECK:     %[[FOUR:.*]] = builtin.unrealized_conversion_cast %[[THREE]] : !wasm.local<i32> to i32
// CHECK:     %[[FIVE:.*]] = wasm.temp.local<i32> : !wasm.local<i32>
// CHECK:     %[[SIX:.*]] = builtin.unrealized_conversion_cast %[[FOUR]] : i32 to !wasm.local<i32>
// CHECK:     %[[SEVEN:.*]] = builtin.unrealized_conversion_cast %[[ZERO]] : i32 to !wasm.local<i32>
// CHECK:     wasm.temp.local.get : i32
// CHECK:     wasm.temp.local.get : i32
// CHECK:     wasm.add : i32
// CHECK:     wasm.temp.local.set : i32
// CHECK:     %[[EIGHT:.*]] = builtin.unrealized_conversion_cast %[[FIVE]] : !wasm.local<i32> to i32
// CHECK:     %[[NINE:.*]] = builtin.unrealized_conversion_cast %[[EIGHT]] : i32 to !wasm.local<i32>
// CHECK:     wasm.temp.local.get : i32
// CHECK:     wasm.return
// CHECK:   }
// CHECK: }