// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack | FileCheck %s

// CHECK-LABEL: wasmstack.func @driver : () -> i32
// CHECK: wasmstack.ref.null : !wasmstack.contref<@ct>
// CHECK: wasmstack.drop : !wasmstack.contref<@ct>
// CHECK: wasmstack.cont.new @ct
// CHECK: wasmstack.drop : !wasmstack.contref_nonnull<@ct>

module {
  wami.type.func @ft = () -> i32
  wami.type.cont @ct = cont @ft

  wasmssa.func @worker() -> i32 {
    %v = wasmssa.const 1 : i32
    wasmssa.return %v : i32
  }

  wasmssa.func @driver() -> i32 {
    %n = wami.ref.null : !wami.cont<@ct, true>
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @ct -> !wami.cont<@ct>
    %z = wasmssa.const 0 : i32
    wasmssa.return %z : i32
  }
}
