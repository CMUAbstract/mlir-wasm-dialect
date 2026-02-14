// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// CHECK-LABEL: wasmstack.func @driver
// CHECK: wasmstack.select : !wasmstack.contref_nonnull<@ct>

module {
  wami.type.func @ft = () -> i32
  wami.type.cont @ct = cont @ft

  wasmssa.func @worker() -> i32 {
    %z = wasmssa.const 1 : i32
    wasmssa.return %z : i32
  }

  wasmssa.func @worker_alt() -> i32 {
    %z = wasmssa.const 2 : i32
    wasmssa.return %z : i32
  }

  wasmssa.func @driver(%cond_ref: !wasmssa<local ref to i32>) -> i32 {
    %cond = wasmssa.local_get %cond_ref : !wasmssa<local ref to i32>

    %f0 = wami.ref.func @worker : !wami.funcref<@worker>
    %c0 = wami.cont.new %f0 : !wami.funcref<@worker> as @ct -> !wami.cont<@ct>

    %f1 = wami.ref.func @worker_alt : !wami.funcref<@worker_alt>
    %c1 = wami.cont.new %f1 : !wami.funcref<@worker_alt> as @ct -> !wami.cont<@ct>

    %c = wami.select %cond, %c0, %c1 : !wami.cont<@ct>
    %r = "wami.resume"(%c) <{cont_type = @ct, handlers = []}> : (!wami.cont<@ct>) -> i32
    wasmssa.return %r : i32
  }
}
