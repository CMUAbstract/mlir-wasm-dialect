// REQUIRES: wizard_exec
// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wizard_bin --input %t.wasm --expect-i32 16 --quiet
//
// Deterministic runtime-result check: avoid host trace prints so the result
// oracle reflects only @main's returned i32.
//
// Data flow under test:
//   worker(x):
//     payload = (x + x) + 5
//     return suspend(payload)       // value produced by @main handler
//
//   main handler:
//     on_yield(payload, k): return payload + 3
//
// With x = 4:
//   payload = 13, handler result = 16, final = 16.

module {
  wami.type.func @worker_ft = (i32) -> i32
  wami.type.cont @worker_ct = cont @worker_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @worker(%x: !wasmssa<local ref to i32>) -> i32 {
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    %twice = wasmssa.add %arg %arg : i32
    %c5 = wasmssa.const 5 : i32
    %payload = wasmssa.add %twice %c5 : i32
    %from_handler = wami.suspend @yield(%payload) : (i32) -> i32
    wasmssa.return %from_handler : i32
  }

  wasmssa.func exported @main() -> i32 {
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @worker_ct -> !wami.cont<@worker_ct>
    %worker_arg = wasmssa.const 4 : i32

    wasmssa.block : {
    ^bb0:
      %r = wami.resume %c(%worker_arg) @worker_ct [#wami.on_label<tag = @yield, level = 0>] : (!wami.cont<@worker_ct>, i32) -> i32
      wasmssa.return %r : i32
    }> ^on_yield

  ^on_yield(%payload: i32, %k: !wami.cont<@worker_ct>):
    %c3 = wasmssa.const 3 : i32
    %resume_arg = wasmssa.add %payload %c3 : i32
    wasmssa.return %resume_arg : i32
  }
}
