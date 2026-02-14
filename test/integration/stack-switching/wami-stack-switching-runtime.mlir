// REQUIRES: wizard_exec
// XFAIL: *
// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wizard_bin --input %t.wasm --expect-i32 7 --quiet

// TODO: remove XFAIL once runtime continuation-state handling is fixed.
// Current Wizard runtime failure for this workload:
//   expected continuation, got <null>

module {
  // Alias Wizard's wizeng.puti host import as print_i32 for readability.
  wasmssa.import_func "puti" from "wizeng" as @print_i32 {type = (i32) -> ()}

  wami.type.func @worker_ft = (i32) -> i32
  wami.type.cont @worker_ct = cont @worker_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @worker(%x: !wasmssa<local ref to i32>) -> i32 {
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    wasmssa.call @print_i32(%arg) : (i32) -> ()
    %from_handler = "wami.suspend"(%arg) <{tag = @yield}> : (i32) -> i32
    wasmssa.call @print_i32(%from_handler) : (i32) -> ()
    wasmssa.return %from_handler : i32
  }

  wasmssa.func exported @main() -> i32 {
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @worker_ct -> !wami.cont<@worker_ct>
    %arg = wasmssa.const 7 : i32
    wasmssa.call @print_i32(%arg) : (i32) -> ()

    wasmssa.block : {
    ^bb0:
      %r = "wami.resume"(%c, %arg) <{cont_type = @worker_ct, handlers = [#wami.on_label<tag = @yield, level = 0>]}> : (!wami.cont<@worker_ct>, i32) -> i32
      wasmssa.return %r : i32
    }> ^on_yield

  ^on_yield(%payload: i32, %k: !wami.cont<@worker_ct>):
    wasmssa.call @print_i32(%payload) : (i32) -> ()
    wasmssa.return %payload : i32
  }
}
