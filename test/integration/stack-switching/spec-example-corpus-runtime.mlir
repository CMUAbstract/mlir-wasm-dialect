// REQUIRES: wizard_exec
// XFAIL: *
// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wizard_bin --input %t.wasm --expect-i32 42 --quiet
//
// Runtime corpus inspired by official stack-switching examples:
//   proposals/stack-switching/examples/{generator,generator-extended}.wast
//
// Trace markers (print_i32):
//   21xx: generator-style flow
//   22xx: generator-extended-style flow
//
// TODO: remove XFAIL once wasm binary emission supports stack-switching ops.

module {
  wasmssa.import_func "puti" from "wizeng" as @print_i32 {type = (i32) -> ()}

  wami.type.func @gen_ft = () -> i32
  wami.type.cont @gen_ct = cont @gen_ft
  wami.tag @gen : (i32) -> i32

  wami.type.func @arg_ft = (i32) -> i32
  wami.type.cont @arg_ct = cont @arg_ft
  wami.tag @yield : (i32) -> i32

  wasmssa.func @generator() -> i32 {
    %m2101 = wasmssa.const 2101 : i32
    wasmssa.call @print_i32(%m2101) : (i32) -> ()

    %payload = wasmssa.const 100 : i32
    %from_handler = "wami.suspend"(%payload) <{tag = @gen}> : (i32) -> i32

    wasmssa.call @print_i32(%from_handler) : (i32) -> ()
    wasmssa.return %from_handler : i32
  }

  wasmssa.func @needs_arg(%x: !wasmssa<local ref to i32>) -> i32 {
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    wasmssa.call @print_i32(%arg) : (i32) -> ()

    %from_handler = "wami.suspend"(%arg) <{tag = @yield}> : (i32) -> i32
    wasmssa.call @print_i32(%from_handler) : (i32) -> ()
    wasmssa.return %from_handler : i32
  }

  wasmssa.func exported @main() -> i32 {
    // generator.wast-style: resume a no-arg continuation and handle suspend payload.
    %m2111 = wasmssa.const 2111 : i32
    wasmssa.call @print_i32(%m2111) : (i32) -> ()

    %gen_f = wami.ref.func @generator : !wami.funcref<@generator>
    %gen_c = wami.cont.new %gen_f : !wami.funcref<@generator> as @gen_ct -> !wami.cont<@gen_ct>
    %gen_result = "wami.resume"(%gen_c) <{cont_type = @gen_ct, handler_tags = [@gen]}> ({
    ^bb0(%payload: i32):
      wasmssa.call @print_i32(%payload) : (i32) -> ()
      wami.handler.yield %payload : i32
    }) : (!wami.cont<@gen_ct>) -> i32
    wasmssa.call @print_i32(%gen_result) : (i32) -> ()

    // generator-extended.wast-style: resume a continuation that takes an argument.
    %m2211 = wasmssa.const 2211 : i32
    wasmssa.call @print_i32(%m2211) : (i32) -> ()

    %arg_f = wami.ref.func @needs_arg : !wami.funcref<@needs_arg>
    %arg_c = wami.cont.new %arg_f : !wami.funcref<@needs_arg> as @arg_ct -> !wami.cont<@arg_ct>
    %in42 = wasmssa.const 42 : i32
    %arg_result = "wami.resume"(%arg_c, %in42) <{cont_type = @arg_ct, handler_tags = [@yield]}> ({
    ^bb0(%payload: i32):
      wasmssa.call @print_i32(%payload) : (i32) -> ()
      wami.handler.yield %payload : i32
    }) : (!wami.cont<@arg_ct>, i32) -> i32
    wasmssa.call @print_i32(%arg_result) : (i32) -> ()

    wasmssa.return %arg_result : i32
  }
}
