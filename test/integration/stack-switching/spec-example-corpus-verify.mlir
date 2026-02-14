// RUN: wasm-opt %s --verify-wasmstack 2>&1 | FileCheck %s

// Seed integration corpus extracted from official stack-switching examples:
//   proposals/stack-switching/examples/*.wast
//
// print_i32 marker ranges:
//   11xx: generator
//   12xx: generator-extended
//   13xx: fun-pipes
//   14xx: scheduler2/switch
//   15xx: scheduler2/throw

// CHECK-LABEL: wasmstack.module @spec_generator
wasmstack.module @spec_generator {
  wasmstack.type.func @gen_ft = () -> ()
  wasmstack.type.cont @gen_ct = cont @gen_ft
  wasmstack.tag @gen : (i32) -> ()
  wasmstack.tag @gen_switch : () -> ()
  wasmstack.import_func "puti" from "wizeng" as @print_i32 : (i32) -> ()

  // From: examples/generator.wast
  wasmstack.func @generator : () -> () {
    wasmstack.i32.const 1101
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.i32.const 100
    wasmstack.suspend @gen
    wasmstack.i32.const 1102
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }

  // CHECK: wasmstack.resume @gen_ct (@gen_switch -> switch)
  wasmstack.func @consumer_once : () -> () {
    wasmstack.ref.func @generator
    wasmstack.cont.new @gen_ct
    wasmstack.i32.const 1111
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.resume @gen_ct (@gen_switch -> switch)
    wasmstack.i32.const 1112
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }
}

// CHECK-LABEL: wasmstack.module @spec_generator_extended_bind
wasmstack.module @spec_generator_extended_bind {
  wasmstack.type.func @ft0 = () -> ()
  wasmstack.type.func @ft1 = (i32) -> ()
  wasmstack.type.cont @ct0 = cont @ft0
  wasmstack.type.cont @ct1 = cont @ft1
  wasmstack.import_func "puti" from "wizeng" as @print_i32 : (i32) -> ()

  wasmstack.func @needs_arg : (i32) -> () {
    wasmstack.local.get 0 : i32
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }

  // From: examples/generator-extended.wast
  // CHECK: wasmstack.cont.bind @ct1 -> @ct0
  wasmstack.func @bind_then_resume : () -> () {
    wasmstack.ref.func @needs_arg
    wasmstack.cont.new @ct1
    wasmstack.i32.const 1201
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.i32.const 42
    wasmstack.cont.bind @ct1 -> @ct0
    wasmstack.i32.const 1202
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.resume @ct0
    wasmstack.i32.const 1203
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }
}

// CHECK-LABEL: wasmstack.module @spec_fun_pipes
wasmstack.module @spec_fun_pipes {
  wasmstack.type.func @pfun = () -> i32
  wasmstack.type.func @cfun = (i32) -> i32
  wasmstack.type.cont @producer_ct = cont @pfun
  wasmstack.type.cont @consumer_ct = cont @cfun

  wasmstack.tag @send : (i32) -> ()
  wasmstack.tag @send_switch : () -> ()
  wasmstack.tag @receive : () -> i32
  wasmstack.import_func "puti" from "wizeng" as @print_i32 : (i32) -> ()

  wasmstack.func @producer : () -> i32 {
    wasmstack.i32.const 1301
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.i32.const 5
    wasmstack.suspend @send
    wasmstack.i32.const 1302
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.i32.const 5
    wasmstack.return
  }

  wasmstack.func @consumer : (i32) -> i32 {
    wasmstack.local.get 0 : i32
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.local.get 0 : i32
    wasmstack.suspend @receive
    wasmstack.add : i32
    wasmstack.return
  }

  // From: examples/fun-pipes.wast
  // CHECK: wasmstack.resume @consumer_ct (@receive -> switch)
  // CHECK: wasmstack.resume @producer_ct (@send_switch -> switch)
  wasmstack.func @pipe_once : () -> () {
    wasmstack.ref.func @producer
    wasmstack.cont.new @producer_ct

    wasmstack.i32.const 1311
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.i32.const 7
    wasmstack.ref.func @consumer
    wasmstack.cont.new @consumer_ct
    wasmstack.resume @consumer_ct (@receive -> switch)
    wasmstack.call @print_i32 : (i32) -> ()

    wasmstack.i32.const 1312
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.resume @producer_ct (@send_switch -> switch)
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }
}

// CHECK-LABEL: wasmstack.module @spec_scheduler2_switch
wasmstack.module @spec_scheduler2_switch {
  wasmstack.type.func @task_ft = (i32) -> ()
  wasmstack.type.cont @task_ct = cont @task_ft
  wasmstack.tag @yield : () -> ()
  wasmstack.import_func "puti" from "wizeng" as @print_i32 : (i32) -> ()

  wasmstack.func @task : (i32) -> () {
    wasmstack.local.get 0 : i32
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }

  // From: examples/scheduler2.wast
  // CHECK: wasmstack.resume @task_ct (@yield -> switch)
  wasmstack.func @entry_resume_switch : () -> () {
    wasmstack.i32.const 1
    wasmstack.ref.func @task
    wasmstack.cont.new @task_ct
    wasmstack.i32.const 1401
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.resume @task_ct (@yield -> switch)
    wasmstack.i32.const 1402
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }

  // CHECK: wasmstack.switch @task_ct(tag : @yield)
  wasmstack.func @yield_to_next : () -> () {
    wasmstack.i32.const 2
    wasmstack.ref.func @task
    wasmstack.cont.new @task_ct
    wasmstack.i32.const 1411
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.switch @task_ct (tag: @yield)
    wasmstack.i32.const 1412
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }
}

// CHECK-LABEL: wasmstack.module @spec_scheduler2_throw
wasmstack.module @spec_scheduler2_throw {
  wasmstack.type.func @task_ft = (i32) -> ()
  wasmstack.type.cont @task_ct = cont @task_ft
  wasmstack.tag @yield : () -> ()
  wasmstack.import_func "puti" from "wizeng" as @print_i32 : (i32) -> ()

  wasmstack.func @task : (i32) -> () {
    wasmstack.local.get 0 : i32
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }

  // From: examples/scheduler2-throw.wast
  // CHECK: wasmstack.resume_throw @task_ct (@yield -> switch)
  wasmstack.func @cancel_one : () -> () {
    wasmstack.i32.const 0
    wasmstack.ref.func @task
    wasmstack.cont.new @task_ct
    wasmstack.i32.const 1501
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.resume_throw @task_ct (@yield -> switch)
    wasmstack.i32.const 1502
    wasmstack.call @print_i32 : (i32) -> ()
    wasmstack.return
  }
}
