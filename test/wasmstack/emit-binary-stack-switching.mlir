// RUN: wasm-opt %s --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: od -An -tx1 -v %t.wasm | tr -d ' \n' | FileCheck %s --check-prefix=HEX

// HEX: 0061736d01000000
// Type section includes continuation constructor (0x5d).
// HEX: 5d
// Tag section id (0x0d) is emitted.
// HEX: 0d
// Stack-switching opcodes appear in function body order.
// Typed non-null continuation refs in locals encode as ref + heaptype index.
// HEX: 01016401
// Typed nullable continuation ref.null encodes heaptype index.
// HEX: d001
// Barrier currently lowers to nop; this checks the emitted identity fence sequence.
// HEX: 4101011a
// HEX: d2
// HEX: e0
// HEX: e1
// HEX: e3
// HEX: e2
// HEX: d2
// HEX: e0
// HEX: e6
// HEX: d2
// HEX: e0
// HEX: e5

wasmstack.module @emit_stack_switching {
  wasmstack.type.func @ft = () -> ()
  wasmstack.type.cont @ct = cont @ft
  wasmstack.tag @yield : () -> ()

  wasmstack.func @worker : () -> () {
    wasmstack.return
  }

  wasmstack.func @nonnull_local : () -> () {
    wasmstack.local 0 : !wasmstack.contref_nonnull<@ct>
    wasmstack.return
  }

  wasmstack.func @driver : () -> () {
    wasmstack.ref.null : !wasmstack.contref<@ct>
    wasmstack.drop : !wasmstack.contref<@ct>

    wasmstack.i32.const 1
    wasmstack.barrier : (i32) -> i32
    wasmstack.drop : i32

    wasmstack.ref.func @worker
    wasmstack.cont.new @ct
    wasmstack.cont.bind @ct -> @ct
    wasmstack.resume @ct (@yield -> switch)

    wasmstack.suspend @yield

    wasmstack.ref.func @worker
    wasmstack.cont.new @ct
    wasmstack.switch @ct (tag: @yield)

    wasmstack.ref.func @worker
    wasmstack.cont.new @ct
    wasmstack.resume_throw @ct (@yield -> switch)
    wasmstack.return
  }
}
