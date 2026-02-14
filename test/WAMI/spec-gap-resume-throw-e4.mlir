// XFAIL: *
// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: od -An -tx1 -v %t.wasm | tr -d ' \n' | FileCheck %s --check-prefix=E4

// TODO(issue-draft): .omc/issue-drafts/spec-gap-resume-throw-e4.md
// Desired: wasm binary should contain resume_throw (e4) encoding when an
// explicit exception tag is provided.
// E4: e4

module {
  wami.type.func @ft = (i32) -> i32
  wami.type.cont @ct = cont @ft

  wami.tag @yield : (i32) -> i32
  wami.tag @abort : () -> ()

  wasmssa.func @worker(%x: !wasmssa<local ref to i32>) -> i32 {
    %v = wasmssa.local_get %x : !wasmssa<local ref to i32>
    wasmssa.return %v : i32
  }

  wasmssa.func @driver(%x: !wasmssa<local ref to i32>) {
    %f = wami.ref.func @worker : !wami.funcref<@worker>
    %c = wami.cont.new %f : !wami.funcref<@worker> as @ct -> !wami.cont<@ct>
    %arg = wasmssa.local_get %x : !wasmssa<local ref to i32>
    "wami.resume_throw"(%c, %arg) <{cont_type = @ct, exn_tag = @abort, handlers = [#wami.on_switch<tag = @yield>]}> : (!wami.cont<@ct>, i32) -> ()
    wasmssa.return
  }
}
