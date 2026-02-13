// REQUIRES: wizard_exec
// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack | wasm-emit --mlir-to-wasm -o %t.wasm
// RUN: %run_wizard_bin --input %t.wasm --expect-i32 42 --quiet

module {
  wasmssa.import_func "puti" from "wizeng" as @print_i32 {type = (i32) -> ()}

  wasmssa.func exported @main() -> i32 {
    %v = wasmssa.const 42 : i32
    wasmssa.call @print_i32(%v) : (i32) -> ()
    wasmssa.return %v : i32
  }
}
