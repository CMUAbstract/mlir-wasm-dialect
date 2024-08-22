// RUN: wasm-opt %s | wasm-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        wasm.constant 1 : i32
        wasm.constant 2 : i32
        // CHECK: wasm.add : i32
        wasm.add : i32
        return
    }

    // CHECK-LABEL: func @wasm_types(%arg0: !wasm.i32)
    func.func @wasm_types(%arg0: !wasm.i32) {
        return
    }
}
