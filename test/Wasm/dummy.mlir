// RUN: wasm-opt %s | wasm-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        wasm.i32.constant 1 
        wasm.i32.constant 2
        // CHECK: wasm.i32.add 
        wasm.i32.add 
        return
    }

    // CHECK-LABEL: func @wasm_types(%arg0: !wasm.i32)
    func.func @wasm_types(%arg0: !wasm.i32) {
        return
    }
}
