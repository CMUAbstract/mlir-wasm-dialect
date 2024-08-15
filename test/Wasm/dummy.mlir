// RUN: wasm-opt %s | wasm-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = wasm.constant(1) : !wasm.i32
        %1 = wasm.constant(2) : !wasm.i32
        // CHECK: %{{.*}} = wasm.add %{{.*}}, %{{.*}} : !wasm.i32
        %res = wasm.add %0, %1 : !wasm.i32
        return
    }

    // CHECK-LABEL: func @wasm_types(%arg0: !wasm.i32)
    func.func @wasm_types(%arg0: !wasm.i32) {
        return
    }
}
