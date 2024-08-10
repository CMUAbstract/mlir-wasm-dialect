// RUN: wasm-opt %s | wasm-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = wasm.foo %{{.*}} : i32
        %res = wasm.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @wasm_types(%arg0: !wasm.custom<"10">)
    func.func @wasm_types(%arg0: !wasm.custom<"10">) {
        return
    }
}
