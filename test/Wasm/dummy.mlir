// RUN: wasm-opt %s | wasm-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %x = wasm.temp.local<i32> : !wasm.local<i32>
        wasm.constant 1 : i32
        wasm.temp.local.set %x : <i32>

        %y = wasm.temp.local<i32> : !wasm.local<i32>
        wasm.constant 2 : i32
        wasm.temp.local.set %y : <i32>

        wasm.temp.local.get %x : <i32>
        wasm.temp.local.get %y : <i32>
        // CHECK: wasm.add : i32
        wasm.add : i32
        return
    }

    // CHECK-LABEL: func @wasm_types(%arg0: !wasm.local<i32>)
    func.func @wasm_types(%arg0: !wasm.local<i32>) {
        return
    }
}
