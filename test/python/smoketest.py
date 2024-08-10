# RUN: %python %s | FileCheck %s

from mlir_wasm.ir import *
from mlir_wasm.dialects import builtin as builtin_d, wasm as wasm_d

with Context():
    wasm_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = wasm.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: wasm.foo %[[C]] : i32
    print(str(module))
