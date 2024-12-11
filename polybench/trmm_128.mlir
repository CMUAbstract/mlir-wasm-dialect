module {
  func.func @main(%arg0: f32, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c128 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c128_1 = arith.constant 128 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg4 = %c0_0 to %c128_1 step %c1_2 {
        %c1_3 = arith.constant 1 : index
        %0 = arith.addi %arg3, %c1_3 : index
        %c128_4 = arith.constant 128 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg5 = %0 to %c128_4 step %c1_5 {
          %3 = memref.load %arg1[%arg5, %arg3] : memref<128x128xf32>
          %4 = memref.load %arg2[%arg5, %arg4] : memref<128x128xf32>
          %5 = memref.load %arg2[%arg3, %arg4] : memref<128x128xf32>
          %6 = arith.mulf %3, %4 : f32
          %7 = arith.addf %5, %6 : f32
          memref.store %7, %arg2[%arg3, %arg4] : memref<128x128xf32>
        }
        %1 = memref.load %arg2[%arg3, %arg4] : memref<128x128xf32>
        %2 = arith.mulf %arg0, %1 : f32
        memref.store %2, %arg2[%arg3, %arg4] : memref<128x128xf32>
      }
    }
    return
  }
}

