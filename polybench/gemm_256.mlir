module {
  func.func @gemm_256(%arg0: f32, %arg1: f32, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0 to %c256 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c256_1 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg6 = %c0_0 to %c256_1 step %c1_2 {
        %0 = memref.load %arg2[%arg5, %arg6] : memref<256x256xf32>
        %1 = arith.mulf %arg1, %0 : f32
        memref.store %1, %arg2[%arg5, %arg6] : memref<256x256xf32>
        %c0_3 = arith.constant 0 : index
        %c256_4 = arith.constant 256 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg7 = %c0_3 to %c256_4 step %c1_5 {
          %2 = memref.load %arg3[%arg5, %arg7] : memref<256x256xf32>
          %3 = memref.load %arg4[%arg7, %arg6] : memref<256x256xf32>
          %4 = memref.load %arg2[%arg5, %arg6] : memref<256x256xf32>
          %5 = arith.mulf %arg0, %2 : f32
          %6 = arith.mulf %5, %3 : f32
          %7 = arith.addf %4, %6 : f32
          memref.store %7, %arg2[%arg5, %arg6] : memref<256x256xf32>
        }
      }
    }
    return
  }
}

