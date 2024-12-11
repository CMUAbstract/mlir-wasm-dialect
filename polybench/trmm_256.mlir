module {
  func.func @trmm_256(%arg0: f32, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c256 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c256_1 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg4 = %c0_0 to %c256_1 step %c1_2 {
        %c1_3 = arith.constant 1 : index
        %0 = arith.addi %arg3, %c1_3 : index
        %c256_4 = arith.constant 256 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg5 = %0 to %c256_4 step %c1_5 {
          %3 = memref.load %arg1[%arg5, %arg3] : memref<256x256xf32>
          %4 = memref.load %arg2[%arg5, %arg4] : memref<256x256xf32>
          %5 = memref.load %arg2[%arg3, %arg4] : memref<256x256xf32>
          %6 = arith.mulf %3, %4 : f32
          %7 = arith.addf %5, %6 : f32
          memref.store %7, %arg2[%arg3, %arg4] : memref<256x256xf32>
        }
        %1 = memref.load %arg2[%arg3, %arg4] : memref<256x256xf32>
        %2 = arith.mulf %arg0, %1 : f32
        memref.store %2, %arg2[%arg3, %arg4] : memref<256x256xf32>
      }
    }
    return
  }
}

