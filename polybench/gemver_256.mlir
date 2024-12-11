module {
  func.func @gemver_256(%arg0: f32, %arg1: f32, %arg2: memref<256x256xf32>, %arg3: memref<256xf32>, %arg4: memref<256xf32>, %arg5: memref<256xf32>, %arg6: memref<256xf32>, %arg7: memref<256xf32>, %arg8: memref<256xf32>, %arg9: memref<256xf32>, %arg10: memref<256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg11 = %c0 to %c1 step %c1_0 {
      %c0_1 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg12 = %c0_1 to %c256 step %c1_2 {
        %c0_12 = arith.constant 0 : index
        %c256_13 = arith.constant 256 : index
        %c1_14 = arith.constant 1 : index
        scf.for %arg13 = %c0_12 to %c256_13 step %c1_14 {
          %0 = memref.load %arg2[%arg12, %arg13] : memref<256x256xf32>
          %1 = memref.load %arg3[%arg12] : memref<256xf32>
          %2 = memref.load %arg4[%arg13] : memref<256xf32>
          %3 = memref.load %arg5[%arg12] : memref<256xf32>
          %4 = memref.load %arg6[%arg13] : memref<256xf32>
          %5 = arith.mulf %1, %2 : f32
          %6 = arith.mulf %3, %4 : f32
          %7 = arith.addf %0, %5 : f32
          %8 = arith.addf %7, %6 : f32
          memref.store %8, %arg2[%arg12, %arg13] : memref<256x256xf32>
        }
      }
      %c0_3 = arith.constant 0 : index
      %c256_4 = arith.constant 256 : index
      %c1_5 = arith.constant 1 : index
      scf.for %arg12 = %c0_3 to %c256_4 step %c1_5 {
        %c0_12 = arith.constant 0 : index
        %c256_13 = arith.constant 256 : index
        %c1_14 = arith.constant 1 : index
        scf.for %arg13 = %c0_12 to %c256_13 step %c1_14 {
          %0 = memref.load %arg8[%arg12] : memref<256xf32>
          %1 = memref.load %arg2[%arg12, %arg13] : memref<256x256xf32>
          %2 = memref.load %arg9[%arg13] : memref<256xf32>
          %3 = arith.mulf %arg1, %1 : f32
          %4 = arith.mulf %2, %3 : f32
          %5 = arith.addf %0, %4 : f32
          memref.store %5, %arg8[%arg12] : memref<256xf32>
        }
      }
      %c0_6 = arith.constant 0 : index
      %c256_7 = arith.constant 256 : index
      %c1_8 = arith.constant 1 : index
      scf.for %arg12 = %c0_6 to %c256_7 step %c1_8 {
        %0 = memref.load %arg8[%arg12] : memref<256xf32>
        %1 = memref.load %arg10[%arg12] : memref<256xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %arg8[%arg12] : memref<256xf32>
      }
      %c0_9 = arith.constant 0 : index
      %c256_10 = arith.constant 256 : index
      %c1_11 = arith.constant 1 : index
      scf.for %arg12 = %c0_9 to %c256_10 step %c1_11 {
        %c0_12 = arith.constant 0 : index
        %c256_13 = arith.constant 256 : index
        %c1_14 = arith.constant 1 : index
        scf.for %arg13 = %c0_12 to %c256_13 step %c1_14 {
          %0 = memref.load %arg7[%arg12] : memref<256xf32>
          %1 = memref.load %arg2[%arg12, %arg13] : memref<256x256xf32>
          %2 = memref.load %arg8[%arg13] : memref<256xf32>
          %3 = arith.mulf %arg0, %1 : f32
          %4 = arith.mulf %2, %3 : f32
          %5 = arith.addf %0, %4 : f32
          memref.store %5, %arg7[%arg12] : memref<256xf32>
        }
      }
    }
    return
  }
}

