module {
  func.func @two_mm_256(%arg0: f32, %arg1: f32, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<256x256xf32>, %arg5: memref<256x256xf32>, %arg6: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg7 = %c0 to %c1 step %c1_0 {
      %c0_1 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg8 = %c0_1 to %c256 step %c1_2 {
        %c0_6 = arith.constant 0 : index
        %c256_7 = arith.constant 256 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg9 = %c0_6 to %c256_7 step %c1_8 {
          %cst = arith.constant 0.000000e+00 : f32
          memref.store %cst, %arg2[%arg8, %arg9] : memref<256x256xf32>
          %c0_9 = arith.constant 0 : index
          %c256_10 = arith.constant 256 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg10 = %c0_9 to %c256_10 step %c1_11 {
            %0 = memref.load %arg3[%arg8, %arg10] : memref<256x256xf32>
            %1 = memref.load %arg4[%arg10, %arg9] : memref<256x256xf32>
            %2 = memref.load %arg2[%arg8, %arg9] : memref<256x256xf32>
            %3 = arith.mulf %arg0, %0 : f32
            %4 = arith.mulf %3, %1 : f32
            %5 = arith.addf %2, %4 : f32
            memref.store %5, %arg2[%arg8, %arg9] : memref<256x256xf32>
          }
        }
      }
      %c0_3 = arith.constant 0 : index
      %c256_4 = arith.constant 256 : index
      %c1_5 = arith.constant 1 : index
      scf.for %arg8 = %c0_3 to %c256_4 step %c1_5 {
        %c0_6 = arith.constant 0 : index
        %c256_7 = arith.constant 256 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg9 = %c0_6 to %c256_7 step %c1_8 {
          %0 = memref.load %arg6[%arg8, %arg9] : memref<256x256xf32>
          %1 = arith.mulf %arg1, %0 : f32
          memref.store %1, %arg6[%arg8, %arg9] : memref<256x256xf32>
          %c0_9 = arith.constant 0 : index
          %c256_10 = arith.constant 256 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg10 = %c0_9 to %c256_10 step %c1_11 {
            %2 = memref.load %arg2[%arg8, %arg10] : memref<256x256xf32>
            %3 = memref.load %arg5[%arg10, %arg9] : memref<256x256xf32>
            %4 = memref.load %arg6[%arg8, %arg9] : memref<256x256xf32>
            %5 = arith.mulf %2, %3 : f32
            %6 = arith.addf %4, %5 : f32
            memref.store %6, %arg6[%arg8, %arg9] : memref<256x256xf32>
          }
        }
      }
    }
    return
  }
}

