module {
  func.func @three_mm_256(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<256x256xf32>, %arg5: memref<256x256xf32>, %arg6: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg7 = %c0 to %c1 step %c1_0 {
      %c0_1 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg8 = %c0_1 to %c256 step %c1_2 {
        %c0_9 = arith.constant 0 : index
        %c256_10 = arith.constant 256 : index
        %c1_11 = arith.constant 1 : index
        scf.for %arg9 = %c0_9 to %c256_10 step %c1_11 {
          %cst = arith.constant 0.000000e+00 : f32
          memref.store %cst, %arg0[%arg8, %arg9] : memref<256x256xf32>
          %c0_12 = arith.constant 0 : index
          %c256_13 = arith.constant 256 : index
          %c1_14 = arith.constant 1 : index
          scf.for %arg10 = %c0_12 to %c256_13 step %c1_14 {
            %0 = memref.load %arg1[%arg8, %arg10] : memref<256x256xf32>
            %1 = memref.load %arg2[%arg10, %arg9] : memref<256x256xf32>
            %2 = memref.load %arg0[%arg8, %arg9] : memref<256x256xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            memref.store %4, %arg0[%arg8, %arg9] : memref<256x256xf32>
          }
        }
      }
      %c0_3 = arith.constant 0 : index
      %c256_4 = arith.constant 256 : index
      %c1_5 = arith.constant 1 : index
      scf.for %arg8 = %c0_3 to %c256_4 step %c1_5 {
        %c0_9 = arith.constant 0 : index
        %c256_10 = arith.constant 256 : index
        %c1_11 = arith.constant 1 : index
        scf.for %arg9 = %c0_9 to %c256_10 step %c1_11 {
          %cst = arith.constant 0.000000e+00 : f32
          memref.store %cst, %arg3[%arg8, %arg9] : memref<256x256xf32>
          %c0_12 = arith.constant 0 : index
          %c256_13 = arith.constant 256 : index
          %c1_14 = arith.constant 1 : index
          scf.for %arg10 = %c0_12 to %c256_13 step %c1_14 {
            %0 = memref.load %arg4[%arg8, %arg10] : memref<256x256xf32>
            %1 = memref.load %arg5[%arg10, %arg9] : memref<256x256xf32>
            %2 = memref.load %arg3[%arg8, %arg9] : memref<256x256xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            memref.store %4, %arg3[%arg8, %arg9] : memref<256x256xf32>
          }
        }
      }
      %c0_6 = arith.constant 0 : index
      %c256_7 = arith.constant 256 : index
      %c1_8 = arith.constant 1 : index
      scf.for %arg8 = %c0_6 to %c256_7 step %c1_8 {
        %c0_9 = arith.constant 0 : index
        %c256_10 = arith.constant 256 : index
        %c1_11 = arith.constant 1 : index
        scf.for %arg9 = %c0_9 to %c256_10 step %c1_11 {
          %cst = arith.constant 0.000000e+00 : f32
          memref.store %cst, %arg6[%arg8, %arg9] : memref<256x256xf32>
          %c0_12 = arith.constant 0 : index
          %c256_13 = arith.constant 256 : index
          %c1_14 = arith.constant 1 : index
          scf.for %arg10 = %c0_12 to %c256_13 step %c1_14 {
            %0 = memref.load %arg0[%arg8, %arg10] : memref<256x256xf32>
            %1 = memref.load %arg3[%arg10, %arg9] : memref<256x256xf32>
            %2 = memref.load %arg6[%arg8, %arg9] : memref<256x256xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            memref.store %4, %arg6[%arg8, %arg9] : memref<256x256xf32>
          }
        }
      }
    }
    return
  }
}

