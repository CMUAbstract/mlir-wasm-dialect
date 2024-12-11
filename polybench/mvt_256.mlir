func.func @mvt_256(
  %x1: memref<256xf32>, 
  %x2: memref<256xf32>, 
  %y1: memref<256xf32>, 
  %y2: memref<256xf32>, 
  %A: memref<256x256xf32>) {
  affine.for %n = 0 to 1 {
    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 256 {
        %0 = affine.load %A[%i, %j] : memref<256x256xf32>
        %1 = affine.load %y1[%j] : memref<256xf32>
        %2 = affine.load %x1[%i] : memref<256xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %x1[%i] : memref<256xf32>
      }
    }

    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 256 {
        %0 = affine.load %A[%j, %i] : memref<256x256xf32>
        %1 = affine.load %y2[%j] : memref<256xf32>
        %2 = affine.load %x2[%i] : memref<256xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %x2[%i] : memref<256xf32>
      }
    }
  }
  return
}
