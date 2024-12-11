func.func @doitgen_256(
              %A: memref<256x256x256xf32>, 
              %C4: memref<256x256xf32>, 
              %sum: memref<256xf32>
  ) {
  affine.for %r = 0 to 256 {
    affine.for %q = 0 to 256 {
      affine.for %p = 0 to 256 {
        %c0 = arith.constant 0.0 : f32
        affine.store %c0, %sum[%p] : memref<256xf32>
        affine.for %s = 0 to 256 {
          %0 = affine.load %A[%r, %q, %s] : memref<256x256x256xf32>
          %1 = affine.load %C4[%s, %p] : memref<256x256xf32>
          %2 = affine.load %sum[%p] : memref<256xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %sum[%p] : memref<256xf32>
        }
      }
      
      affine.for %p = 0 to 256 {
        %0 = affine.load %sum[%p] : memref<256xf32>
        affine.store %0, %A[%r, %q, %p] : memref<256x256x256xf32>
      }
    }
  }
  return
}
