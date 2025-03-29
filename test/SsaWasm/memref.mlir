module {
    memref.global constant @hi : memref<10x2xf32> = dense<[
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
      [7.0, 8.0],
      [9.0, 10.0],
      [11.0, 12.0],
      [13.0, 14.0],
      [15.0, 16.0],
      [17.0, 18.0],
      [19.0, 20.0]
    ]>
    func.func @main() -> f32 {
        %b = memref.get_global @hi : memref<10x2xf32>
        %c0 = arith.constant 5 : index
        %c1 = arith.constant 1 : index
        %e = memref.load %b[%c0, %c1] : memref<10x2xf32>
        %f = memref.load %b[%c1, %c0] : memref<10x2xf32>
        %g = arith.addf %e, %f : f32
        return %g : f32
    }
}
