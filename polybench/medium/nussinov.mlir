#map = affine_map<(d0) -> (-d0 + 499)>
#map1 = affine_map<(d0) -> (-d0 + 500)>
#map2 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 1 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 + d1 - 501 >= 0)>
module {
  func.func @main() -> i32 attributes { exported } {
    %c500 = arith.constant 500 : index
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<500x500xi32>
    scf.for %arg0 = %c0 to %c500 step %c1 {
      scf.for %arg1 = %c0 to %c500 step %c1 {
        memref.store %c0_i32, %alloc[%arg0, %arg1] : memref<500x500xi32>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 500 {
      %0 = affine.apply #map(%arg0)
      %1 = arith.index_cast %0 : index to i32
      %2 = arith.addi %1, %c1_i32 : i32
      %3 = arith.remsi %2, %c4_i32 : i32
      %4 = arith.trunci %3 : i32 to i8
      %5 = arith.extsi %4 : i8 to i32
      affine.for %arg1 = #map1(%arg0) to 500 {
        affine.if #set(%arg1) {
          %6 = affine.load %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
          %7 = affine.load %alloc[-%arg0 + 499, %arg1 - 1] : memref<500x500xi32>
          %8 = arith.cmpi sge, %6, %7 : i32
          %9 = arith.select %8, %6, %7 : i32
          affine.store %9, %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
        }
        affine.if #set(%arg0) {
          %6 = affine.load %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
          %7 = affine.load %alloc[-%arg0 + 500, %arg1] : memref<500x500xi32>
          %8 = arith.cmpi sge, %6, %7 : i32
          %9 = arith.select %8, %6, %7 : i32
          affine.store %9, %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
        }
        affine.if #set1(%arg1, %arg0) {
          affine.if #set2(%arg0, %arg1) {
            %6 = affine.load %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
            %7 = affine.load %alloc[-%arg0 + 500, %arg1 - 1] : memref<500x500xi32>
            %8 = arith.index_cast %arg1 : index to i32
            %9 = arith.addi %8, %c1_i32 : i32
            %10 = arith.remsi %9, %c4_i32 : i32
            %11 = arith.trunci %10 : i32 to i8
            %12 = arith.extsi %11 : i8 to i32
            %13 = arith.addi %5, %12 : i32
            %14 = arith.cmpi eq, %13, %c3_i32 : i32
            %15 = arith.extui %14 : i1 to i32
            %16 = arith.addi %7, %15 : i32
            %17 = arith.cmpi sge, %6, %16 : i32
            %18 = arith.select %17, %6, %16 : i32
            affine.store %18, %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
          } else {
            %6 = affine.load %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
            %7 = affine.load %alloc[-%arg0 + 500, %arg1 - 1] : memref<500x500xi32>
            %8 = arith.cmpi sge, %6, %7 : i32
            %9 = arith.select %8, %6, %7 : i32
            affine.store %9, %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
          }
        }
        affine.for %arg2 = #map1(%arg0) to #map2(%arg1) {
          %6 = affine.load %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
          %7 = affine.load %alloc[-%arg0 + 499, %arg2] : memref<500x500xi32>
          %8 = affine.load %alloc[%arg2 + 1, %arg1] : memref<500x500xi32>
          %9 = arith.addi %7, %8 : i32
          %10 = arith.cmpi sge, %6, %9 : i32
          %11 = arith.select %10, %6, %9 : i32
          affine.store %11, %alloc[-%arg0 + 499, %arg1] : memref<500x500xi32>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c500 step %c1 {
      scf.for %arg1 = %arg0 to %c500 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<500x500xi32>
        func.call @print_i32(%0) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<500x500xi32>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
