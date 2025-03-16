#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2000 = arith.constant 2000 : index
    %c2600 = arith.constant 2600 : index
    %cst = arith.constant 2.000000e+03 : f64
    %cst_0 = arith.constant 1.000000e+02 : f64
    %cst_1 = arith.constant 1.000000e+01 : f64
    %cst_2 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2000_i32 = arith.constant 2000 : i32
    %alloc = memref.alloc() : memref<2000x2600xf64>
    %alloc_3 = memref.alloc() : memref<2600x2600xf64>
    %alloc_4 = memref.alloc() : memref<2000x2600xf64>
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c2600 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.muli %1, %2 : i32
        %4 = arith.remsi %3, %c2000_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        %7 = arith.mulf %6, %cst_0 : f64
        %8 = arith.addf %7, %cst_1 : f64
        memref.store %8, %alloc[%arg0, %arg1] : memref<2000x2600xf64>
        memref.store %cst_2, %alloc_4[%arg0, %arg1] : memref<2000x2600xf64>
      }
    }
    scf.for %arg0 = %c0 to %c2600 step %c1 {
      scf.for %arg1 = %c0 to %c2600 step %c1 {
        memref.store %cst_2, %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    %alloca = memref.alloca() : memref<f64>
    %0 = llvm.mlir.undef : f64
    affine.store %0, %alloca[] : memref<f64>
    affine.for %arg0 = 0 to 2600 {
      affine.store %cst_2, %alloca[] : memref<f64>
      affine.for %arg1 = 0 to 2000 {
        %3 = affine.load %alloc[%arg1, %arg0] : memref<2000x2600xf64>
        %4 = arith.mulf %3, %3 : f64
        %5 = affine.load %alloca[] : memref<f64>
        %6 = arith.addf %5, %4 : f64
        affine.store %6, %alloca[] : memref<f64>
      }
      %1 = affine.load %alloca[] : memref<f64>
      %2 = math.sqrt %1 : f64
      affine.store %2, %alloc_3[%arg0, %arg0] : memref<2600x2600xf64>
      affine.for %arg1 = 0 to 2000 {
        %3 = affine.load %alloc[%arg1, %arg0] : memref<2000x2600xf64>
        %4 = arith.divf %3, %2 : f64
        affine.store %4, %alloc_4[%arg1, %arg0] : memref<2000x2600xf64>
      }
      affine.for %arg1 = #map(%arg0) to 2600 {
        affine.store %cst_2, %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        affine.for %arg2 = 0 to 2000 {
          %4 = affine.load %alloc_4[%arg2, %arg0] : memref<2000x2600xf64>
          %5 = affine.load %alloc[%arg2, %arg1] : memref<2000x2600xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = affine.load %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
          %8 = arith.addf %7, %6 : f64
          affine.store %8, %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        }
        %3 = affine.load %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        affine.for %arg2 = 0 to 2000 {
          %4 = affine.load %alloc[%arg2, %arg1] : memref<2000x2600xf64>
          %5 = affine.load %alloc_4[%arg2, %arg0] : memref<2000x2600xf64>
          %6 = arith.mulf %5, %3 : f64
          %7 = arith.subf %4, %6 : f64
          affine.store %7, %alloc[%arg2, %arg1] : memref<2000x2600xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2600 step %c1 {
      scf.for %arg1 = %c0 to %c2600 step %c1 {
        %1 = memref.load %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        %2 = arith.fptosi %1 : f64 to i32
        func.call @print_i32(%2) : (i32) -> ()
      }
    }
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      scf.for %arg1 = %c0 to %c2600 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<2000x2600xf64>
        %2 = arith.fptosi %1 : f64 to i32
        func.call @print_i32(%2) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<2000x2600xf64>
    memref.dealloc %alloc_3 : memref<2600x2600xf64>
    memref.dealloc %alloc_4 : memref<2000x2600xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
