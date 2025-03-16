#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (-d0 + 4000)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4000 = arith.constant 4000 : index
    %cst = arith.constant 4.000000e+03 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 4.000000e+00 : f64
    %cst_3 = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c4000_i32 = arith.constant 4000 : i32
    %alloc = memref.alloc() : memref<4000x4000xf64>
    %alloc_4 = memref.alloc() : memref<4000xf64>
    %alloc_5 = memref.alloc() : memref<4000xf64>
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      memref.store %cst_0, %alloc_4[%arg0] : memref<4000xf64>
      memref.store %cst_0, %alloc_5[%arg0] : memref<4000xf64>
    }
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %1, %c1_i32 : i32
      %3 = arith.index_cast %2 : i32 to index
      scf.for %arg1 = %c0 to %3 step %c1 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.subi %c0_i32, %4 : i32
        %6 = arith.remsi %5, %c4000_i32 : i32
        %7 = arith.sitofp %6 : i32 to f64
        %8 = arith.divf %7, %cst : f64
        %9 = arith.addf %8, %cst_3 : f64
        memref.store %9, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
      scf.for %arg1 = %3 to %c4000 step %c1 {
        memref.store %cst_0, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
      memref.store %cst_3, %alloc[%arg0, %arg0] : memref<4000x4000xf64>
    }
    %alloc_6 = memref.alloc() : memref<4000x4000xf64>
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        memref.store %cst_0, %alloc_6[%arg0, %arg1] : memref<4000x4000xf64>
      }
    }
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        %1 = memref.load %alloc[%arg1, %arg0] : memref<4000x4000xf64>
        scf.for %arg2 = %c0 to %c4000 step %c1 {
          %2 = memref.load %alloc[%arg2, %arg0] : memref<4000x4000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = memref.load %alloc_6[%arg1, %arg2] : memref<4000x4000xf64>
          %5 = arith.addf %4, %3 : f64
          memref.store %5, %alloc_6[%arg1, %arg2] : memref<4000x4000xf64>
        }
      }
    }
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        %1 = memref.load %alloc_6[%arg0, %arg1] : memref<4000x4000xf64>
        memref.store %1, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
    }
    memref.dealloc %alloc_6 : memref<4000x4000xf64>
    call @toggle_gpio() : () -> ()
    %alloca = memref.alloca() : memref<f64>
    %0 = llvm.mlir.undef : f64
    affine.store %0, %alloca[] : memref<f64>
    affine.for %arg0 = 0 to 4000 {
      affine.for %arg1 = 0 to #map(%arg0) {
        %1 = affine.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        affine.store %1, %alloca[] : memref<f64>
        affine.for %arg2 = 0 to #map(%arg1) {
          %5 = affine.load %alloc[%arg0, %arg2] : memref<4000x4000xf64>
          %6 = affine.load %alloc[%arg2, %arg1] : memref<4000x4000xf64>
          %7 = arith.mulf %5, %6 : f64
          %8 = affine.load %alloca[] : memref<f64>
          %9 = arith.subf %8, %7 : f64
          affine.store %9, %alloca[] : memref<f64>
        }
        %2 = affine.load %alloca[] : memref<f64>
        %3 = affine.load %alloc[%arg1, %arg1] : memref<4000x4000xf64>
        %4 = arith.divf %2, %3 : f64
        affine.store %4, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
      affine.for %arg1 = #map(%arg0) to 4000 {
        %1 = affine.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        affine.store %1, %alloca[] : memref<f64>
        affine.for %arg2 = 0 to #map(%arg0) {
          %3 = affine.load %alloc[%arg0, %arg2] : memref<4000x4000xf64>
          %4 = affine.load %alloc[%arg2, %arg1] : memref<4000x4000xf64>
          %5 = arith.mulf %3, %4 : f64
          %6 = affine.load %alloca[] : memref<f64>
          %7 = arith.subf %6, %5 : f64
          affine.store %7, %alloca[] : memref<f64>
        }
        %2 = affine.load %alloca[] : memref<f64>
        affine.store %2, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
    }
    affine.for %arg0 = 0 to 4000 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %1, %c1_i32 : i32
      %3 = arith.sitofp %2 : i32 to f64
      %4 = arith.divf %3, %cst : f64
      %5 = arith.divf %4, %cst_1 : f64
      %6 = arith.addf %5, %cst_2 : f64
      affine.store %6, %alloca[] : memref<f64>
      affine.for %arg1 = 0 to #map(%arg0) {
        %8 = affine.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        %9 = affine.load %alloc_5[%arg1] : memref<4000xf64>
        %10 = arith.mulf %8, %9 : f64
        %11 = affine.load %alloca[] : memref<f64>
        %12 = arith.subf %11, %10 : f64
        affine.store %12, %alloca[] : memref<f64>
      }
      %7 = affine.load %alloca[] : memref<f64>
      affine.store %7, %alloc_5[%arg0] : memref<4000xf64>
    }
    affine.for %arg0 = 0 to 4000 {
      %1 = affine.load %alloc_5[-%arg0 + 3999] : memref<4000xf64>
      affine.store %1, %alloca[] : memref<f64>
      affine.for %arg1 = #map1(%arg0) to 4000 {
        %5 = affine.load %alloc[-%arg0 + 3999, %arg1] : memref<4000x4000xf64>
        %6 = affine.load %alloc_4[%arg1] : memref<4000xf64>
        %7 = arith.mulf %5, %6 : f64
        %8 = affine.load %alloca[] : memref<f64>
        %9 = arith.subf %8, %7 : f64
        affine.store %9, %alloca[] : memref<f64>
      }
      %2 = affine.load %alloca[] : memref<f64>
      %3 = affine.load %alloc[-%arg0 + 3999, -%arg0 + 3999] : memref<4000x4000xf64>
      %4 = arith.divf %2, %3 : f64
      affine.store %4, %alloc_4[-%arg0 + 3999] : memref<4000xf64>
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %1 = memref.load %alloc_4[%arg0] : memref<4000xf64>
      %2 = arith.fptosi %1 : f64 to i32
      func.call @print_i32(%2) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<4000x4000xf64>
    memref.dealloc %alloc_4 : memref<4000xf64>
    memref.dealloc %alloc_5 : memref<4000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
