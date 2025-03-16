#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4000 = arith.constant 4000 : index
    %cst = arith.constant 4.000000e+03 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c4000_i32 = arith.constant 4000 : i32
    %alloc = memref.alloc() : memref<4000x4000xf64>
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.addi %0, %c1_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      scf.for %arg1 = %c0 to %2 step %c1 {
        %3 = arith.index_cast %arg1 : index to i32
        %4 = arith.subi %c0_i32, %3 : i32
        %5 = arith.remsi %4, %c4000_i32 : i32
        %6 = arith.sitofp %5 : i32 to f64
        %7 = arith.divf %6, %cst : f64
        %8 = arith.addf %7, %cst_0 : f64
        memref.store %8, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
      scf.for %arg1 = %2 to %c4000 step %c1 {
        memref.store %cst_1, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
      memref.store %cst_0, %alloc[%arg0, %arg0] : memref<4000x4000xf64>
    }
    %alloc_2 = memref.alloc() : memref<4000x4000xf64>
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        memref.store %cst_1, %alloc_2[%arg0, %arg1] : memref<4000x4000xf64>
      }
    }
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        %0 = memref.load %alloc[%arg1, %arg0] : memref<4000x4000xf64>
        scf.for %arg2 = %c0 to %c4000 step %c1 {
          %1 = memref.load %alloc[%arg2, %arg0] : memref<4000x4000xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = memref.load %alloc_2[%arg1, %arg2] : memref<4000x4000xf64>
          %4 = arith.addf %3, %2 : f64
          memref.store %4, %alloc_2[%arg1, %arg2] : memref<4000x4000xf64>
        }
      }
    }
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        %0 = memref.load %alloc_2[%arg0, %arg1] : memref<4000x4000xf64>
        memref.store %0, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
    }
    memref.dealloc %alloc_2 : memref<4000x4000xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 4000 {
      affine.for %arg1 = 0 to #map(%arg0) {
        affine.for %arg2 = 0 to #map(%arg1) {
          %3 = affine.load %alloc[%arg0, %arg2] : memref<4000x4000xf64>
          %4 = affine.load %alloc[%arg2, %arg1] : memref<4000x4000xf64>
          %5 = arith.mulf %3, %4 : f64
          %6 = affine.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
          %7 = arith.subf %6, %5 : f64
          affine.store %7, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        }
        %0 = affine.load %alloc[%arg1, %arg1] : memref<4000x4000xf64>
        %1 = affine.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        %2 = arith.divf %1, %0 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
      affine.for %arg1 = #map(%arg0) to 4000 {
        affine.for %arg2 = 0 to #map(%arg0) {
          %0 = affine.load %alloc[%arg0, %arg2] : memref<4000x4000xf64>
          %1 = affine.load %alloc[%arg2, %arg1] : memref<4000x4000xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
          %4 = arith.subf %3, %2 : f64
          affine.store %4, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<4000x4000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
