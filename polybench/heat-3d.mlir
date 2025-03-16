module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c200 = arith.constant 200 : index
    %cst = arith.constant 2.000000e+02 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant 1.250000e-01 : f64
    %cst_2 = arith.constant 1.000000e+01 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c200_i32 = arith.constant 200 : i32
    %alloc = memref.alloc() : memref<200x200x200xf64>
    %alloc_3 = memref.alloc() : memref<200x200x200xf64>
    scf.for %arg0 = %c0 to %c200 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c200 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.addi %0, %1 : i32
        scf.for %arg2 = %c0 to %c200 step %c1 {
          %3 = arith.index_cast %arg2 : index to i32
          %4 = arith.subi %c200_i32, %3 : i32
          %5 = arith.addi %2, %4 : i32
          %6 = arith.sitofp %5 : i32 to f64
          %7 = arith.mulf %6, %cst_2 : f64
          %8 = arith.divf %7, %cst : f64
          memref.store %8, %alloc_3[%arg0, %arg1, %arg2] : memref<200x200x200xf64>
          memref.store %8, %alloc[%arg0, %arg1, %arg2] : memref<200x200x200xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 1 to 1001 {
      affine.for %arg1 = 1 to 199 {
        affine.for %arg2 = 1 to 199 {
          affine.for %arg3 = 1 to 199 {
            %0 = affine.load %alloc[%arg1 + 1, %arg2, %arg3] : memref<200x200x200xf64>
            %1 = affine.load %alloc[%arg1, %arg2, %arg3] : memref<200x200x200xf64>
            %2 = arith.mulf %1, %cst_0 : f64
            %3 = arith.subf %0, %2 : f64
            %4 = affine.load %alloc[%arg1 - 1, %arg2, %arg3] : memref<200x200x200xf64>
            %5 = arith.addf %3, %4 : f64
            %6 = arith.mulf %5, %cst_1 : f64
            %7 = affine.load %alloc[%arg1, %arg2 + 1, %arg3] : memref<200x200x200xf64>
            %8 = arith.subf %7, %2 : f64
            %9 = affine.load %alloc[%arg1, %arg2 - 1, %arg3] : memref<200x200x200xf64>
            %10 = arith.addf %8, %9 : f64
            %11 = arith.mulf %10, %cst_1 : f64
            %12 = arith.addf %6, %11 : f64
            %13 = affine.load %alloc[%arg1, %arg2, %arg3 + 1] : memref<200x200x200xf64>
            %14 = arith.subf %13, %2 : f64
            %15 = affine.load %alloc[%arg1, %arg2, %arg3 - 1] : memref<200x200x200xf64>
            %16 = arith.addf %14, %15 : f64
            %17 = arith.mulf %16, %cst_1 : f64
            %18 = arith.addf %12, %17 : f64
            %19 = arith.addf %18, %1 : f64
            affine.store %19, %alloc_3[%arg1, %arg2, %arg3] : memref<200x200x200xf64>
          }
        }
      }
      affine.for %arg1 = 1 to 199 {
        affine.for %arg2 = 1 to 199 {
          affine.for %arg3 = 1 to 199 {
            %0 = affine.load %alloc_3[%arg1 + 1, %arg2, %arg3] : memref<200x200x200xf64>
            %1 = affine.load %alloc_3[%arg1, %arg2, %arg3] : memref<200x200x200xf64>
            %2 = arith.mulf %1, %cst_0 : f64
            %3 = arith.subf %0, %2 : f64
            %4 = affine.load %alloc_3[%arg1 - 1, %arg2, %arg3] : memref<200x200x200xf64>
            %5 = arith.addf %3, %4 : f64
            %6 = arith.mulf %5, %cst_1 : f64
            %7 = affine.load %alloc_3[%arg1, %arg2 + 1, %arg3] : memref<200x200x200xf64>
            %8 = arith.subf %7, %2 : f64
            %9 = affine.load %alloc_3[%arg1, %arg2 - 1, %arg3] : memref<200x200x200xf64>
            %10 = arith.addf %8, %9 : f64
            %11 = arith.mulf %10, %cst_1 : f64
            %12 = arith.addf %6, %11 : f64
            %13 = affine.load %alloc_3[%arg1, %arg2, %arg3 + 1] : memref<200x200x200xf64>
            %14 = arith.subf %13, %2 : f64
            %15 = affine.load %alloc_3[%arg1, %arg2, %arg3 - 1] : memref<200x200x200xf64>
            %16 = arith.addf %14, %15 : f64
            %17 = arith.mulf %16, %cst_1 : f64
            %18 = arith.addf %12, %17 : f64
            %19 = arith.addf %18, %1 : f64
            affine.store %19, %alloc[%arg1, %arg2, %arg3] : memref<200x200x200xf64>
          }
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c200 step %c1 {
      scf.for %arg1 = %c0 to %c200 step %c1 {
        scf.for %arg2 = %c0 to %c200 step %c1 {
          %0 = memref.load %alloc[%arg0, %arg1, %arg2] : memref<200x200x200xf64>
          %1 = arith.fptosi %0 : f64 to i32
          func.call @print_i32(%1) : (i32) -> ()
        }
      }
    }
    memref.dealloc %alloc : memref<200x200x200xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
