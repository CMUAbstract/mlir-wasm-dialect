//===- wasm-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DCont/DContDialect.h"
#include "WAMI/WAMIDialect.h"
#include "WAMI/WAMIPasses.h"
#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackOps.h"
#include "wasmstack/WasmStackPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::wami::registerPasses();
  mlir::wasmstack::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect, mlir::dcont::DContDialect,
                  mlir::math::MathDialect, mlir::wasmssa::WasmSSADialect,
                  mlir::wami::WAMIDialect, mlir::wasmstack::WasmStackDialect,
                  mlir::LLVM::LLVMDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Wasm optimizer driver\n", registry));
}
