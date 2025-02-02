//===- wasm-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Intermittent/IntermittentDialect.h"
#include "Intermittent/IntermittentPasses.h"
#include "SsaWasm/SsaWasmDialect.h"
#include "SsaWasm/SsaWasmPasses.h"
#include "Wasm/WasmDialect.h"
#include "Wasm/WasmPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::wasm::registerPasses();
  mlir::intermittent::registerPasses();
  mlir::ssawasm::registerPasses();

  mlir::DialectRegistry registry;
  registry
      .insert<mlir::wasm::WasmDialect, mlir::intermittent::IntermittentDialect,
              mlir::arith::ArithDialect, mlir::func::FuncDialect,
              mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
              mlir::affine::AffineDialect, mlir::ssawasm::SsaWasmDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Wasm optimizer driver\n", registry));
}
