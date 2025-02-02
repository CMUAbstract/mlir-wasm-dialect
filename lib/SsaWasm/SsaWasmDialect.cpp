//===- SsaWasmDialect.cpp - SsaWasm dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SsaWasm/SsaWasmDialect.h"
#include "SsaWasm/SsaWasmOps.h"
#include "SsaWasm/SsaWasmTypes.h"

using namespace mlir;
using namespace mlir::ssawasm;

#include "SsaWasm/SsaWasmDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SsaWasm dialect.
//===----------------------------------------------------------------------===//

void SsaWasmDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SsaWasm/SsaWasmOps.cpp.inc"
      >();
  registerTypes();
}
