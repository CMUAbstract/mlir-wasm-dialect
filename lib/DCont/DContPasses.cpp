
//===- DContPasses.cpp - DCont passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "DCont/DContPasses.h"

using namespace std;

namespace mlir::dcont {
#define GEN_PASS_DEF_CONVERTTOSSAWASM
#define GEN_PASS_DEF_INTRODUCELOCALS
#define GEN_PASS_DEF_CONVERTSSAWASMTOWASM
#define GEN_PASS_DEF_SSAWASMDATATOLOCAL
#include "DCont/DContPasses.h.inc"

} // namespace mlir::dcont