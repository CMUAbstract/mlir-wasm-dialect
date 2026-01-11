//===- WAMIPasses.cpp - WAMI dialect passes ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements passes for the WAMI dialect and conversion passes
// to/from the upstream WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/WAMIPasses.h"
#include "WAMI/ConversionPatterns/WAMIConvertArith.h"
#include "WAMI/ConversionPatterns/WAMIConvertFunc.h"
#include "WAMI/ConversionPatterns/WAMIConvertMemref.h"
#include "WAMI/ConversionPatterns/WAMIConvertScf.h"
#include "WAMI/WAMIDialect.h"
#include "WAMI/WAMITypeConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::wami {

#define GEN_PASS_DEF_WAMICONVERTARITH
#define GEN_PASS_DEF_WAMICONVERTFUNC
#define GEN_PASS_DEF_WAMICONVERTMEMREF
#define GEN_PASS_DEF_WAMICONVERTSCF
#include "WAMI/WAMIPasses.h.inc"

//===----------------------------------------------------------------------===//
// WAMIConvertArith Pass
//===----------------------------------------------------------------------===//

class WAMIConvertArith : public impl::WAMIConvertArithBase<WAMIConvertArith> {
public:
  using impl::WAMIConvertArithBase<WAMIConvertArith>::WAMIConvertArithBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // WasmSSA and WAMI dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();
    target.addLegalDialect<WAMIDialect>();

    // Arith dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<arith::ArithDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertArithPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// WAMIConvertFunc Pass
//===----------------------------------------------------------------------===//

class WAMIConvertFunc : public impl::WAMIConvertFuncBase<WAMIConvertFunc> {
public:
  using impl::WAMIConvertFuncBase<WAMIConvertFunc>::WAMIConvertFuncBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // WasmSSA dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();

    // Func dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<func::FuncDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertFuncPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// WAMIConvertScf Pass
//===----------------------------------------------------------------------===//

class WAMIConvertScf : public impl::WAMIConvertScfBase<WAMIConvertScf> {
public:
  using impl::WAMIConvertScfBase<WAMIConvertScf>::WAMIConvertScfBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // WasmSSA dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();

    // SCF dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<scf::SCFDialect>();

    // Arith and Func dialects are legal (may be used in loop bodies)
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertScfPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// WAMIConvertMemref Pass
//===----------------------------------------------------------------------===//

class WAMIConvertMemref
    : public impl::WAMIConvertMemrefBase<WAMIConvertMemref> {
public:
  using impl::WAMIConvertMemrefBase<WAMIConvertMemref>::WAMIConvertMemrefBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *context = module.getContext();
    WAMITypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // Analyze module to assign base addresses to globals
    WAMIBaseAddressAnalysis baseAddressAnalysis(module);

    // WasmSSA and WAMI dialect operations are legal
    target.addLegalDialect<wasmssa::WasmSSADialect>();
    target.addLegalDialect<WAMIDialect>();

    // MemRef dialect operations are illegal (we want to convert them)
    target.addIllegalDialect<memref::MemRefDialect>();

    // Arith dialect is legal (used for address computation)
    target.addLegalDialect<arith::ArithDialect>();

    // Allow unrealized conversion casts for type mismatches
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateWAMIConvertMemrefPatterns(typeConverter, patterns,
                                      baseAddressAnalysis);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::wami
