//===- WAMIConvertArith.cpp - Arith to WasmSSA conversion -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion patterns from Arith dialect to the upstream
// WasmSSA dialect.
//
//===----------------------------------------------------------------------===//

#include "WAMI/ConversionPatterns/WAMIConvertArith.h"
#include "WAMI/ConversionPatterns/WAMIConversionUtils.h"

#include "WAMI/WAMIOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"

namespace mlir::wami {

namespace {

//===----------------------------------------------------------------------===//
// Binary Operation Lowering Templates
//===----------------------------------------------------------------------===//

template <typename SrcOp, typename TgtOp>
struct BinaryOpLowering : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<TgtOp>(op, resultType, adaptor.getLhs(),
                                       adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Unary Operation Lowering Templates
//===----------------------------------------------------------------------===//

template <typename SrcOp, typename TgtOp>
struct UnaryOpLowering : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<TgtOp>(op, resultType, adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arithmetic Operation Lowerings
//===----------------------------------------------------------------------===//

// Integer and floating-point add/sub/mul use the same WasmSSA ops
using AddIOpLowering = BinaryOpLowering<arith::AddIOp, wasmssa::AddOp>;
using AddFOpLowering = BinaryOpLowering<arith::AddFOp, wasmssa::AddOp>;
using SubIOpLowering = BinaryOpLowering<arith::SubIOp, wasmssa::SubOp>;
using SubFOpLowering = BinaryOpLowering<arith::SubFOp, wasmssa::SubOp>;
using MulIOpLowering = BinaryOpLowering<arith::MulIOp, wasmssa::MulOp>;
using MulFOpLowering = BinaryOpLowering<arith::MulFOp, wasmssa::MulOp>;

// Division ops
using DivSIOpLowering = BinaryOpLowering<arith::DivSIOp, wasmssa::DivSIOp>;
using DivUIOpLowering = BinaryOpLowering<arith::DivUIOp, wasmssa::DivUIOp>;
using DivFOpLowering = BinaryOpLowering<arith::DivFOp, wasmssa::DivOp>;

// Remainder ops
using RemSIOpLowering = BinaryOpLowering<arith::RemSIOp, wasmssa::RemSIOp>;
using RemUIOpLowering = BinaryOpLowering<arith::RemUIOp, wasmssa::RemUIOp>;

// Bitwise ops
using AndIOpLowering = BinaryOpLowering<arith::AndIOp, wasmssa::AndOp>;
using OrIOpLowering = BinaryOpLowering<arith::OrIOp, wasmssa::OrOp>;
using XOrIOpLowering = BinaryOpLowering<arith::XOrIOp, wasmssa::XOrOp>;

// Min/Max ops (floating-point only in WasmSSA)
// minimumf/maximumf - IEEE 754-2019 minimum/maximum (propagate NaN)
// WebAssembly min/max also propagate NaN, so direct mapping is correct.
// Reference: https://github.com/WebAssembly/design/issues/1548
using MinimumFOpLowering = BinaryOpLowering<arith::MinimumFOp, wasmssa::MinOp>;
using MaximumFOpLowering = BinaryOpLowering<arith::MaximumFOp, wasmssa::MaxOp>;

// minnumf/maxnumf - IEEE 754-2008 minNum/maxNum semantics (return non-NaN)
// WebAssembly min/max propagate NaN, so we need conditional logic.
// Logic: if lhs is NaN, return rhs; else if rhs is NaN, return lhs; else
// min/max
struct MinNumFOpLowering : public OpConversionPattern<arith::MinNumFOp> {
  using OpConversionPattern<arith::MinNumFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MinNumFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type resultType = getTypeConverter()->convertType(op.getType());

    // Check if operands are NaN (a != a is true iff a is NaN)
    Value lhsIsNaN = wasmssa::NeOp::create(rewriter, loc, lhs, lhs);
    Value rhsIsNaN = wasmssa::NeOp::create(rewriter, loc, rhs, rhs);

    // Compute min using WebAssembly min
    Value minResult =
        wasmssa::MinOp::create(rewriter, loc, resultType, lhs, rhs);

    // If rhs is NaN, use lhs; otherwise use min result
    Value step1 =
        SelectOp::create(rewriter, loc, resultType, lhs, minResult, rhsIsNaN);
    // If lhs is NaN, use rhs; otherwise use step1
    Value result =
        SelectOp::create(rewriter, loc, resultType, rhs, step1, lhsIsNaN);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MaxNumFOpLowering : public OpConversionPattern<arith::MaxNumFOp> {
  using OpConversionPattern<arith::MaxNumFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MaxNumFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type resultType = getTypeConverter()->convertType(op.getType());

    // Check if operands are NaN (a != a is true iff a is NaN)
    Value lhsIsNaN = wasmssa::NeOp::create(rewriter, loc, lhs, lhs);
    Value rhsIsNaN = wasmssa::NeOp::create(rewriter, loc, rhs, rhs);

    // Compute max using WebAssembly max
    Value maxResult =
        wasmssa::MaxOp::create(rewriter, loc, resultType, lhs, rhs);

    // If rhs is NaN, use lhs; otherwise use max result
    Value step1 =
        SelectOp::create(rewriter, loc, resultType, lhs, maxResult, rhsIsNaN);
    // If lhs is NaN, use rhs; otherwise use step1
    Value result =
        SelectOp::create(rewriter, loc, resultType, rhs, step1, lhsIsNaN);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// arith.select -> wami.select
struct SelectOpLowering : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());

    // Ensure condition is i32 (WebAssembly select requires i32 condition)
    Value cond =
        ensureI32Condition(adaptor.getCondition(), op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<SelectOp>(
        op, resultType, adaptor.getTrueValue(), adaptor.getFalseValue(), cond);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant Operation Lowering
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Attribute value = adaptor.getValue();

    // Convert index type to i32 (WebAssembly uses 32-bit addresses)
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      if (intAttr.getType().isIndex()) {
        APInt indexValue = intAttr.getValue();
        if (!indexValue.isSignedIntN(32))
          return rewriter.notifyMatchFailure(
              op, "index constant exceeds 32-bit range");
        auto i32Type = IntegerType::get(op.getContext(), 32);
        value = IntegerAttr::get(i32Type, indexValue.sextOrTrunc(32));
      }
    }

    // Convert bool to i32 (WebAssembly represents booleans as i32)
    if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
      auto i32Type = IntegerType::get(op.getContext(), 32);
      value = IntegerAttr::get(i32Type, boolAttr.getValue() ? 1 : 0);
    }

    auto typedAttr = cast<TypedAttr>(value);
    rewriter.replaceOpWithNewOp<wasmssa::ConstOp>(op, typedAttr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Comparison Operation Lowerings
//===----------------------------------------------------------------------===//

struct CmpIOpLowering : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto predicate = op.getPredicate();

    switch (predicate) {
    case arith::CmpIPredicate::eq:
      rewriter.replaceOpWithNewOp<wasmssa::EqOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::ne:
      rewriter.replaceOpWithNewOp<wasmssa::NeOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<wasmssa::LtSIOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<wasmssa::LtUIOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::sle:
      rewriter.replaceOpWithNewOp<wasmssa::LeSIOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::ule:
      rewriter.replaceOpWithNewOp<wasmssa::LeUIOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<wasmssa::GtSIOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<wasmssa::GtUIOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::sge:
      rewriter.replaceOpWithNewOp<wasmssa::GeSIOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpIPredicate::uge:
      rewriter.replaceOpWithNewOp<wasmssa::GeUIOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    default:
      return rewriter.notifyMatchFailure(op,
                                         "unsupported comparison predicate");
    }
    return success();
  }
};

struct CmpFOpLowering : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto predicate = op.getPredicate();

    // Handle ordered comparisons (most common case)
    switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      rewriter.replaceOpWithNewOp<wasmssa::EqOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::ONE:
      rewriter.replaceOpWithNewOp<wasmssa::NeOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::OLT:
      rewriter.replaceOpWithNewOp<wasmssa::LtOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::OLE:
      rewriter.replaceOpWithNewOp<wasmssa::LeOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::OGT:
      rewriter.replaceOpWithNewOp<wasmssa::GtOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::OGE:
      rewriter.replaceOpWithNewOp<wasmssa::GeOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      break;
    default:
      return rewriter.notifyMatchFailure(op,
                                         "unsupported comparison predicate");
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Extension and Truncation Operation Lowerings
//===----------------------------------------------------------------------===//

struct ExtUIOpLowering : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern<arith::ExtUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getIn().getType();
    Type dstType = getTypeConverter()->convertType(op.getResult().getType());

    // If source and dest types are the same after conversion (e.g., i1→i32 when
    // both become i32), just replace with the input value
    if (srcType == dstType) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }

    // i32 to i64 extension
    if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<wasmssa::ExtendUI32Op>(op, dstType,
                                                         adaptor.getIn());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported extension");
  }
};

struct ExtSIOpLowering : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern<arith::ExtSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getIn().getType();
    Type dstType = getTypeConverter()->convertType(op.getResult().getType());

    // If source and dest types are the same after conversion, replace with
    // input
    if (srcType == dstType) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }

    // i32 to i64 extension
    if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<wasmssa::ExtendSI32Op>(op, dstType,
                                                         adaptor.getIn());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported extension");
  }
};

struct TruncIOpLowering : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getIn().getType();
    Type dstType = getTypeConverter()->convertType(op.getResult().getType());

    // If source and dest types are the same after conversion, replace with
    // input
    if (srcType == dstType) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }

    // i64 to i32 truncation
    if (srcType.isInteger(64) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<wasmssa::WrapOp>(op, dstType,
                                                   adaptor.getIn());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported truncation");
  }
};

//===----------------------------------------------------------------------===//
// Index Cast Operation Lowering
//===----------------------------------------------------------------------===//

struct IndexCastOpLowering : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getIn().getType();
    Type dstType = getTypeConverter()->convertType(op.getResult().getType());

    // After type conversion, index becomes i32.
    // If source and dest types are the same, this is a no-op.
    if (srcType == dstType) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }

    // i32 to i64 (index_cast i32 to index when index is 64-bit, which is rare)
    if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<wasmssa::ExtendSI32Op>(op, dstType,
                                                         adaptor.getIn());
      return success();
    }

    // i64 to i32 (index_cast index to i32 when index is 64-bit)
    if (srcType.isInteger(64) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<wasmssa::WrapOp>(op, dstType,
                                                   adaptor.getIn());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported index cast");
  }
};

struct IndexCastUIOpLowering
    : public OpConversionPattern<arith::IndexCastUIOp> {
  using OpConversionPattern<arith::IndexCastUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getIn().getType();
    Type dstType = getTypeConverter()->convertType(op.getResult().getType());

    // After type conversion, index becomes i32.
    // If source and dest types are the same, this is a no-op.
    if (srcType == dstType) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }

    // i32 to i64 (unsigned extension)
    if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<wasmssa::ExtendUI32Op>(op, dstType,
                                                         adaptor.getIn());
      return success();
    }

    // i64 to i32 (wrap)
    if (srcType.isInteger(64) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<wasmssa::WrapOp>(op, dstType,
                                                   adaptor.getIn());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported index cast");
  }
};

//===----------------------------------------------------------------------===//
// Shift Operation Lowerings
//===----------------------------------------------------------------------===//

struct ShLIOpLowering : public OpConversionPattern<arith::ShLIOp> {
  using OpConversionPattern<arith::ShLIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<wasmssa::ShLOp>(
        op, resultType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ShRSIOpLowering : public OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern<arith::ShRSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<wasmssa::ShRSOp>(
        op, resultType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ShRUIOpLowering : public OpConversionPattern<arith::ShRUIOp> {
  using OpConversionPattern<arith::ShRUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<wasmssa::ShRUOp>(
        op, resultType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Float/Int Conversion Operation Lowerings
//===----------------------------------------------------------------------===//

// arith.sitofp: signed int to float
// Maps to wasmssa.convert_s (WebAssembly f32.convert_i32_s, etc.)
struct SIToFPOpLowering : public OpConversionPattern<arith::SIToFPOp> {
  using OpConversionPattern<arith::SIToFPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<wasmssa::ConvertSOp>(op, resultType,
                                                     adaptor.getIn());
    return success();
  }
};

// arith.uitofp: unsigned int to float
// Maps to wasmssa.convert_u (WebAssembly f32.convert_i32_u, etc.)
struct UIToFPOpLowering : public OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern<arith::UIToFPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<wasmssa::ConvertUOp>(op, resultType,
                                                     adaptor.getIn());
    return success();
  }
};

// arith.extf: f32 to f64 (promote)
// Maps to wasmssa.promote (WebAssembly f64.promote_f32)
struct ExtFOpLowering : public OpConversionPattern<arith::ExtFOp> {
  using OpConversionPattern<arith::ExtFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getIn().getType();
    Type dstType = getTypeConverter()->convertType(op.getResult().getType());

    // f32 to f64 promotion
    if (srcType.isF32() && dstType.isF64()) {
      rewriter.replaceOpWithNewOp<wasmssa::PromoteOp>(op, dstType,
                                                      adaptor.getIn());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported float extension");
  }
};

// arith.truncf: f64 to f32 (demote)
// Maps to wasmssa.demote (WebAssembly f32.demote_f64)
struct TruncFOpLowering : public OpConversionPattern<arith::TruncFOp> {
  using OpConversionPattern<arith::TruncFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getIn().getType();
    Type dstType = getTypeConverter()->convertType(op.getResult().getType());

    // f64 to f32 demotion
    if (srcType.isF64() && dstType.isF32()) {
      rewriter.replaceOpWithNewOp<wasmssa::DemoteOp>(op, dstType,
                                                     adaptor.getIn());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported float truncation");
  }
};

// arith.negf: float negation
// Maps to wasmssa.neg (WebAssembly f32.neg, f64.neg)
struct NegFOpLowering : public OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern<arith::NegFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<wasmssa::NegOp>(op, resultType,
                                                adaptor.getOperand());
    return success();
  }
};

// arith.fptosi: float to signed int
// Maps to wami.trunc_s (WebAssembly i32.trunc_f32_s, i64.trunc_f64_s, etc.)
struct FPToSIOpLowering : public OpConversionPattern<arith::FPToSIOp> {
  using OpConversionPattern<arith::FPToSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::FPToSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<TruncSOp>(op, resultType, adaptor.getIn());
    return success();
  }
};

// arith.fptoui: float to unsigned int
// Maps to wami.trunc_u (WebAssembly i32.trunc_f32_u, i64.trunc_f64_u, etc.)
struct FPToUIOpLowering : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern<arith::FPToUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::FPToUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<TruncUOp>(op, resultType, adaptor.getIn());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateWAMIConvertArithPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  // Arithmetic operations
  patterns.add<
      AddIOpLowering, AddFOpLowering, SubIOpLowering, SubFOpLowering,
      MulIOpLowering, MulFOpLowering, DivSIOpLowering, DivUIOpLowering,
      DivFOpLowering, RemSIOpLowering, RemUIOpLowering, AndIOpLowering,
      OrIOpLowering, XOrIOpLowering, MinimumFOpLowering, MaximumFOpLowering,
      MinNumFOpLowering, MaxNumFOpLowering, ConstantOpLowering, CmpIOpLowering,
      CmpFOpLowering, ShLIOpLowering, ShRSIOpLowering, ShRUIOpLowering,
      ExtUIOpLowering, ExtSIOpLowering, TruncIOpLowering, IndexCastOpLowering,
      IndexCastUIOpLowering, SelectOpLowering, SIToFPOpLowering,
      UIToFPOpLowering, ExtFOpLowering, TruncFOpLowering, FPToSIOpLowering,
      FPToUIOpLowering, NegFOpLowering>(typeConverter, context);
}

} // namespace mlir::wami
