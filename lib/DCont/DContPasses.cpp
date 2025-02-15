
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
#include "Intermittent/IntermittentOps.h"
#include "SsaWasm/SsaWasmDialect.h"
#include "SsaWasm/SsaWasmOps.h"
#include "SsaWasm/SsaWasmTypes.h"

using namespace std;

namespace mlir::dcont {
#define GEN_PASS_DEF_CONVERTINTERMITTENTTODCONT
#define GEN_PASS_DEF_INTRODUCEMAINFUNCTION
#define GEN_PASS_DEF_CONVERTDCONTTOSSAWASM
#include "DCont/DContPasses.h.inc"

// Intermittent to DCont passes

class IntroduceMainFunction
    : public impl::IntroduceMainFunctionBase<IntroduceMainFunction> {
  using impl::IntroduceMainFunctionBase<
      IntroduceMainFunction>::IntroduceMainFunctionBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    Location loc = module.getLoc();

    // introduce main function
    rewriter.setInsertionPoint(module.getBody(), module.getBody()->begin());
    auto mainFuncOp = rewriter.create<func::FuncOp>(
        loc, "main", rewriter.getFunctionType({}, {}));
    auto &entryRegion = mainFuncOp.getBody();
    auto *entryBlock = rewriter.createBlock(&entryRegion);
    rewriter.setInsertionPointToEnd(entryBlock);

    // run the initial task
    StringRef initialTaskName = "task1";
    auto contType = ContType::get(context, StringAttr::get(context, "ct"));
    // each task will be converted to a function
    // that takes a continuation as an argument
    auto handle =
        rewriter
            .create<NewOp>(loc, contType,
                           FlatSymbolRefAttr::get(context, initialTaskName))
            .getResult();
    auto nullCont = rewriter.create<NullContOp>(loc, contType).getResult();
    rewriter.create<ResumeSwitchOp>(loc,
                                    /*return continuation type=*/contType,
                                    /*returned results*/ TypeRange{},
                                    /*continuation*/ handle,
                                    /*arguments*/ ValueRange{nullCont});
    rewriter.create<func::ReturnOp>(loc, TypeRange{}, ValueRange{});
  }
};

namespace {
struct IdempotentTaskOpLowering
    : public OpConversionPattern<intermittent::IdempotentTaskOp> {
  using OpConversionPattern<
      intermittent::IdempotentTaskOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(intermittent::IdempotentTaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // IdempotentTaskOp is lowered to a function that takes a continuation as an
    // argument
    MLIRContext *context = op.getContext();
    auto contType = ContType::get(context, StringAttr::get(context, "ct"));
    auto funcOp = rewriter.create<func::FuncOp>(
        op.getLoc(), op.getSymName(),
        rewriter.getFunctionType(/*inputs=*/{contType}, /*results=*/{}));

    // Create an continuation argument to the function entry block
    TypeConverter::SignatureConversion signatureConversion(1);
    signatureConversion.addInputs(0, contType);
    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *getTypeConverter(),
                                           &signatureConversion))) {
      return failure();
    }
    // Inline the region into the function body
    rewriter.inlineRegionBefore(op.getRegion(), funcOp.getBody(),
                                funcOp.getBody().end());

    // Replace the original op
    rewriter.eraseOp(op);

    return success();
  }
};

struct TransitionToOpLowering
    : public OpConversionPattern<intermittent::TransitionToOp> {
  using OpConversionPattern<intermittent::TransitionToOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(intermittent::TransitionToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Save nonvolatile variables here
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    auto nextTaskName = op.getNextTask().str();
    auto contType = ContType::get(context, StringAttr::get(context, "ct"));
    auto handle = rewriter.create<NewOp>(
        loc, contType, FlatSymbolRefAttr::get(context, nextTaskName));
    rewriter.create<SwitchOp>(loc,
                              /*returedCont=*/contType,
                              /*results=*/TypeRange{},
                              /*cont=*/handle,
                              /*args=*/ValueRange{});
    rewriter.create<func::ReturnOp>(loc, TypeRange{}, ValueRange{});
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace
class ConvertIntermittentToDCont
    : public impl::ConvertIntermittentToDContBase<ConvertIntermittentToDCont> {
  using impl::ConvertIntermittentToDContBase<
      ConvertIntermittentToDCont>::ConvertIntermittentToDContBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<IdempotentTaskOpLowering, TransitionToOpLowering>(context);

    ConversionTarget target(getContext());
    target.addLegalDialect<dcont::DContDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalOp<intermittent::IdempotentTaskOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
namespace {

struct DContToSsaWasmTypeConverter : public TypeConverter {
  DContToSsaWasmTypeConverter(MLIRContext *ctx) {
    addConversion([ctx](ContType type) -> Type {
      return ssawasm::WasmContinuationType::get(ctx, type.getFunctionName());
    });
  }
};

struct NewOpLowering : public OpConversionPattern<NewOp> {
  using OpConversionPattern<NewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ssawasm::ContNewOp>(
        op, getTypeConverter()->convertType(op.getCont().getType()),
        adaptor.getFunctionName());
    return success();
  }
};

struct NullContOpLowering : public OpConversionPattern<NullContOp> {
  using OpConversionPattern<NullContOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NullContOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ssawasm::ContNullOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()));
    return success();
  }
};

struct ResumeSwitchOpLowering : public OpConversionPattern<ResumeSwitchOp> {
  using OpConversionPattern<ResumeSwitchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ResumeSwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ssawasm::ResumeSwitchOp>(
        op,
        /*returnedCont=*/
        getTypeConverter()->convertType(op.getReturnedCont().getType()),
        /*results=*/op.getResults().getType(),
        /*tag=*/rewriter.getStringAttr("yield"),
        /*cont=*/adaptor.getCont(),
        /*args=*/adaptor.getArgs());
    return success();
  }
};

struct SwitchOpLowering : public OpConversionPattern<SwitchOp> {
  using OpConversionPattern<SwitchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ssawasm::SwitchOp>(
        op,
        /*returnedCont=*/
        getTypeConverter()->convertType(op.getReturnedCont().getType()),
        /*results=*/op.getResults().getType(),
        /*cont=*/adaptor.getCont(),
        /*args=*/adaptor.getArgs());
    return success();
  }
};
} // namespace

class ConvertDContToSsaWasm
    : public impl::ConvertDContToSsaWasmBase<ConvertDContToSsaWasm> {
  using impl::ConvertDContToSsaWasmBase<
      ConvertDContToSsaWasm>::ConvertDContToSsaWasmBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    IRRewriter rewriter(context);
    rewriter.setInsertionPoint(module.getBody(), module.getBody()->begin());
    rewriter.create<ssawasm::TagOp>(module.getLoc(),
                                    rewriter.getStringAttr("yield"));
    rewriter.create<ssawasm::RecContFuncDeclOp>(module.getLoc(),
                                                rewriter.getStringAttr("ft"),
                                                rewriter.getStringAttr("ct"));

    RewritePatternSet patterns(context);
    DContToSsaWasmTypeConverter typeConverter(context);
    patterns.add<NewOpLowering, NullContOpLowering, ResumeSwitchOpLowering,
                 SwitchOpLowering>(typeConverter, context);
    // TODO: Support ResumeOp

    ConversionTarget target(getContext());
    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<DContDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::dcont