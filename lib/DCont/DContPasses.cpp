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
      return ssawasm::WasmContinuationType::get(ctx, type.getId());
    });
    addConversion([ctx](IntegerType type) -> Type {
      auto width = type.getWidth();
      if (width == 1) {
        return ssawasm::WasmIntegerType::get(ctx, 32);
      }
      return ssawasm::WasmIntegerType::get(ctx, width);
    });
    addConversion([ctx](IndexType type) -> Type {
      return ssawasm::WasmIntegerType::get(ctx, 32);
    });
    addConversion([ctx](StorageType type) -> Type {
      return ssawasm::WasmContinuationType::get(ctx, type.getId());
    });
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });

    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });
  }
};

struct NewOpLowering : public OpConversionPattern<NewOp> {
  using OpConversionPattern<NewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcRef =
        rewriter
            .create<ssawasm::FuncRefOp>(
                op->getLoc(), ssawasm::WasmRefType::get(op.getContext()),
                adaptor.getFunctionTypeName())
            .getResult();
    rewriter.replaceOpWithNewOp<ssawasm::ContNewOp>(
        op, getTypeConverter()->convertType(op.getCont().getType()), funcRef);
    return success();
  }
};

struct NullContOpLowering : public OpConversionPattern<NullContOp> {
  using OpConversionPattern<NullContOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NullContOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ssawasm::NullContRefOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()));
    return success();
  }
};

// FIXME
// struct ResumeSwitchOpLowering : public OpConversionPattern<ResumeSwitchOp> {
//   using OpConversionPattern<ResumeSwitchOp>::OpConversionPattern;
//
//   LogicalResult
//   matchAndRewrite(ResumeSwitchOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     rewriter.replaceOpWithNewOp<ssawasm::ResumeSwitchOp>(
//         op,
//         /*returnedCont=*/
//         getTypeConverter()->convertType(op.getReturnedCont().getType()),
//         /*results=*/op.getResults().getType(),
//         /*tag=*/rewriter.getStringAttr("yield"),
//         /*cont=*/adaptor.getCont(),
//         /*args=*/adaptor.getArgs());
//     return success();
//   }
// };

// struct SwitchOpLowering : public OpConversionPattern<SwitchOp> {
//   using OpConversionPattern<SwitchOp>::OpConversionPattern;
//
//   LogicalResult
//   matchAndRewrite(SwitchOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     rewriter.replaceOpWithNewOp<ssawasm::SwitchOp>(
//         op,
//         /*returnedCont=*/
//         getTypeConverter()->convertType(op.getReturnedCont().getType()),
//         /*results=*/op.getResults().getType(),
//         /*tag=*/rewriter.getStringAttr("yield"),
//         /*cont=*/adaptor.getCont(),
//         /*args=*/adaptor.getArgs());
//     return success();
//   }
// };

struct ResumeOpLowering : public OpConversionPattern<ResumeOp> {
  using OpConversionPattern<ResumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ResumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // overall structure:
    // block {
    //     ^outerEntryBlock
    //  	     tempBranch to innerEntryBlock1
    //     block {
    //     ^innerEntryBlock1
    // 	        ssawasm.resume(on_yield:^innerExitBlock,fallback:^innerEntryBlock2)
    //     ^innerEntryBlock2
    //        	branch ^outerExitBlock
    //     }
    //     ^innerExitBlock
    // 	        handler
    // 	        exit
    // }
    //     ^outerExitBlock
    // 	        exit
    //

    auto blockBlockOp = rewriter.create<ssawasm::BlockBlockOp>(loc);
    auto [outerEntryBlock, innerEntryBlock2, innerExitBlock, outerExitBlock] =
        blockBlockOp.initialize(rewriter);
    auto innerEntryBlock1 =
        rewriter.splitBlock(innerEntryBlock2, innerEntryBlock2->begin());

    rewriter.setInsertionPointToEnd(outerEntryBlock);
    rewriter.create<ssawasm::TempBranchOp>(loc, innerEntryBlock1);

    rewriter.setInsertionPointToEnd(innerEntryBlock1);
    rewriter.create<ssawasm::ResumeOp>(loc,
                                       /*results=*/op.getResults().getType(),
                                       /*tag=*/rewriter.getStringAttr("yield"),
                                       /*cont=*/adaptor.getCont(),
                                       /*args=*/adaptor.getArgs(),
                                       /*on_yield=*/innerExitBlock,
                                       /*fallback=*/innerEntryBlock2);

    rewriter.setInsertionPointToEnd(innerEntryBlock2);
    rewriter.create<ssawasm::BlockBlockBranchOp>(loc, outerExitBlock);

    rewriter.setInsertionPointToEnd(innerExitBlock);
    // Copy block arguments to the innerExitBlock with type conversion
    SmallVector<Value> newArguments;
    if (!op.getSuspendHandler().empty()) { // Check if region has blocks
      for (auto arg : op.getSuspendHandler().front().getArguments()) {
        Type convertedType = getTypeConverter()->convertType(arg.getType());
        newArguments.push_back(
            innerExitBlock->addArgument(convertedType, arg.getLoc()));
      }

      // Inline handler operations into innerExitBlock, except the terminator
      Block &handlerBlock = op.getSuspendHandler().front();
      auto *terminator = handlerBlock.getTerminator();
      rewriter.inlineBlockBefore(&handlerBlock, innerExitBlock,
                                 innerExitBlock->begin(), newArguments);
      rewriter.eraseOp(terminator);

      // block parameters are in stack, so ideally we don't necessarily need to
      // store them in locals, but we do it for now for simplicity
      // TODO: We should fix this in the future
      rewriter.setInsertionPointToStart(innerExitBlock);
      for (auto it = innerExitBlock->getArguments().rbegin();
           it != innerExitBlock->getArguments().rend(); ++it) {
        auto arg = *it;
        auto localOp =
            rewriter.create<ssawasm::LocalOp>(op.getLoc(), arg.getType(), arg);
        rewriter.replaceAllUsesExcept(arg, localOp.getResult(), localOp);
      }
    }

    rewriter.setInsertionPointToEnd(innerExitBlock);
    rewriter.create<ssawasm::TempBranchOp>(op.getLoc(), outerExitBlock);

    rewriter.setInsertionPointToEnd(outerExitBlock);
    rewriter.create<ssawasm::BlockBlockTerminatorOp>(loc);

    rewriter.eraseOp(op);

    return success();
  }
};

struct SuspendOpLowering : public OpConversionPattern<SuspendOp> {
  using OpConversionPattern<SuspendOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SuspendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<ssawasm::SuspendOp>(
        op,
        /*results=*/op.getResults().getType(),
        /*tag=*/rewriter.getStringAttr("yield"),
        /*args=*/adaptor.getArgs());
    return success();
  }
};

struct StorageOpLowering : public OpConversionPattern<StorageOp> {
  using OpConversionPattern<StorageOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StorageOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // create a local variable initialized to null
    auto nullContRef =
        rewriter
            .create<ssawasm::NullContRefOp>(
                op.getLoc(),
                getTypeConverter()->convertType(op.getStorage().getType()))
            .getResult();
    rewriter.replaceOpWithNewOp<ssawasm::LocalOp>(op, nullContRef);
    return success();
  }
};
struct LoadOpLowering : public OpConversionPattern<LoadOp> {
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ssawasm::LocalGetOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getStorage());
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<StoreOp> {
  using OpConversionPattern<StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ssawasm::LocalSetOp>(op, adaptor.getStorage(),
                                                     adaptor.getCont());
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

    // FIXME: We should not hardcode this
    // We assume that all functions except the main have type: "ct" -> ()
    // and use "ft" to denote the type
    rewriter.create<ssawasm::RecContFuncDeclOp>(module.getLoc(),
                                                rewriter.getStringAttr("ft"),
                                                rewriter.getStringAttr("ct"));

    // NOTE: func::FuncOp will be converted to ssawasm::FuncOp
    // by the ConvertToSsaWasm pass
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.getName() != "main") {
        funcOp->setAttr("type_id", StringAttr::get(context, "ft"));
      }
    });

    RewritePatternSet patterns(context);
    DContToSsaWasmTypeConverter typeConverter(context);
    patterns.add<NewOpLowering, NullContOpLowering, ResumeOpLowering,
                 SuspendOpLowering, StorageOpLowering, LoadOpLowering,
                 StoreOpLowering>(typeConverter, context);
    // TODO: Fix and add ResumeSwitchOpLowering and SwitchOpLowering

    ConversionTarget target(getContext());
    target.addLegalDialect<ssawasm::SsaWasmDialect>();
    target.addIllegalDialect<DContDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::dcont