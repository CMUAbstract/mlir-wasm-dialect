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
#include <set>

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
      if (width != 32 && width != 64) {
        return IntegerType::get(ctx, 32);
      }
      return type;
    });
    addConversion(
        [ctx](IndexType type) -> Type { return IntegerType::get(ctx, 32); });
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
                op->getLoc(), ssawasm::WasmFuncRefType::get(op.getContext()),
                adaptor.getFunctionName())
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

    // create local variables for the results
    SmallVector<Value> newResults;
    for (auto result : op.getResults()) {
      auto convertedType = getTypeConverter()->convertType(result.getType());
      auto local = rewriter.create<ssawasm::LocalDeclOp>(loc, convertedType);
      rewriter.create<ssawasm::LocalSetOp>(loc, local, result);
      newResults.push_back(local);
    }

    // create a BlockBlockOp
    auto blockBlockOp = rewriter.create<ssawasm::BlockBlockOp>(loc);
    blockBlockOp.setInnerBlockResultTypesAttr(
        rewriter.getArrayAttr(ArrayRef<Attribute>{TypeAttr::get(
            getTypeConverter()->convertType(op.getCont().getType()))}));
    auto [outerEntryBlock, innerEntryBlock2, innerExitBlock, outerExitBlock] =
        blockBlockOp.initialize(rewriter);
    auto innerEntryBlock1 =
        rewriter.splitBlock(innerEntryBlock2, innerEntryBlock2->begin());

    rewriter.setInsertionPointToEnd(outerEntryBlock);
    rewriter.create<ssawasm::TempBranchOp>(loc, innerEntryBlock1);

    rewriter.setInsertionPointToEnd(innerEntryBlock1);
    auto resumeOp = rewriter.create<ssawasm::ResumeOp>(
        loc,
        /*results=*/op.getResults().getType(),
        /*tag=*/rewriter.getStringAttr("yield"),
        /*cont=*/adaptor.getCont(),
        /*args=*/adaptor.getArgs(),
        /*on_yield=*/innerExitBlock,
        /*fallback=*/innerEntryBlock2);

    // if resumeOp returns a value, we need to store it in the local variables
    if (resumeOp.getResults().size() > 0) {
      for (auto [result, local] :
           llvm::zip(resumeOp.getResults(), newResults)) {
        rewriter.create<ssawasm::LocalSetOp>(loc, local, result);
      }
    }

    rewriter.setInsertionPointToEnd(innerEntryBlock2);
    rewriter.create<ssawasm::BlockBlockBranchOp>(loc, outerExitBlock);

    rewriter.setInsertionPointToEnd(innerExitBlock);
    // Copy block arguments to the innerExitBlock with type conversion
    SmallVector<Value> stackArgs;
    for (auto arg : op.getSuspendHandler().front().getArguments()) {
      auto onStackOp = rewriter.create<ssawasm::OnStackOp>(
          loc, getTypeConverter()->convertType(arg.getType()));
      stackArgs.push_back(onStackOp.getResult());
    }

    // Inline handler operations into innerExitBlock, except the terminator
    Block &handlerBlock = op.getSuspendHandler().front();
    auto *handlerReturnOp = handlerBlock.getTerminator();
    rewriter.mergeBlocks(&handlerBlock, innerExitBlock, stackArgs);

    // if the handler returns a value, we need to save it in the local
    // variables
    for (auto [result, local] :
         llvm::zip(handlerReturnOp->getResults(), newResults)) {
      rewriter.create<ssawasm::LocalSetOp>(loc, local, result);
    }
    rewriter.eraseOp(handlerReturnOp);

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
    rewriter.replaceOpWithNewOp<ssawasm::LocalDeclOp>(
        op, op.getStorage().getType());
    return success();
  }
};
struct LoadOpLowering : public OpConversionPattern<LoadOp> {
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We just erase this op (with type casting)
    // because reading from a local will be converted to a local get
    // through the introduce-locals pass
    Value convertedValue =
        rewriter
            .create<UnrealizedConversionCastOp>(
                op.getLoc(),
                getTypeConverter()->convertType(op.getResult().getType()),
                adaptor.getStorage())
            .getResult(0);

    rewriter.replaceOp(op, convertedValue);
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

    // search for all functions that are called by `dcont.new`

    set<func::FuncOp> contFunctions;

    module.walk([&](NewOp newOp) {
      // search for funcOp with the same name as newOp.getFunctionName()
      auto funcOp = module.lookupSymbol<func::FuncOp>(newOp.getFunctionName());
      if (funcOp) {
        contFunctions.insert(funcOp);
      }
    });
    // TODO: We only support one continuation type for now
    // this means that all functions called by dcont.new
    // should have the same type
    FunctionType firstFuncType = nullptr;
    for (auto funcOp : contFunctions) {
      FunctionType funcType = funcOp.getFunctionType();

      if (!firstFuncType) {
        firstFuncType = funcType;
      } else if (firstFuncType != funcType) {
        funcOp->emitError()
            << "All functions called by dcont.new must have the same type.\n"
            << "  First function type: " << firstFuncType << "\n"
            << "  Current function type: " << funcType;
        return;
      }
    }
    // assign type_id to all functions called by dcont.new
    for (auto funcOp : contFunctions) {
      funcOp->setAttr("type_id", StringAttr::get(context, "ft"));
    }
    // declare the type for all functions called by dcont.new
    for (auto funcOp : contFunctions) {
      rewriter.setInsertionPointAfter(funcOp);
      rewriter.create<ssawasm::ElemDeclFuncOp>(module.getLoc(),
                                               funcOp.getName());
    }
    // declare func type
    rewriter.setInsertionPoint(module.getBody(), module.getBody()->begin());
    rewriter.create<ssawasm::FuncTypeDeclOp>(
        module.getLoc(), rewriter.getStringAttr("ft"), firstFuncType);

    // check if the cont type is recursive
    bool isRecursive = false;
    if (!contFunctions.empty()) {
      auto funcOp = *contFunctions.begin();
      FunctionType funcType = funcOp.getFunctionType();

      // Check if any of the input types matches the function type
      for (Type inputType : funcType.getInputs()) {
        if (auto contType = dyn_cast<ContType>(inputType)) {
          if (contType.getId() == "ct") {
            isRecursive = true;
            break;
          }
        }
      }
    }

    if (isRecursive) {
      // FIXME: We should not hardcode this
      rewriter.create<ssawasm::RecContFuncDeclOp>(module.getLoc(),
                                                  rewriter.getStringAttr("ft"),
                                                  rewriter.getStringAttr("ct"));
    } else {
      rewriter.create<ssawasm::ContTypeDeclOp>(module.getLoc(),
                                               rewriter.getStringAttr("ct"),
                                               rewriter.getStringAttr("ft"));
    }

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