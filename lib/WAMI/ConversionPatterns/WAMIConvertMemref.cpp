//===- WAMIConvertMemref.cpp - MemRef to WasmSSA/WAMI conversion --*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion patterns from MemRef dialect to the upstream
// WasmSSA dialect and WAMI dialect for memory operations.
//
//===----------------------------------------------------------------------===//

#include "WAMI/ConversionPatterns/WAMIConvertMemref.h"

#include "WAMI/WAMIOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::wami {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

int64_t computeMemRefSize(MemRefType memRefType, int64_t alignment) {
  auto shape = memRefType.getShape();

  // Compute the total number of elements
  int64_t totalElements = 1;
  for (int64_t dimSize : shape) {
    if (ShapedType::isDynamic(dimSize))
      return -1; // Dynamic size
    totalElements *= dimSize;
  }

  // Determine the size of each element
  Type elementType = memRefType.getElementType();
  int64_t elementSize = 0;
  if (elementType.isIntOrFloat())
    elementSize = elementType.getIntOrFloatBitWidth() / 8;
  else
    return -1; // Unsupported element type

  // Calculate the total memory size
  int64_t totalMemorySize = totalElements * elementSize;

  // Adjust for alignment
  int64_t alignedMemorySize =
      ((totalMemorySize + alignment - 1) / alignment) * alignment;

  return alignedMemorySize;
}

/// Ensure a wasmssa.import_func exists for a runtime function.
/// Inserts a declaration at the module top if no symbol with the same name
/// currently exists.
static void ensureRuntimeImport(Operation *anchor,
                                ConversionPatternRewriter &rewriter,
                                StringRef symName, FunctionType type) {
  auto module = anchor->getParentOfType<ModuleOp>();
  if (!module)
    return;

  if (SymbolTable::lookupSymbolIn(module, symName))
    return;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  wasmssa::FuncImportOp::create(rewriter, anchor->getLoc(), symName, "env",
                                symName, type);
}

//===----------------------------------------------------------------------===//
// BaseAddressAnalysis Implementation
//===----------------------------------------------------------------------===//

WAMIBaseAddressAnalysis::WAMIBaseAddressAnalysis(ModuleOp &moduleOp) {
  // Start from 1024 - this is the WebAssembly convention to skip the first 1KB
  uint32_t baseAddress = 1024;

  moduleOp.walk([this, &baseAddress](memref::GlobalOp op) {
    auto symName = op.getSymName().str();

    // Get alignment requirement for this global
    int64_t alignment = op.getAlignment().value_or(1);

    // Align the base address to the required alignment boundary
    // Formula: alignedAddr = ((addr + alignment - 1) / alignment) * alignment
    if (alignment > 1) {
      baseAddress = ((baseAddress + alignment - 1) / alignment) * alignment;
    }

    // Assign the aligned address to this global
    baseAddressMap.emplace(symName, baseAddress);

    // Compute size and advance to next position
    int64_t size = computeMemRefSize(op.getType(), alignment);
    if (size > 0)
      baseAddress += size;
  });
}

uint32_t
WAMIBaseAddressAnalysis::getBaseAddress(const std::string &globalOpName) const {
  auto it = baseAddressMap.find(globalOpName);
  if (it != baseAddressMap.end())
    return it->second;
  return 0;
}

namespace {

//===----------------------------------------------------------------------===//
// Pointer Computation Helper
//===----------------------------------------------------------------------===//

/// Generates WebAssembly-style pointer computation for memref access.
/// Returns the computed i32 address for the given memref base and indices.
std::pair<LogicalResult, Value> generatePointerComputation(
    Operation *op, Value base, MemRefType memRefType, ValueRange indices,
    const TypeConverter *typeConverter, ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();

  // Get strides and offset for the memref layout
  SmallVector<int64_t, 4> strides;
  int64_t offset;
  if (failed(memRefType.getStridesAndOffset(strides, offset))) {
    return std::make_pair(
        rewriter.notifyMatchFailure(
            op, "Cannot compute strides and offset for the given MemRefType."),
        Value());
  }

  // Start with the base address (already i32)
  Value result = base;

  // Add the base offset if non-zero
  // Address = base + offset * elementSize + sum(indices[i] * strides[i] *
  // elementSize)
  if (!ShapedType::isDynamic(offset) && offset != 0) {
    Type memRefElementType = memRefType.getElementType();
    int64_t elementSize = memRefElementType.getIntOrFloatBitWidth() / 8;
    int64_t byteOffset = offset * elementSize;
    Value offsetConst = wasmssa::ConstOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(byteOffset));
    result = wasmssa::AddOp::create(rewriter, loc, rewriter.getI32Type(),
                                    result, offsetConst);
  } else if (ShapedType::isDynamic(offset)) {
    return std::make_pair(
        rewriter.notifyMatchFailure(
            op, "Cannot handle dynamic offset in the MemRefType."),
        Value());
  }

  // Compute: base + offset * elementSize + sum(indices[i] * strides[i] *
  // elementSize)
  for (int64_t i = 0; i < memRefType.getRank(); i++) {
    if (ShapedType::isDynamic(strides[i])) {
      return std::make_pair(
          rewriter.notifyMatchFailure(
              op, "Cannot handle dynamic strides in the MemRefType."),
          Value());
    }

    Type memRefElementType = memRefType.getElementType();
    int64_t elementSize = memRefElementType.getIntOrFloatBitWidth() / 8;

    // Create constants for stride and element size multiplication
    auto strideXSize = strides[i] * elementSize;
    Value strideConst = wasmssa::ConstOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(strideXSize));

    // Multiply index by (stride * elementSize)
    Value index = indices[i];
    // Ensure index is i32
    if (!index.getType().isInteger(32)) {
      index =
          wasmssa::WrapOp::create(rewriter, loc, rewriter.getI32Type(), index);
    }

    Value indexOffset = wasmssa::MulOp::create(
        rewriter, loc, rewriter.getI32Type(), index, strideConst);

    // Add to running total
    result = wasmssa::AddOp::create(rewriter, loc, rewriter.getI32Type(),
                                    result, indexOffset);
  }

  return std::make_pair(success(), result);
}

//===----------------------------------------------------------------------===//
// Global Memory Patterns
//===----------------------------------------------------------------------===//

/// Converts memref.global to wami.data + wasmssa.global
struct GlobalOpLowering : public OpConversionPattern<memref::GlobalOp> {
  GlobalOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                   WAMIBaseAddressAnalysis &baseAddressAnalysis)
      : OpConversionPattern<memref::GlobalOp>(typeConverter, context),
        baseAddressAnalysis(baseAddressAnalysis) {}

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto symName = op.getSymName();
    auto baseAddress = baseAddressAnalysis.getBaseAddress(symName.str());

    // Get the initial value if present
    auto initialValue = op.getInitialValue();

    if (initialValue.has_value()) {
      // Create wami.data operation for the data section
      std::string dataSymName = (symName + "_data").str();
      auto elementsAttr = cast<ElementsAttr>(initialValue.value());
      DataOp::create(rewriter, loc, rewriter.getStringAttr(dataSymName),
                     elementsAttr, rewriter.getI32IntegerAttr(baseAddress));
    }

    // Create wasmssa.global to store the base address
    // The global holds an i32 value representing the memory address
    std::string baseSymName = (symName + "_base").str();

    // Create the global op using the OpTy::create interface
    auto globalOp = wasmssa::GlobalOp::create(
        rewriter, loc, rewriter.getStringAttr(baseSymName),
        TypeAttr::get(rewriter.getI32Type()),
        /*isMutable=*/rewriter.getUnitAttr(),
        /*exported=*/nullptr);

    // Add initializer region with the base address constant
    Block *initBlock = rewriter.createBlock(&globalOp.getInitializer());
    rewriter.setInsertionPointToStart(initBlock);
    Value baseAddrConst = wasmssa::ConstOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(baseAddress));
    wasmssa::ReturnOp::create(rewriter, loc, baseAddrConst);

    rewriter.eraseOp(op);
    return success();
  }

private:
  WAMIBaseAddressAnalysis &baseAddressAnalysis;
};

/// Converts memref.get_global to wasmssa.global_get
struct GetGlobalOpLowering : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern<memref::GetGlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the base address from the corresponding global
    std::string baseSymName = (op.getName() + "_base").str();
    rewriter.replaceOpWithNewOp<wasmssa::GlobalGetOp>(
        op, rewriter.getI32Type(), rewriter.getStringAttr(baseSymName));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Load/Store Patterns
//===----------------------------------------------------------------------===//

/// Converts memref.load to wami.load with address computation
struct LoadOpLowering : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [result, pointer] = generatePointerComputation(
        op, adaptor.getMemref(), op.getMemRefType(), adaptor.getIndices(),
        typeConverter, rewriter);

    if (failed(result))
      return result;

    auto elementType = op.getMemRefType().getElementType();
    auto resultType = getTypeConverter()->convertType(elementType);

    rewriter.replaceOpWithNewOp<LoadOp>(op, resultType, pointer);
    return success();
  }
};

/// Converts memref.store to wami.store with address computation
struct StoreOpLowering : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [result, pointer] = generatePointerComputation(
        op, adaptor.getMemref(), op.getMemRefType(), adaptor.getIndices(),
        typeConverter, rewriter);

    if (failed(result))
      return result;

    rewriter.replaceOpWithNewOp<StoreOp>(op, pointer, adaptor.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Allocation Patterns
//===----------------------------------------------------------------------===//

/// Converts memref.alloc to malloc call
struct AllocOpLowering : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memRefType = op.getType();

    int64_t alignment = adaptor.getAlignment().value_or(1);
    int64_t size = computeMemRefSize(memRefType, alignment);

    if (size < 0)
      return rewriter.notifyMatchFailure(op, "Cannot compute memref size");

    Value sizeConst = wasmssa::ConstOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(size));

    ensureRuntimeImport(
        op, rewriter, "malloc",
        rewriter.getFunctionType(rewriter.getI32Type(), rewriter.getI32Type()));

    // Call malloc and return the pointer as i32
    rewriter.replaceOpWithNewOp<wasmssa::FuncCallOp>(
        op, rewriter.getI32Type(), "malloc", ValueRange{sizeConst});
    return success();
  }
};

/// Converts memref.dealloc to free call
struct DeallocOpLowering : public OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ensureRuntimeImport(
        op, rewriter, "free",
        rewriter.getFunctionType(rewriter.getI32Type(), TypeRange{}));

    // Call free with the pointer
    rewriter.replaceOpWithNewOp<wasmssa::FuncCallOp>(
        op, TypeRange{}, "free", ValueRange{adaptor.getMemref()});
    return success();
  }
};

/// Converts memref.alloca to malloc call (could be optimized to stack later)
struct AllocaOpLowering : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern<memref::AllocaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For simplicity, treat alloca like alloc (use malloc)
    // TODO: Could optimize small fixed-size allocas to use locals
    Location loc = op.getLoc();
    MemRefType memRefType = op.getType();

    int64_t alignment = adaptor.getAlignment().value_or(1);
    int64_t size = computeMemRefSize(memRefType, alignment);

    if (size < 0)
      return rewriter.notifyMatchFailure(op, "Cannot compute memref size");

    Value sizeConst = wasmssa::ConstOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(size));

    ensureRuntimeImport(
        op, rewriter, "malloc",
        rewriter.getFunctionType(rewriter.getI32Type(), rewriter.getI32Type()));

    rewriter.replaceOpWithNewOp<wasmssa::FuncCallOp>(
        op, rewriter.getI32Type(), "malloc", ValueRange{sizeConst});
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateWAMIConvertMemrefPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    WAMIBaseAddressAnalysis &baseAddressAnalysis) {
  MLIRContext *context = patterns.getContext();

  patterns.add<GlobalOpLowering>(typeConverter, context, baseAddressAnalysis);
  patterns.add<GetGlobalOpLowering, LoadOpLowering, StoreOpLowering,
               AllocOpLowering, DeallocOpLowering, AllocaOpLowering>(
      typeConverter, context);
}

} // namespace mlir::wami
