//===- WasmStackPasses.cpp - WasmStack dialect passes -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the stackification pass that converts WasmSSA+WAMI
// dialects to the WasmStack dialect using LLVM-style stackification.
//
//===----------------------------------------------------------------------===//

#include "wasmstack/WasmStackPasses.h"
#include "WAMI/WAMIDialect.h"
#include "WAMI/WAMIOps.h"
#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackOps.h"

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::wasmstack {

#define GEN_PASS_DEF_CONVERTTOWASMSTACK
#include "wasmstack/WasmStackPasses.h.inc"

//===----------------------------------------------------------------------===//
// Dependency Analysis
//===----------------------------------------------------------------------===//

/// Query result for operation dependencies
struct DepInfo {
  bool readsMemory = false;
  bool writesMemory = false;
  bool hasSideEffects = false;
  // Note: WebAssembly has no volatile semantics - all memory accesses
  // can be reordered unless they have data dependencies
};

/// Analyze an operation for its memory and side effect dependencies
static DepInfo queryDependencies(Operation *op) {
  DepInfo info;

  // Memory loads
  if (isa<wami::LoadOp>(op)) {
    info.readsMemory = true;
  }

  // Memory stores - both write memory AND have side effects
  // (order of stores must be preserved)
  if (isa<wami::StoreOp>(op)) {
    info.writesMemory = true;
    info.hasSideEffects = true;
  }

  // Global reads - treat like memory reads for ordering purposes
  if (isa<wasmssa::GlobalGetOp>(op)) {
    info.readsMemory = true;
  }

  // Note: WasmSSA doesn't have GlobalSetOp - mutable globals are
  // handled differently. If added later, it would have side effects.

  // Calls have side effects and may read/write memory
  if (isa<wasmssa::FuncCallOp>(op)) {
    info.hasSideEffects = true;
    info.readsMemory = true;
    info.writesMemory = true;
  }

  // Local operations don't have side effects - they're SSA values
  // LocalGetOp, LocalSetOp, LocalTeeOp are fine to reorder

  // TODO: Add stack switching operations when implemented
  // suspend, resume, switch all have significant side effects

  return info;
}

/// Check if it's safe to move defOp to immediately before insertBefore
static bool isSafeToMove(Operation *defOp, Operation *insertBefore) {
  // Can't move across basic block boundaries
  if (defOp->getBlock() != insertBefore->getBlock())
    return false;

  DepInfo defDeps = queryDependencies(defOp);

  // Walk operations between defOp and insertBefore
  for (Operation *op = defOp->getNextNode(); op != insertBefore;
       op = op->getNextNode()) {
    if (!op)
      return false; // Reached end of block without finding insertBefore

    DepInfo opDeps = queryDependencies(op);

    // Check memory dependencies
    if (defDeps.readsMemory && opDeps.writesMemory)
      return false; // Read-after-write hazard
    if (defDeps.writesMemory && opDeps.readsMemory)
      return false; // Write-after-read hazard
    if (defDeps.writesMemory && opDeps.writesMemory)
      return false; // Write-after-write hazard

    // Check side effects (calls, traps, etc.)
    if (defDeps.hasSideEffects || opDeps.hasSideEffects)
      return false;
  }
  return true;
}

/// Check if an operation should be rematerialized instead of using a local
static bool shouldRematerialize(Operation *op) {
  // Constants are always cheap to rematerialize
  if (isa<wasmssa::ConstOp>(op))
    return true;

  // Local.get is cheap (in the source dialect)
  if (isa<wasmssa::LocalGetOp>(op))
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Use Count Analysis
//===----------------------------------------------------------------------===//

/// Counts the number of uses for each operation's results
class UseCountAnalysis {
  DenseMap<Value, unsigned> useCounts;

public:
  UseCountAnalysis(Block &block) { analyze(block); }

  void analyze(Block &block) {
    for (Operation &op : block) {
      for (Value operand : op.getOperands()) {
        useCounts[operand]++;
      }
      // Recursively analyze nested regions
      for (Region &region : op.getRegions()) {
        for (Block &nestedBlock : region) {
          analyze(nestedBlock);
        }
      }
    }
  }

  unsigned getUseCount(Value value) const {
    auto it = useCounts.find(value);
    return it != useCounts.end() ? it->second : 0;
  }

  bool hasSingleUse(Value value) const { return getUseCount(value) == 1; }
};

//===----------------------------------------------------------------------===//
// Local Allocation
//===----------------------------------------------------------------------===//

/// Allocates local indices to values that need locals
/// Parameters: indices 0..N-1
/// Introduced locals: indices N..M
class LocalAllocator {
  /// Mapping from Value to local index
  DenseMap<Value, unsigned> localIndices;

  /// Type of each local (indexed by local index)
  SmallVector<Type> localTypes;

  /// Number of function parameters
  unsigned numParams = 0;

public:
  /// Unwrap a WasmSSA local ref type to get the underlying value type
  static Type unwrapLocalRefType(Type type) {
    // Check if this is a LocalRefType (!wasmssa<local ref to T>)
    if (auto localRefType = dyn_cast<wasmssa::LocalRefType>(type)) {
      return localRefType.getElementType();
    }
    return type;
  }

  /// Allocate locals for a function
  void allocate(wasmssa::FuncOp funcOp, const DenseSet<Value> &needsLocal,
                const DenseSet<Value> &needsTee) {
    // First, assign indices to parameters
    // WasmSSA functions use !wasmssa<local ref to T> for parameters
    // The block arguments in the entry block are the parameters
    if (!funcOp.getBody().empty()) {
      Block &entryBlock = funcOp.getBody().front();
      for (BlockArgument arg : entryBlock.getArguments()) {
        localIndices[arg] = numParams;
        // Unwrap local ref types to get the underlying value type
        Type unwrappedType = unwrapLocalRefType(arg.getType());
        localTypes.push_back(unwrappedType);
        numParams++;
      }
    }

    // Then, assign indices to introduced locals
    unsigned nextLocalIdx = numParams;

    // Assign indices to values that need full locals (local.set/local.get)
    for (Value value : needsLocal) {
      if (!localIndices.count(value)) {
        localIndices[value] = nextLocalIdx++;
        localTypes.push_back(value.getType());
      }
    }

    // Assign indices to values that use tee (local.tee + local.get)
    for (Value value : needsTee) {
      if (!localIndices.count(value)) {
        localIndices[value] = nextLocalIdx++;
        localTypes.push_back(value.getType());
      }
    }
  }

  /// Get the local index for a value, or -1 if not allocated
  int getLocalIndex(Value value) const {
    auto it = localIndices.find(value);
    if (it != localIndices.end())
      return it->second;
    return -1;
  }

  /// Check if a value has an allocated local
  bool hasLocal(Value value) const { return localIndices.count(value); }

  /// Get the number of parameters
  unsigned getNumParams() const { return numParams; }

  /// Get the total number of locals (including parameters)
  unsigned getNumLocals() const { return localTypes.size(); }

  /// Get the type of a local by index
  Type getLocalType(unsigned idx) const { return localTypes[idx]; }

  /// Get all local types
  ArrayRef<Type> getLocalTypes() const { return localTypes; }
};

//===----------------------------------------------------------------------===//
// WasmStack Emitter
//===----------------------------------------------------------------------===//

/// Emits WasmStack operations from stackified WasmSSA/WAMI operations
class WasmStackEmitter {
  OpBuilder &builder;
  const LocalAllocator &allocator;
  const DenseSet<Value> &needsTee;

  /// Tracks which values have been emitted to the stack
  DenseSet<Value> emittedToStack;

  /// Counter for generating unique labels (member variable to avoid static)
  unsigned labelCounter = 0;

  /// Stack of labels for control flow structures
  /// Each entry is (label, isLoop) - isLoop determines branch behavior
  SmallVector<std::pair<std::string, bool>> labelStack;

public:
  WasmStackEmitter(OpBuilder &builder, const LocalAllocator &allocator,
                   const DenseSet<Value> &needsTee)
      : builder(builder), allocator(allocator), needsTee(needsTee) {}

  /// Emit a WasmStack function from a WasmSSA function
  FuncOp emitFunction(wasmssa::FuncOp srcFunc) {
    Location loc = srcFunc.getLoc();

    // Get function type
    FunctionType funcType = srcFunc.getFunctionType();

    // Create WasmStack function
    auto dstFunc =
        FuncOp::create(builder, loc, srcFunc.getName(), funcType, StringAttr());

    // Create entry block
    Block *entryBlock = new Block();
    dstFunc.getBody().push_back(entryBlock);
    builder.setInsertionPointToStart(entryBlock);

    // Emit local declarations for introduced locals (not parameters)
    for (unsigned i = allocator.getNumParams(); i < allocator.getNumLocals();
         ++i) {
      Type localType = allocator.getLocalType(i);
      LocalOp::create(builder, loc, static_cast<uint32_t>(i), localType);
    }

    // Emit operations from the source function using CFG linearization
    // This handles functions with multiple blocks (e.g., control flow with
    // successor blocks containing the return statement)
    if (!srcFunc.getBody().empty()) {
      Block *currentBlock = &srcFunc.getBody().front();
      llvm::DenseSet<Block *> processed;

      while (currentBlock && !processed.contains(currentBlock)) {
        processed.insert(currentBlock);

        // Mark block arguments as available on stack
        for (BlockArgument arg : currentBlock->getArguments()) {
          emittedToStack.insert(arg);
        }

        // Emit all operations EXCEPT the terminator
        for (Operation &op : currentBlock->without_terminator()) {
          // Skip operations that have results but no users (e.g., cloned ops
          // that were created but the original became unused after
          // rematerialization)
          if (op.getNumResults() > 0 && op.use_empty()) {
            continue;
          }
          emitOperation(&op);
        }

        // Handle terminator and get next block to process
        Operation *terminator = currentBlock->getTerminator();
        if (terminator) {
          currentBlock =
              emitTerminatorAndGetNext(terminator, /*isInLoop=*/false);
        } else {
          currentBlock = nullptr;
        }
      }
    }

    return dstFunc;
  }

  /// Emit a single operation
  void emitOperation(Operation *op) {
    Location loc = op->getLoc();

    // Handle different operation types

    // Constants
    if (auto constOp = dyn_cast<wasmssa::ConstOp>(op)) {
      emitConst(constOp);
    }
    // Basic arithmetic
    else if (auto addOp = dyn_cast<wasmssa::AddOp>(op)) {
      emitBinaryOp<AddOp>(addOp, addOp.getLhs(), addOp.getRhs(),
                          addOp.getResult());
    } else if (auto subOp = dyn_cast<wasmssa::SubOp>(op)) {
      emitBinaryOp<SubOp>(subOp, subOp.getLhs(), subOp.getRhs(),
                          subOp.getResult());
    } else if (auto mulOp = dyn_cast<wasmssa::MulOp>(op)) {
      emitBinaryOp<MulOp>(mulOp, mulOp.getLhs(), mulOp.getRhs(),
                          mulOp.getResult());
    }
    // Division and remainder
    else if (auto divOp = dyn_cast<wasmssa::DivOp>(op)) {
      emitBinaryOp<FDivOp>(divOp, divOp.getLhs(), divOp.getRhs(),
                           divOp.getResult());
    } else if (auto divSIOp = dyn_cast<wasmssa::DivSIOp>(op)) {
      emitBinaryOp<DivSOp>(divSIOp, divSIOp.getLhs(), divSIOp.getRhs(),
                           divSIOp.getResult());
    } else if (auto divUIOp = dyn_cast<wasmssa::DivUIOp>(op)) {
      emitBinaryOp<DivUOp>(divUIOp, divUIOp.getLhs(), divUIOp.getRhs(),
                           divUIOp.getResult());
    } else if (auto remSIOp = dyn_cast<wasmssa::RemSIOp>(op)) {
      emitBinaryOp<RemSOp>(remSIOp, remSIOp.getLhs(), remSIOp.getRhs(),
                           remSIOp.getResult());
    } else if (auto remUIOp = dyn_cast<wasmssa::RemUIOp>(op)) {
      emitBinaryOp<RemUOp>(remUIOp, remUIOp.getLhs(), remUIOp.getRhs(),
                           remUIOp.getResult());
    }
    // Bitwise operations
    else if (auto andOp = dyn_cast<wasmssa::AndOp>(op)) {
      emitBinaryOp<AndOp>(andOp, andOp.getLhs(), andOp.getRhs(),
                          andOp.getResult());
    } else if (auto orOp = dyn_cast<wasmssa::OrOp>(op)) {
      emitBinaryOp<OrOp>(orOp, orOp.getLhs(), orOp.getRhs(), orOp.getResult());
    } else if (auto xorOp = dyn_cast<wasmssa::XOrOp>(op)) {
      emitBinaryOp<XorOp>(xorOp, xorOp.getLhs(), xorOp.getRhs(),
                          xorOp.getResult());
    }
    // Shift and rotate
    else if (auto shlOp = dyn_cast<wasmssa::ShLOp>(op)) {
      emitBinaryOp<ShlOp>(shlOp, shlOp.getVal(), shlOp.getBits(),
                          shlOp.getResult());
    } else if (auto shrSOp = dyn_cast<wasmssa::ShRSOp>(op)) {
      emitBinaryOp<ShrSOp>(shrSOp, shrSOp.getVal(), shrSOp.getBits(),
                           shrSOp.getResult());
    } else if (auto shrUOp = dyn_cast<wasmssa::ShRUOp>(op)) {
      emitBinaryOp<ShrUOp>(shrUOp, shrUOp.getVal(), shrUOp.getBits(),
                           shrUOp.getResult());
    } else if (auto rotlOp = dyn_cast<wasmssa::RotlOp>(op)) {
      emitBinaryOp<RotlOp>(rotlOp, rotlOp.getVal(), rotlOp.getBits(),
                           rotlOp.getResult());
    } else if (auto rotrOp = dyn_cast<wasmssa::RotrOp>(op)) {
      emitBinaryOp<RotrOp>(rotrOp, rotrOp.getVal(), rotrOp.getBits(),
                           rotrOp.getResult());
    }
    // Float-specific binary ops
    else if (auto minOp = dyn_cast<wasmssa::MinOp>(op)) {
      emitBinaryOp<FMinOp>(minOp, minOp.getLhs(), minOp.getRhs(),
                           minOp.getResult());
    } else if (auto maxOp = dyn_cast<wasmssa::MaxOp>(op)) {
      emitBinaryOp<FMaxOp>(maxOp, maxOp.getLhs(), maxOp.getRhs(),
                           maxOp.getResult());
    } else if (auto copysignOp = dyn_cast<wasmssa::CopySignOp>(op)) {
      emitBinaryOp<FCopysignOp>(copysignOp, copysignOp.getLhs(),
                                copysignOp.getRhs(), copysignOp.getResult());
    }
    // Integer comparison operations
    else if (auto eqOp = dyn_cast<wasmssa::EqOp>(op)) {
      emitCompareOp<EqOp>(eqOp, eqOp.getLhs(), eqOp.getRhs());
      emittedToStack.insert(eqOp.getResult());
    } else if (auto neOp = dyn_cast<wasmssa::NeOp>(op)) {
      emitCompareOp<NeOp>(neOp, neOp.getLhs(), neOp.getRhs());
      emittedToStack.insert(neOp.getResult());
    } else if (auto ltSIOp = dyn_cast<wasmssa::LtSIOp>(op)) {
      emitCompareOp<LtSOp>(ltSIOp, ltSIOp.getLhs(), ltSIOp.getRhs());
      emittedToStack.insert(ltSIOp.getResult());
    } else if (auto ltUIOp = dyn_cast<wasmssa::LtUIOp>(op)) {
      emitCompareOp<LtUOp>(ltUIOp, ltUIOp.getLhs(), ltUIOp.getRhs());
      emittedToStack.insert(ltUIOp.getResult());
    } else if (auto leSIOp = dyn_cast<wasmssa::LeSIOp>(op)) {
      emitCompareOp<LeSOp>(leSIOp, leSIOp.getLhs(), leSIOp.getRhs());
      emittedToStack.insert(leSIOp.getResult());
    } else if (auto leUIOp = dyn_cast<wasmssa::LeUIOp>(op)) {
      emitCompareOp<LeUOp>(leUIOp, leUIOp.getLhs(), leUIOp.getRhs());
      emittedToStack.insert(leUIOp.getResult());
    } else if (auto gtSIOp = dyn_cast<wasmssa::GtSIOp>(op)) {
      emitCompareOp<GtSOp>(gtSIOp, gtSIOp.getLhs(), gtSIOp.getRhs());
      emittedToStack.insert(gtSIOp.getResult());
    } else if (auto gtUIOp = dyn_cast<wasmssa::GtUIOp>(op)) {
      emitCompareOp<GtUOp>(gtUIOp, gtUIOp.getLhs(), gtUIOp.getRhs());
      emittedToStack.insert(gtUIOp.getResult());
    } else if (auto geSIOp = dyn_cast<wasmssa::GeSIOp>(op)) {
      emitCompareOp<GeSOp>(geSIOp, geSIOp.getLhs(), geSIOp.getRhs());
      emittedToStack.insert(geSIOp.getResult());
    } else if (auto geUIOp = dyn_cast<wasmssa::GeUIOp>(op)) {
      emitCompareOp<GeUOp>(geUIOp, geUIOp.getLhs(), geUIOp.getRhs());
      emittedToStack.insert(geUIOp.getResult());
    }
    // Float comparison operations
    else if (auto ltOp = dyn_cast<wasmssa::LtOp>(op)) {
      emitCompareOp<FLtOp>(ltOp, ltOp.getLhs(), ltOp.getRhs());
      emittedToStack.insert(ltOp.getResult());
    } else if (auto leOp = dyn_cast<wasmssa::LeOp>(op)) {
      emitCompareOp<FLeOp>(leOp, leOp.getLhs(), leOp.getRhs());
      emittedToStack.insert(leOp.getResult());
    } else if (auto gtOp = dyn_cast<wasmssa::GtOp>(op)) {
      emitCompareOp<FGtOp>(gtOp, gtOp.getLhs(), gtOp.getRhs());
      emittedToStack.insert(gtOp.getResult());
    } else if (auto geOp = dyn_cast<wasmssa::GeOp>(op)) {
      emitCompareOp<FGeOp>(geOp, geOp.getLhs(), geOp.getRhs());
      emittedToStack.insert(geOp.getResult());
    }
    // Test operation
    else if (auto eqzOp = dyn_cast<wasmssa::EqzOp>(op)) {
      emitTestOp<EqzOp>(eqzOp, eqzOp.getInput());
      emittedToStack.insert(eqzOp.getResult());
    }
    // Unary integer operations
    else if (auto clzOp = dyn_cast<wasmssa::ClzOp>(op)) {
      emitUnaryOp<ClzOp>(clzOp, clzOp.getSrc(), clzOp.getResult());
    } else if (auto ctzOp = dyn_cast<wasmssa::CtzOp>(op)) {
      emitUnaryOp<CtzOp>(ctzOp, ctzOp.getSrc(), ctzOp.getResult());
    } else if (auto popcntOp = dyn_cast<wasmssa::PopCntOp>(op)) {
      emitUnaryOp<PopcntOp>(popcntOp, popcntOp.getSrc(), popcntOp.getResult());
    }
    // Unary float operations
    else if (auto absOp = dyn_cast<wasmssa::AbsOp>(op)) {
      emitUnaryOp<FAbsOp>(absOp, absOp.getSrc(), absOp.getResult());
    } else if (auto negOp = dyn_cast<wasmssa::NegOp>(op)) {
      emitUnaryOp<FNegOp>(negOp, negOp.getSrc(), negOp.getResult());
    } else if (auto sqrtOp = dyn_cast<wasmssa::SqrtOp>(op)) {
      emitUnaryOp<FSqrtOp>(sqrtOp, sqrtOp.getSrc(), sqrtOp.getResult());
    } else if (auto ceilOp = dyn_cast<wasmssa::CeilOp>(op)) {
      emitUnaryOp<FCeilOp>(ceilOp, ceilOp.getSrc(), ceilOp.getResult());
    } else if (auto floorOp = dyn_cast<wasmssa::FloorOp>(op)) {
      emitUnaryOp<FFloorOp>(floorOp, floorOp.getSrc(), floorOp.getResult());
    } else if (auto truncOp = dyn_cast<wasmssa::TruncOp>(op)) {
      emitUnaryOp<FTruncOp>(truncOp, truncOp.getSrc(), truncOp.getResult());
    }
    // Source dialect local operations
    else if (auto localGetOp = dyn_cast<wasmssa::LocalGetOp>(op)) {
      emitSourceLocalGet(localGetOp);
    } else if (auto localSetOp = dyn_cast<wasmssa::LocalSetOp>(op)) {
      emitSourceLocalSet(localSetOp);
    } else if (auto localTeeOp = dyn_cast<wasmssa::LocalTeeOp>(op)) {
      emitSourceLocalTee(localTeeOp);
    }
    // Memory operations
    else if (auto loadOp = dyn_cast<wami::LoadOp>(op)) {
      emitLoad(loadOp);
    } else if (auto storeOp = dyn_cast<wami::StoreOp>(op)) {
      emitStore(storeOp);
    }
    // Function call
    else if (auto callOp = dyn_cast<wasmssa::FuncCallOp>(op)) {
      emitCall(callOp);
    }
    // Control flow
    else if (auto returnOp = dyn_cast<wasmssa::ReturnOp>(op)) {
      // Emit return operands to ensure they're on the stack
      // (TreeWalker usually handles this, but we emit explicitly for
      // robustness)
      for (Value operand : returnOp.getOperands()) {
        emitOperandIfNeeded(operand);
      }
      ReturnOp::create(builder, loc);
    } else if (auto blockOp = dyn_cast<wasmssa::BlockOp>(op)) {
      emitBlock(blockOp);
    } else if (auto loopOp = dyn_cast<wasmssa::LoopOp>(op)) {
      emitLoop(loopOp);
    } else if (auto ifOp = dyn_cast<wasmssa::IfOp>(op)) {
      emitIf(ifOp);
    } else if (auto branchIfOp = dyn_cast<wasmssa::BranchIfOp>(op)) {
      emitBranchIf(branchIfOp);
    } else if (auto blockReturnOp = dyn_cast<wasmssa::BlockReturnOp>(op)) {
      // Block return: emit the operands that will be the block's result
      // These values must be on the stack when control exits the block
      for (Value input : blockReturnOp.getInputs()) {
        emitOperandIfNeeded(input);
      }
      // No explicit wasmstack instruction needed - values are now on stack
      // and control flows to the block's end
    } else {
      // Report unhandled operations to avoid silent failures
      op->emitWarning("unhandled operation in stackification: ")
          << op->getName();
    }
  }

private:
  /// Emit a constant operation
  void emitConst(wasmssa::ConstOp constOp) {
    Location loc = constOp.getLoc();
    Type resultType = constOp.getResult().getType();
    Attribute value = constOp.getValueAttr();

    if (resultType.isInteger(32)) {
      auto intVal = cast<IntegerAttr>(value).getInt();
      I32ConstOp::create(
          builder, loc,
          builder.getI32IntegerAttr(static_cast<int32_t>(intVal)));
    } else if (resultType.isInteger(64)) {
      auto intVal = cast<IntegerAttr>(value).getInt();
      I64ConstOp::create(builder, loc, builder.getI64IntegerAttr(intVal));
    } else if (resultType.isF32()) {
      auto floatVal = cast<FloatAttr>(value).getValueAsDouble();
      F32ConstOp::create(builder, loc,
                         builder.getF32FloatAttr(static_cast<float>(floatVal)));
    } else if (resultType.isF64()) {
      auto floatVal = cast<FloatAttr>(value).getValueAsDouble();
      F64ConstOp::create(builder, loc, builder.getF64FloatAttr(floatVal));
    }

    // Check if this value needs tee
    Value result = constOp.getResult();
    if (needsTee.contains(result)) {
      int idx = allocator.getLocalIndex(result);
      if (idx >= 0) {
        LocalTeeOp::create(builder, loc, static_cast<uint32_t>(idx),
                           resultType);
      }
    }

    emittedToStack.insert(result);
  }

  /// Emit a binary operation
  template <typename WasmStackOp>
  void emitBinaryOp(Operation *srcOp, Value lhs, Value rhs, Value result) {
    Location loc = srcOp->getLoc();
    Type resultType = result.getType();

    // Emit operands if not already on stack
    // Special case: if both operands are the same value, we need to handle
    // the stack carefully - one may be on stack, but we need two copies
    if (lhs == rhs) {
      emitOperandIfNeeded(lhs);
      // For the second operand with same value, always use local.get if
      // available
      int idx = allocator.getLocalIndex(lhs);
      if (idx >= 0) {
        LocalGetOp::create(builder, loc, static_cast<uint32_t>(idx),
                           lhs.getType());
      } else {
        // Fallback: re-emit the defining operation (shouldn't happen for
        // well-stackified code)
        if (Operation *defOp = lhs.getDefiningOp()) {
          emitOperation(defOp);
        }
      }
    } else {
      emitOperandIfNeeded(lhs);
      emitOperandIfNeeded(rhs);
    }

    // Emit the operation
    WasmStackOp::create(builder, loc, TypeAttr::get(resultType));

    // Check if this value needs tee
    if (needsTee.contains(result)) {
      int idx = allocator.getLocalIndex(result);
      if (idx >= 0) {
        LocalTeeOp::create(builder, loc, static_cast<uint32_t>(idx),
                           resultType);
      }
    }

    emittedToStack.insert(result);
  }

  /// Emit an operand value if it's not already on stack
  void emitOperandIfNeeded(Value value) {
    // If already emitted to stack, nothing to do
    if (emittedToStack.contains(value))
      return;

    // If it has a local, emit local.get
    int idx = allocator.getLocalIndex(value);
    if (idx >= 0) {
      LocalGetOp::create(builder, value.getLoc(), static_cast<uint32_t>(idx),
                         value.getType());
      return;
    }

    // Otherwise, we need to emit the defining operation
    // This should have been handled by stackification, but as a fallback:
    if (Operation *defOp = value.getDefiningOp()) {
      emitOperation(defOp);
    }
  }

  /// Generate a unique label for control flow structures
  std::string generateLabel(StringRef prefix) {
    return (prefix + "_" + Twine(labelCounter++)).str();
  }

  /// RAII guard to save/restore emittedToStack state for control flow regions
  /// Each WebAssembly control flow region has its own stack frame
  class ScopedStackState {
    WasmStackEmitter &emitter;
    DenseSet<Value> savedState;

  public:
    ScopedStackState(WasmStackEmitter &emitter)
        : emitter(emitter), savedState(emitter.emittedToStack) {}
    ~ScopedStackState() { emitter.emittedToStack = std::move(savedState); }
  };

  /// RAII guard for label stack management.
  /// Ensures labels are always pushed on entry and popped on exit of control
  /// flow regions. This guarantees that labelStack.back() always refers to
  /// the innermost enclosing control flow structure.
  ///
  /// Key invariant: When processing a loop body, labelStack.back() is always
  /// the loop's own label. Even if the loop body contains nested blocks/ifs,
  /// those nested structures use their own ScopedLabel which is destroyed
  /// before we return to the loop's block_return handling.
  class ScopedLabel {
    WasmStackEmitter &emitter;

  public:
    /// Push a label for a control flow structure
    /// @param label The unique label string (e.g., "block_0", "loop_1")
    /// @param isLoop True for loop (br continues), false for block/if (br
    /// exits)
    ScopedLabel(WasmStackEmitter &emitter, const std::string &label,
                bool isLoop)
        : emitter(emitter) {
      emitter.labelStack.push_back({label, isLoop});
    }

    ~ScopedLabel() {
      assert(!emitter.labelStack.empty() &&
             "ScopedLabel destroyed with empty label stack - mismatched "
             "push/pop");
      emitter.labelStack.pop_back();
    }

    // Non-copyable, non-movable
    ScopedLabel(const ScopedLabel &) = delete;
    ScopedLabel &operator=(const ScopedLabel &) = delete;
  };

  /// Get the branch label for a given exit level.
  /// Exit level 0 = innermost enclosing block/loop
  /// Exit level 1 = next outer enclosing structure, etc.
  ///
  /// Precondition: exitLevel < labelStack.size()
  /// This precondition should always hold for valid WasmSSA IR, as the
  /// SCF-to-WasmSSA lowering generates correct exit levels.
  std::string getLabelForExitLevel(unsigned exitLevel) {
    assert(exitLevel < labelStack.size() &&
           "Exit level exceeds label stack depth - invalid WasmSSA IR");
    // Labels are indexed from the top of the stack (innermost first)
    return labelStack[labelStack.size() - 1 - exitLevel].first;
  }

  /// Emit a WasmSSA block operation
  void emitBlock(wasmssa::BlockOp blockOp) {
    Location loc = blockOp.getLoc();

    // 1. Emit input values to stack BEFORE entering the block
    // These become the block's parameters in WebAssembly
    for (Value input : blockOp.getInputs()) {
      emitOperandIfNeeded(input);
    }

    // Generate label for this block
    std::string label = generateLabel("block");

    // 2. Extract param types from the block's inputs
    SmallVector<Attribute> paramTypes;
    for (Value input : blockOp.getInputs()) {
      paramTypes.push_back(TypeAttr::get(input.getType()));
    }

    // 3. Extract result types from the target successor block's arguments
    SmallVector<Attribute> resultTypes;
    Block *target = blockOp.getTarget();
    for (BlockArgument arg : target->getArguments()) {
      resultTypes.push_back(TypeAttr::get(arg.getType()));
    }

    // Create WasmStack block with param and result types
    auto wasmBlock = BlockOp::create(builder, loc, builder.getStringAttr(label),
                                     builder.getArrayAttr(paramTypes),
                                     builder.getArrayAttr(resultTypes));

    // Create entry block for the WasmStack block
    Block *entryBlock = new Block();
    wasmBlock.getBody().push_back(entryBlock);

    // Save current insertion point and emittedToStack state
    OpBuilder::InsertionGuard guard(builder);
    ScopedStackState stackGuard(*this);
    // Push label - ScopedLabel ensures it's popped when we exit this scope
    ScopedLabel labelGuard(*this, label, /*isLoop=*/false);
    builder.setInsertionPointToStart(entryBlock);

    // 4. CFG linearization: process all blocks by following terminators
    // This handles multi-block regions where branch_if has else successors
    if (!blockOp.getBody().empty()) {
      Block *currentBlock = &blockOp.getBody().front();
      llvm::DenseSet<Block *> processed;

      while (currentBlock && !processed.contains(currentBlock)) {
        processed.insert(currentBlock);

        // Mark block arguments as available on stack
        for (BlockArgument arg : currentBlock->getArguments()) {
          emittedToStack.insert(arg);
        }

        // Emit all operations EXCEPT the terminator
        for (Operation &op : currentBlock->without_terminator()) {
          emitOperation(&op);
        }

        // Handle terminator and get next block to process
        Operation *terminator = currentBlock->getTerminator();
        if (terminator) {
          currentBlock =
              emitTerminatorAndGetNext(terminator, /*isInLoop=*/false);
        } else {
          currentBlock = nullptr;
        }
      }
    }
    // ScopedLabel destructor pops the label automatically
  }

  /// Emit a WasmSSA loop operation
  void emitLoop(wasmssa::LoopOp loopOp) {
    Location loc = loopOp.getLoc();

    // 1. Emit input values to stack BEFORE entering the loop
    // These become the loop's parameters in WebAssembly
    for (Value input : loopOp.getInputs()) {
      emitOperandIfNeeded(input);
    }

    // Generate label for this loop
    std::string label = generateLabel("loop");

    // 2. Extract param types from the loop's inputs
    SmallVector<Attribute> paramTypes;
    for (Value input : loopOp.getInputs()) {
      paramTypes.push_back(TypeAttr::get(input.getType()));
    }

    // 3. Extract result types from the target successor block's arguments
    SmallVector<Attribute> resultTypes;
    Block *target = loopOp.getTarget();
    for (BlockArgument arg : target->getArguments()) {
      resultTypes.push_back(TypeAttr::get(arg.getType()));
    }

    // Create WasmStack loop with param and result types
    auto wasmLoop = LoopOp::create(builder, loc, builder.getStringAttr(label),
                                   builder.getArrayAttr(paramTypes),
                                   builder.getArrayAttr(resultTypes));

    // Create entry block
    Block *entryBlock = new Block();
    wasmLoop.getBody().push_back(entryBlock);

    // Save current insertion point and emittedToStack state
    OpBuilder::InsertionGuard guard(builder);
    ScopedStackState stackGuard(*this);
    // Push label - ScopedLabel ensures it's popped when we exit this scope.
    // This is critical for correctness: block_return inside this loop will
    // use labelStack.back() to get the loop label for "br @loop" to continue.
    // Any nested blocks/ifs inside this loop will have their own ScopedLabel
    // that is destroyed before we return here.
    ScopedLabel labelGuard(*this, label, /*isLoop=*/true);
    builder.setInsertionPointToStart(entryBlock);

    // 4. CFG linearization: process all blocks by following terminators
    if (!loopOp.getBody().empty()) {
      Block *currentBlock = &loopOp.getBody().front();
      llvm::DenseSet<Block *> processed;

      while (currentBlock && !processed.contains(currentBlock)) {
        processed.insert(currentBlock);

        // Mark block arguments as available on stack
        for (BlockArgument arg : currentBlock->getArguments()) {
          emittedToStack.insert(arg);
        }

        // Emit all operations EXCEPT the terminator
        for (Operation &op : currentBlock->without_terminator()) {
          emitOperation(&op);
        }

        // Handle terminator and get next block to process
        Operation *terminator = currentBlock->getTerminator();
        if (terminator) {
          currentBlock =
              emitTerminatorAndGetNext(terminator, /*isInLoop=*/true);
        } else {
          currentBlock = nullptr;
        }
      }
    }
    // ScopedLabel destructor pops the label automatically
  }

  /// Emit a WasmSSA if operation
  void emitIf(wasmssa::IfOp ifOp) {
    Location loc = ifOp.getLoc();

    // 1. Emit additional inputs to stack BEFORE the condition
    // (In WebAssembly, params are consumed before condition)
    for (Value input : ifOp.getInputs()) {
      emitOperandIfNeeded(input);
    }

    // Emit the condition to the stack (always last, as it's popped first)
    emitOperandIfNeeded(ifOp.getCondition());

    // 2. Extract param types from the if's inputs (not including condition)
    SmallVector<Attribute> paramTypes;
    for (Value input : ifOp.getInputs()) {
      paramTypes.push_back(TypeAttr::get(input.getType()));
    }

    // 3. Extract result types from the target successor block's arguments
    SmallVector<Attribute> resultTypes;
    Block *target = ifOp.getTarget();
    for (BlockArgument arg : target->getArguments()) {
      resultTypes.push_back(TypeAttr::get(arg.getType()));
    }

    // Create WasmStack if with param and result types
    auto wasmIf = IfOp::create(builder, loc, builder.getArrayAttr(paramTypes),
                               builder.getArrayAttr(resultTypes));

    // Create then block
    Block *thenBlock = new Block();
    wasmIf.getThenBody().push_back(thenBlock);

    {
      OpBuilder::InsertionGuard guard(builder);
      ScopedStackState stackGuard(*this); // Each branch has its own stack
      builder.setInsertionPointToStart(thenBlock);

      // 4. Handle block arguments in then region
      if (!ifOp.getIf().empty()) {
        Block &bodyBlock = ifOp.getIf().front();
        for (BlockArgument arg : bodyBlock.getArguments()) {
          emittedToStack.insert(arg);
        }
      }

      // Emit then region operations
      if (!ifOp.getIf().empty()) {
        for (Operation &op : ifOp.getIf().front()) {
          emitOperation(&op);
        }
      }
    }

    // Create else block if present
    if (!ifOp.getElse().empty()) {
      Block *elseBlock = new Block();
      wasmIf.getElseBody().push_back(elseBlock);

      OpBuilder::InsertionGuard guard(builder);
      ScopedStackState stackGuard(*this); // Each branch has its own stack
      builder.setInsertionPointToStart(elseBlock);

      // Handle block arguments in else region
      Block &bodyBlock = ifOp.getElse().front();
      for (BlockArgument arg : bodyBlock.getArguments()) {
        emittedToStack.insert(arg);
      }

      for (Operation &op : ifOp.getElse().front()) {
        emitOperation(&op);
      }
    }
  }

  /// Emit a WasmSSA branch_if operation
  void emitBranchIf(wasmssa::BranchIfOp branchIfOp) {
    Location loc = branchIfOp.getLoc();

    // Emit the condition to the stack
    emitOperandIfNeeded(branchIfOp.getCondition());

    // Resolve the exit level to the actual enclosing block/loop label
    unsigned exitLevel = branchIfOp.getExitLevel();
    std::string label = getLabelForExitLevel(exitLevel);

    BrIfOp::create(builder, loc, builder.getAttr<FlatSymbolRefAttr>(label));
  }

  /// Handle a terminator operation and return the next block to process.
  /// Returns nullptr if processing should stop (e.g., block_return, exit
  /// branch). The isInLoop parameter indicates whether we're inside a loop
  /// body, which affects how block_return is handled (emit br to continue
  /// loop).
  Block *emitTerminatorAndGetNext(Operation *terminator, bool isInLoop) {
    Location loc = terminator->getLoc();

    // Handle structured control flow terminators (loop, block, if)
    // These have nested regions and a successor block
    if (auto loopOp = dyn_cast<wasmssa::LoopOp>(terminator)) {
      // Emit the loop (which recursively processes its body)
      emitLoop(loopOp);
      // Continue with the loop's successor block
      return loopOp.getTarget();
    }

    if (auto blockOp = dyn_cast<wasmssa::BlockOp>(terminator)) {
      // Emit the block (which recursively processes its body)
      emitBlock(blockOp);
      // Continue with the block's successor block
      return blockOp.getTarget();
    }

    if (auto ifOp = dyn_cast<wasmssa::IfOp>(terminator)) {
      // Emit the if (which recursively processes its body)
      emitIf(ifOp);
      // Continue with the if's successor block
      return ifOp.getTarget();
    }

    if (auto branchIfOp = dyn_cast<wasmssa::BranchIfOp>(terminator)) {
      // branch_if %cond to level N with args(...) else ^successor
      // In WebAssembly, br_if: if condition true, branch with args;
      // if false, fall through (args remain on stack).

      // 1. Emit exit args to stack first (they stay if branch not taken)
      for (Value arg : branchIfOp.getInputs()) {
        emitOperandIfNeeded(arg);
      }

      // 2. Emit condition (must be on top of stack for br_if)
      emitOperandIfNeeded(branchIfOp.getCondition());

      // 3. Emit br_if to the exit level
      unsigned exitLevel = branchIfOp.getExitLevel();
      std::string label = getLabelForExitLevel(exitLevel);
      BrIfOp::create(builder, loc, builder.getAttr<FlatSymbolRefAttr>(label));

      // 4. Return the else successor to continue processing (fallthrough path)
      return branchIfOp.getElseSuccessor();
    }

    if (auto blockReturnOp = dyn_cast<wasmssa::BlockReturnOp>(terminator)) {
      // Emit return values to stack
      for (Value input : blockReturnOp.getInputs()) {
        emitOperandIfNeeded(input);
      }

      // If inside a loop, emit br to continue the loop.
      // INVARIANT: labelStack.back() is always the correct loop label here.
      // This is guaranteed by ScopedLabel RAII in emitLoop/emitBlock:
      // - Any nested blocks/ifs inside this loop have their own ScopedLabel
      // - Those ScopedLabels are destroyed before we return to this scope
      // - Therefore labelStack.back() is always the enclosing loop's label
      if (isInLoop && !labelStack.empty()) {
        assert(labelStack.back().second &&
               "isInLoop is true but labelStack.back() is not a loop label");
        std::string loopLabel = labelStack.back().first;
        BrOp::create(builder, loc,
                     builder.getAttr<FlatSymbolRefAttr>(loopLabel));
      }
      // Else: values on stack, control flows to block end naturally

      return nullptr; // Stop processing this CFG path
    }

    // Handle function return
    if (auto returnOp = dyn_cast<wasmssa::ReturnOp>(terminator)) {
      // Emit return operands to ensure they're on the stack
      for (Value operand : returnOp.getOperands()) {
        emitOperandIfNeeded(operand);
      }
      ReturnOp::create(builder, loc);
      return nullptr; // Stop processing - function ends
    }

    // For other terminators, just return nullptr
    return nullptr;
  }

  /// Emit a comparison operation (2 inputs, 1 i32 output)
  template <typename WasmStackOp>
  void emitCompareOp(Operation *srcOp, Value lhs, Value rhs) {
    Location loc = srcOp->getLoc();
    Type operandType = lhs.getType();

    if (lhs == rhs) {
      emitOperandIfNeeded(lhs);
      int idx = allocator.getLocalIndex(lhs);
      if (idx >= 0) {
        LocalGetOp::create(builder, loc, static_cast<uint32_t>(idx),
                           lhs.getType());
      } else if (Operation *defOp = lhs.getDefiningOp()) {
        emitOperation(defOp);
      }
    } else {
      emitOperandIfNeeded(lhs);
      emitOperandIfNeeded(rhs);
    }

    WasmStackOp::create(builder, loc, TypeAttr::get(operandType));
  }

  /// Emit a test operation (1 input, 1 i32 output)
  template <typename WasmStackOp>
  void emitTestOp(Operation *srcOp, Value input) {
    Location loc = srcOp->getLoc();
    Type inputType = input.getType();

    emitOperandIfNeeded(input);
    WasmStackOp::create(builder, loc, TypeAttr::get(inputType));
  }

  /// Emit a unary operation (1 input, 1 output of same type)
  template <typename WasmStackOp>
  void emitUnaryOp(Operation *srcOp, Value input, Value result) {
    Location loc = srcOp->getLoc();
    Type resultType = result.getType();

    emitOperandIfNeeded(input);
    WasmStackOp::create(builder, loc, TypeAttr::get(resultType));

    if (needsTee.contains(result)) {
      int idx = allocator.getLocalIndex(result);
      if (idx >= 0) {
        LocalTeeOp::create(builder, loc, static_cast<uint32_t>(idx),
                           resultType);
      }
    }
    emittedToStack.insert(result);
  }

  /// Emit a WasmSSA local_get operation (from source dialect)
  void emitSourceLocalGet(wasmssa::LocalGetOp localGetOp) {
    Location loc = localGetOp.getLoc();
    Value localRef = localGetOp.getLocalVar();
    Value result = localGetOp.getResult();

    int idx = allocator.getLocalIndex(localRef);
    if (idx >= 0) {
      LocalGetOp::create(builder, loc, static_cast<uint32_t>(idx),
                         result.getType());
    }

    if (needsTee.contains(result)) {
      int resIdx = allocator.getLocalIndex(result);
      if (resIdx >= 0) {
        LocalTeeOp::create(builder, loc, static_cast<uint32_t>(resIdx),
                           result.getType());
      }
    }
    emittedToStack.insert(result);
  }

  /// Emit a WasmSSA local_set operation (from source dialect)
  void emitSourceLocalSet(wasmssa::LocalSetOp localSetOp) {
    Location loc = localSetOp.getLoc();
    Value localRef = localSetOp.getLocalVar();
    Value value = localSetOp.getValue();

    emitOperandIfNeeded(value);

    int idx = allocator.getLocalIndex(localRef);
    if (idx >= 0) {
      LocalSetOp::create(builder, loc, static_cast<uint32_t>(idx),
                         value.getType());
    }
  }

  /// Emit a WasmSSA local_tee operation (from source dialect)
  void emitSourceLocalTee(wasmssa::LocalTeeOp localTeeOp) {
    Location loc = localTeeOp.getLoc();
    Value localRef = localTeeOp.getLocalVar();
    Value value = localTeeOp.getValue();
    Value result = localTeeOp.getResult();

    emitOperandIfNeeded(value);

    int idx = allocator.getLocalIndex(localRef);
    if (idx >= 0) {
      LocalTeeOp::create(builder, loc, static_cast<uint32_t>(idx),
                         value.getType());
    }
    emittedToStack.insert(result);
  }

  /// Emit a wami.load operation
  void emitLoad(wami::LoadOp loadOp) {
    Location loc = loadOp.getLoc();
    Value addr = loadOp.getAddress();
    Value result = loadOp.getResult();
    Type resultType = result.getType();

    emitOperandIfNeeded(addr);

    // Emit appropriate load instruction based on type
    if (resultType.isInteger(32)) {
      I32LoadOp::create(builder, loc, builder.getI32IntegerAttr(0),
                        builder.getI32IntegerAttr(4),
                        TypeAttr::get(resultType));
    } else if (resultType.isInteger(64)) {
      I64LoadOp::create(builder, loc, builder.getI32IntegerAttr(0),
                        builder.getI32IntegerAttr(8),
                        TypeAttr::get(resultType));
    } else if (resultType.isF32()) {
      F32LoadOp::create(builder, loc, builder.getI32IntegerAttr(0),
                        builder.getI32IntegerAttr(4),
                        TypeAttr::get(resultType));
    } else if (resultType.isF64()) {
      F64LoadOp::create(builder, loc, builder.getI32IntegerAttr(0),
                        builder.getI32IntegerAttr(8),
                        TypeAttr::get(resultType));
    }

    if (needsTee.contains(result)) {
      int idx = allocator.getLocalIndex(result);
      if (idx >= 0) {
        LocalTeeOp::create(builder, loc, static_cast<uint32_t>(idx),
                           resultType);
      }
    }
    emittedToStack.insert(result);
  }

  /// Emit a wami.store operation
  void emitStore(wami::StoreOp storeOp) {
    Location loc = storeOp.getLoc();
    Value addr = storeOp.getAddress();
    Value value = storeOp.getValue();
    Type valueType = value.getType();

    emitOperandIfNeeded(addr);
    emitOperandIfNeeded(value);

    // Emit appropriate store instruction based on type
    if (valueType.isInteger(32)) {
      I32StoreOp::create(builder, loc, builder.getI32IntegerAttr(0),
                         builder.getI32IntegerAttr(4),
                         TypeAttr::get(valueType));
    } else if (valueType.isInteger(64)) {
      I64StoreOp::create(builder, loc, builder.getI32IntegerAttr(0),
                         builder.getI32IntegerAttr(8),
                         TypeAttr::get(valueType));
    } else if (valueType.isF32()) {
      F32StoreOp::create(builder, loc, builder.getI32IntegerAttr(0),
                         builder.getI32IntegerAttr(4),
                         TypeAttr::get(valueType));
    } else if (valueType.isF64()) {
      F64StoreOp::create(builder, loc, builder.getI32IntegerAttr(0),
                         builder.getI32IntegerAttr(8),
                         TypeAttr::get(valueType));
    }
  }

  /// Emit a WasmSSA call operation
  void emitCall(wasmssa::FuncCallOp callOp) {
    Location loc = callOp.getLoc();

    // Emit all operands to the stack
    for (Value operand : callOp.getOperands()) {
      emitOperandIfNeeded(operand);
    }

    // Get function type from operands and results
    SmallVector<Type> inputTypes;
    for (Value operand : callOp.getOperands()) {
      inputTypes.push_back(operand.getType());
    }
    SmallVector<Type> resultTypes;
    for (Value result : callOp.getResults()) {
      resultTypes.push_back(result.getType());
    }

    FunctionType funcType =
        FunctionType::get(builder.getContext(), inputTypes, resultTypes);

    CallOp::create(builder, loc, callOp.getCalleeAttr(),
                   TypeAttr::get(funcType));

    // Mark results as emitted to stack
    for (Value result : callOp.getResults()) {
      if (needsTee.contains(result)) {
        int idx = allocator.getLocalIndex(result);
        if (idx >= 0) {
          LocalTeeOp::create(builder, loc, static_cast<uint32_t>(idx),
                             result.getType());
        }
      }
      emittedToStack.insert(result);
    }
  }
};

//===----------------------------------------------------------------------===//
// TreeWalker - Core Stackification Algorithm
//===----------------------------------------------------------------------===//

/// TreeWalker performs LLVM-style stackification by building expression trees.
/// It walks operations bottom-up and tries to move definitions immediately
/// before their uses, creating a stack-based evaluation order.
class TreeWalker {
  /// Operations that have been "stackified" - their result is on the
  /// implicit value stack and will be consumed by the next operation.
  DenseSet<Operation *> stackifiedOps;

  /// Values that need to be stored in locals (all uses via local.get)
  DenseSet<Value> needsLocal;

  /// Values that should use local.tee (first use from stack, rest via
  /// local.get)
  DenseSet<Value> needsTee;

  /// Use count analysis
  const UseCountAnalysis &useCount;

public:
  TreeWalker(const UseCountAnalysis &useCount) : useCount(useCount) {}

  /// Get the set of values that need locals (all uses via local.get)
  const DenseSet<Value> &getValuesNeedingLocals() const { return needsLocal; }

  /// Get the set of values that should use tee (first use from stack)
  const DenseSet<Value> &getValuesNeedingTee() const { return needsTee; }

  /// Check if an operation has been stackified
  bool isStackified(Operation *op) const { return stackifiedOps.contains(op); }

  /// Process a block, attempting to stackify operations
  void processBlock(Block &block) {
    // Collect operations in reverse order (bottom-up processing)
    SmallVector<Operation *> ops;
    for (Operation &op : block) {
      ops.push_back(&op);
    }

    // Process from last to first
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      processOperation(*it);
    }
  }

  /// Process a single operation, trying to stackify its operands
  void processOperation(Operation *op) {
    // Process operands left-to-right so that after reordering:
    // - Left operand's definition ends up further from use (pushed first)
    // - Right operand's definition ends up immediately before use (pushed
    // second) This ensures stack order: [lhs, rhs] with rhs on top. Binary ops
    // compute: bottom op top = lhs op rhs = CORRECT
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      Value operand = op->getOperand(i);
      Operation *defOp = operand.getDefiningOp();

      // Block arguments can't be stackified - they need local.get
      if (!defOp) {
        needsLocal.insert(operand);
        continue;
      }

      // Already stackified operations need special handling for additional uses
      if (stackifiedOps.contains(defOp)) {
        if (shouldRematerialize(defOp)) {
          // Clone cheap ops for additional uses (e.g., second use of same
          // const)
          OpBuilder builder(op);
          Operation *clone = builder.clone(*defOp);
          op->setOperand(i, clone->getResult(0));
          stackifiedOps.insert(clone);
          processOperation(clone);
        }
        // Non-rematerializable ops that are already stackified: the first use
        // consumes from stack, other uses need locals (handled by tee/local
        // logic when defOp was first processed)
        continue;
      }

      // Try to stackify this operand (single-use values)
      if (canStackify(defOp, op)) {
        // Move the defining operation immediately before this operation
        defOp->moveBefore(op);
        stackifiedOps.insert(defOp);

        // Recursively process the moved operation's operands
        processOperation(defOp);
      } else if (shouldRematerialize(defOp)) {
        // Multi-use rematerializable: move original for first use
        // (subsequent uses will hit the stackifiedOps check above and clone)
        defOp->moveBefore(op);
        stackifiedOps.insert(defOp);
        processOperation(defOp);
      } else {
        // Can't stackify directly - check if we can use tee
        // Tee is useful when one use can consume from stack, others from local
        if (canUseTee(defOp, operand)) {
          needsTee.insert(operand);
        } else {
          needsLocal.insert(operand);
        }
      }
    }
  }

private:
  /// Check if a value can benefit from local.tee
  /// This is true when the value's defining op can be moved to immediately
  /// before ONE of its uses (which will consume from stack)
  bool canUseTee(Operation *defOp, Value value) {
    // Must have multiple uses
    if (useCount.hasSingleUse(value))
      return false;

    // Check if any use is immediately after defOp (or can be made so)
    for (Operation *user : value.getUsers()) {
      // Check if we can move defOp immediately before this user
      if (defOp->getBlock() == user->getBlock() && isSafeToMove(defOp, user)) {
        return true; // At least one use can consume from stack
      }
    }
    return false;
  }

  /// Check if defOp can be stackified (moved immediately before useOp)
  bool canStackify(Operation *defOp, Operation *useOp) {
    // Must be single-use to stackify directly
    if (defOp->getNumResults() != 1)
      return false;

    Value result = defOp->getResult(0);
    if (!useCount.hasSingleUse(result))
      return false;

    // Check if it's safe to move (no dependency hazards)
    if (!isSafeToMove(defOp, useOp))
      return false;

    return true;
  }
};

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class WasmStackTypeConverter : public TypeConverter {
public:
  WasmStackTypeConverter(MLIRContext *ctx) {
    // Value types pass through unchanged
    addConversion([](IntegerType t) { return t; });
    addConversion([](FloatType t) { return t; });

    // MemRef types convert to i32 (linear memory pointer)
    addConversion(
        [ctx](MemRefType t) -> Type { return IntegerType::get(ctx, 32); });

    // Function types pass through
    addConversion([](FunctionType t) { return t; });

    // Index type converts to i32
    addConversion(
        [ctx](IndexType t) -> Type { return IntegerType::get(ctx, 32); });
  }
};

//===----------------------------------------------------------------------===//
// ConvertToWasmStack Pass
//===----------------------------------------------------------------------===//

class ConvertToWasmStack
    : public impl::ConvertToWasmStackBase<ConvertToWasmStack> {
public:
  using impl::ConvertToWasmStackBase<
      ConvertToWasmStack>::ConvertToWasmStackBase;

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *ctx = module.getContext();

    llvm::errs() << "ConvertToWasmStack pass running on module\n";
    llvm::errs().flush();

    // Collect functions to process
    SmallVector<wasmssa::FuncOp> funcsToConvert;
    module.walk(
        [&](wasmssa::FuncOp funcOp) { funcsToConvert.push_back(funcOp); });

    // Create builder for emitting new operations
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    // Process each WasmSSA function
    for (wasmssa::FuncOp funcOp : funcsToConvert) {
      // Skip empty functions
      if (funcOp.getBody().empty())
        continue;

      // Analyze use counts for the function
      Block &entryBlock = funcOp.getBody().front();
      UseCountAnalysis useCount(entryBlock);

      // Run the TreeWalker to stackify operations
      TreeWalker walker(useCount);
      walker.processBlock(entryBlock);

      // Get values that need locals or tee
      const auto &needsLocal = walker.getValuesNeedingLocals();
      const auto &needsTee = walker.getValuesNeedingTee();

      // Allocate local indices
      LocalAllocator allocator;
      allocator.allocate(funcOp, needsLocal, needsTee);

      // Report allocation results
      if (!needsLocal.empty()) {
        llvm::errs() << "    Values needing locals: " << needsLocal.size()
                     << "\n";
      }
      if (!needsTee.empty()) {
        llvm::errs() << "    Values needing tee: " << needsTee.size() << "\n";
      }
      if (allocator.getNumLocals() > allocator.getNumParams()) {
        llvm::errs() << "    Allocated locals: "
                     << (allocator.getNumLocals() - allocator.getNumParams())
                     << " (params: " << allocator.getNumParams()
                     << ", total: " << allocator.getNumLocals() << ")\n";
      }

      // Emit WasmStack function
      WasmStackEmitter emitter(builder, allocator, needsTee);
      emitter.emitFunction(funcOp);

      // Restore insertion point to module body for next function
      builder.setInsertionPointToEnd(module.getBody());

      // Remove the original WasmSSA function
      funcOp.erase();
    }
  }
};

} // namespace mlir::wasmstack
