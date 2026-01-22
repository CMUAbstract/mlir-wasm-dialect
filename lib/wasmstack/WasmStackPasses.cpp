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
        localTypes.push_back(arg.getType());
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

public:
  WasmStackEmitter(OpBuilder &builder, const LocalAllocator &allocator,
                   const DenseSet<Value> &needsTee)
      : builder(builder), allocator(allocator), needsTee(needsTee) {}

  /// Emit a WasmStack function from a WasmSSA function
  FuncOp emitFunction(wasmssa::FuncOp srcFunc) {
    Location loc = srcFunc.getLoc();
    MLIRContext *ctx = builder.getContext();

    // Get function type
    FunctionType funcType = srcFunc.getFunctionType();

    // Create WasmStack function
    auto dstFunc =
        builder.create<FuncOp>(loc, srcFunc.getName(), funcType, StringAttr());

    // Create entry block
    Block *entryBlock = new Block();
    dstFunc.getBody().push_back(entryBlock);
    builder.setInsertionPointToStart(entryBlock);

    // Emit local declarations for introduced locals (not parameters)
    for (unsigned i = allocator.getNumParams(); i < allocator.getNumLocals();
         ++i) {
      Type localType = allocator.getLocalType(i);
      builder.create<LocalOp>(loc, static_cast<uint32_t>(i), localType);
    }

    // Emit operations from the source function
    if (!srcFunc.getBody().empty()) {
      Block &srcBlock = srcFunc.getBody().front();
      for (Operation &op : srcBlock) {
        emitOperation(&op);
      }
    }

    return dstFunc;
  }

  /// Emit a single operation
  void emitOperation(Operation *op) {
    Location loc = op->getLoc();

    // Handle different operation types
    if (auto constOp = dyn_cast<wasmssa::ConstOp>(op)) {
      emitConst(constOp);
    } else if (auto addOp = dyn_cast<wasmssa::AddOp>(op)) {
      emitBinaryOp<AddOp>(addOp, addOp.getLhs(), addOp.getRhs(),
                          addOp.getResult());
    } else if (auto subOp = dyn_cast<wasmssa::SubOp>(op)) {
      emitBinaryOp<SubOp>(subOp, subOp.getLhs(), subOp.getRhs(),
                          subOp.getResult());
    } else if (auto mulOp = dyn_cast<wasmssa::MulOp>(op)) {
      emitBinaryOp<MulOp>(mulOp, mulOp.getLhs(), mulOp.getRhs(),
                          mulOp.getResult());
    } else if (isa<wasmssa::ReturnOp>(op)) {
      builder.create<ReturnOp>(loc);
    } else if (auto blockOp = dyn_cast<wasmssa::BlockOp>(op)) {
      emitBlock(blockOp);
    } else if (auto loopOp = dyn_cast<wasmssa::LoopOp>(op)) {
      emitLoop(loopOp);
    } else if (auto ifOp = dyn_cast<wasmssa::IfOp>(op)) {
      emitIf(ifOp);
    } else if (auto branchIfOp = dyn_cast<wasmssa::BranchIfOp>(op)) {
      emitBranchIf(branchIfOp);
    } else if (isa<wasmssa::BlockReturnOp>(op)) {
      // Block return is implicit - just falls through
      // The values should already be on stack
    }
    // TODO: Add more operation types (comparisons, etc.)
  }

private:
  /// Emit a constant operation
  void emitConst(wasmssa::ConstOp constOp) {
    Location loc = constOp.getLoc();
    Type resultType = constOp.getResult().getType();
    Attribute value = constOp.getValueAttr();

    if (resultType.isInteger(32)) {
      auto intVal = cast<IntegerAttr>(value).getInt();
      builder.create<I32ConstOp>(
          loc, builder.getI32IntegerAttr(static_cast<int32_t>(intVal)));
    } else if (resultType.isInteger(64)) {
      auto intVal = cast<IntegerAttr>(value).getInt();
      builder.create<I64ConstOp>(loc, builder.getI64IntegerAttr(intVal));
    } else if (resultType.isF32()) {
      auto floatVal = cast<FloatAttr>(value).getValueAsDouble();
      builder.create<F32ConstOp>(
          loc, builder.getF32FloatAttr(static_cast<float>(floatVal)));
    } else if (resultType.isF64()) {
      auto floatVal = cast<FloatAttr>(value).getValueAsDouble();
      builder.create<F64ConstOp>(loc, builder.getF64FloatAttr(floatVal));
    }

    // Check if this value needs tee
    Value result = constOp.getResult();
    if (needsTee.contains(result)) {
      int idx = allocator.getLocalIndex(result);
      if (idx >= 0) {
        builder.create<LocalTeeOp>(loc, static_cast<uint32_t>(idx), resultType);
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
        builder.create<LocalGetOp>(loc, static_cast<uint32_t>(idx),
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
    builder.create<WasmStackOp>(loc, TypeAttr::get(resultType));

    // Check if this value needs tee
    if (needsTee.contains(result)) {
      int idx = allocator.getLocalIndex(result);
      if (idx >= 0) {
        builder.create<LocalTeeOp>(loc, static_cast<uint32_t>(idx), resultType);
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
      builder.create<LocalGetOp>(value.getLoc(), static_cast<uint32_t>(idx),
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
    static unsigned counter = 0;
    return (prefix + "_" + Twine(counter++)).str();
  }

  /// Emit a WasmSSA block operation
  void emitBlock(wasmssa::BlockOp blockOp) {
    Location loc = blockOp.getLoc();

    // Generate label for this block
    std::string label = generateLabel("block");

    // Get result types (empty for now - blocks can have result types)
    SmallVector<Attribute> resultTypes;

    // Create WasmStack block
    auto wasmBlock = builder.create<BlockOp>(loc, builder.getStringAttr(label),
                                             builder.getArrayAttr(resultTypes));

    // Create entry block for the WasmStack block
    Block *entryBlock = new Block();
    wasmBlock.getBody().push_back(entryBlock);

    // Save current insertion point
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(entryBlock);

    // Emit operations from the WasmSSA block body
    if (!blockOp.getBody().empty()) {
      for (Operation &op : blockOp.getBody().front()) {
        emitOperation(&op);
      }
    }
  }

  /// Emit a WasmSSA loop operation
  void emitLoop(wasmssa::LoopOp loopOp) {
    Location loc = loopOp.getLoc();

    // Generate label for this loop
    std::string label = generateLabel("loop");

    // Get result types
    SmallVector<Attribute> resultTypes;

    // Create WasmStack loop
    auto wasmLoop = builder.create<LoopOp>(loc, builder.getStringAttr(label),
                                           builder.getArrayAttr(resultTypes));

    // Create entry block
    Block *entryBlock = new Block();
    wasmLoop.getBody().push_back(entryBlock);

    // Save current insertion point
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(entryBlock);

    // Emit operations from the WasmSSA loop body
    if (!loopOp.getBody().empty()) {
      for (Operation &op : loopOp.getBody().front()) {
        emitOperation(&op);
      }
    }
  }

  /// Emit a WasmSSA if operation
  void emitIf(wasmssa::IfOp ifOp) {
    Location loc = ifOp.getLoc();

    // Emit the condition to the stack
    emitOperandIfNeeded(ifOp.getCondition());

    // Get result types
    SmallVector<Attribute> resultTypes;

    // Create WasmStack if
    auto wasmIf = builder.create<IfOp>(loc, builder.getArrayAttr(resultTypes));

    // Create then block
    Block *thenBlock = new Block();
    wasmIf.getThenBody().push_back(thenBlock);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(thenBlock);

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
      builder.setInsertionPointToStart(elseBlock);

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

    // For now, we use a placeholder label - proper label resolution
    // would require tracking the label stack during emission
    // The exitLevel attribute tells us how many nesting levels to exit
    unsigned exitLevel = branchIfOp.getExitLevel();
    std::string label = "level_" + std::to_string(exitLevel);

    builder.create<BrIfOp>(loc, builder.getAttr<FlatSymbolRefAttr>(label));
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
    // Process operands right-to-left (matching WebAssembly stack order)
    // The rightmost operand should be pushed last (on top of stack)
    for (int i = op->getNumOperands() - 1; i >= 0; --i) {
      Value operand = op->getOperand(i);
      Operation *defOp = operand.getDefiningOp();

      // Block arguments can't be stackified - they need local.get
      if (!defOp) {
        needsLocal.insert(operand);
        continue;
      }

      // Already stackified operations are already handled
      if (stackifiedOps.contains(defOp)) {
        continue;
      }

      // Try to stackify this operand
      if (canStackify(defOp, op)) {
        // Move the defining operation immediately before this operation
        defOp->moveBefore(op);
        stackifiedOps.insert(defOp);

        // Recursively process the moved operation's operands
        processOperation(defOp);
      } else if (shouldRematerialize(defOp) &&
                 !useCount.hasSingleUse(operand)) {
        // Clone cheap operations instead of using locals
        OpBuilder builder(op);
        Operation *clone = builder.clone(*defOp);
        op->setOperand(i, clone->getResult(0));
        stackifiedOps.insert(clone);

        // Process the clone's operands
        processOperation(clone);
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
      llvm::errs() << "  Processing function: " << funcOp.getName() << "\n";

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
