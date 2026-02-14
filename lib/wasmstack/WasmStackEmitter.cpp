//===- WasmStackEmitter.cpp - WasmStack code emitter ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the WasmStackEmitter class which emits WasmStack
// operations from stackified WasmSSA/WAMI operations.
//
//===----------------------------------------------------------------------===//

#include "wasmstack/WasmStackEmitter.h"
#include "WAMI/WAMIOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "wasmstack/WasmConstUtils.h"
#include "llvm/ADT/Twine.h"

namespace mlir::wasmstack {

static bool isNoOpCastPair(Type srcType, Type dstType) {
  if (srcType == dstType)
    return true;

  if ((srcType.isIndex() && dstType.isInteger(32)) ||
      (dstType.isIndex() && srcType.isInteger(32)))
    return true;

  if (auto srcInt = dyn_cast<IntegerType>(srcType)) {
    if (auto dstInt = dyn_cast<IntegerType>(dstType))
      return srcInt.getWidth() == dstInt.getWidth();
  }

  if (auto srcFloat = dyn_cast<FloatType>(srcType)) {
    if (auto dstFloat = dyn_cast<FloatType>(dstType))
      return srcFloat.getWidth() == dstFloat.getWidth();
  }

  return false;
}

static Type toWasmStackType(Type type) {
  if (auto cont = dyn_cast<wami::ContType>(type))
    return ContRefType::get(type.getContext(), cont.getTypeName());
  if (auto func = dyn_cast<wami::FuncRefType>(type))
    return FuncRefType::get(type.getContext(), func.getFuncName());
  return type;
}

//===----------------------------------------------------------------------===//
// RAII Guards
//===----------------------------------------------------------------------===//

/// RAII guard to save/restore emittedToStack state for control flow regions
/// Each WebAssembly control flow region has its own stack frame.
/// In WebAssembly, when entering a control flow construct (if, block, loop),
/// the inner stack starts fresh - outer stack values are NOT accessible
/// except through locals. We must clear emittedToStack on entry so that
/// values from the outer scope are re-emitted (via local.get or
/// rematerialization) inside the inner scope.
class WasmStackEmitter::ScopedStackState {
  WasmStackEmitter &emitter;
  DenseSet<Value> savedState;

public:
  ScopedStackState(WasmStackEmitter &emitter)
      : emitter(emitter), savedState(emitter.emittedToStack) {
    // Clear emittedToStack for the inner scope - WebAssembly control flow
    // regions have their own stack frame
    emitter.emittedToStack.clear();
  }
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
class WasmStackEmitter::ScopedLabel {
  WasmStackEmitter &emitter;

public:
  /// Push a label for a control flow structure
  /// @param label The unique label string (e.g., "block_0", "loop_1")
  /// @param isLoop True for loop (br continues), false for block/if (br
  /// exits)
  ScopedLabel(WasmStackEmitter &emitter, const std::string &label, bool isLoop)
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

//===----------------------------------------------------------------------===//
// WasmStackEmitter Implementation
//===----------------------------------------------------------------------===//

FuncOp WasmStackEmitter::emitFunction(wasmssa::FuncOp srcFunc) {
  Location loc = srcFunc.getLoc();

  // Get function type
  FunctionType funcType = srcFunc.getFunctionType();

  StringAttr exportName;
  if (srcFunc.getExported())
    exportName = builder.getStringAttr(srcFunc.getSymName());

  // Create WasmStack function
  auto dstFunc =
      FuncOp::create(builder, loc, srcFunc.getName(), funcType, exportName);

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
    bool isEntryBlock = true;

    while (currentBlock && !processed.contains(currentBlock)) {
      processed.insert(currentBlock);

      if (isEntryBlock) {
        // For the entry block, arguments are function parameters which are
        // already in locals - just mark them as available for local.get.
        for (BlockArgument arg : currentBlock->getArguments()) {
          emittedToStack.insert(arg);
        }
        isEntryBlock = false;
      } else {
        // For successor blocks (after control flow), arguments represent
        // values on the stack. If they have locals allocated (multi-use or
        // used in nested control flow), emit local.set to save them.
        // IMPORTANT: Process in REVERSE order because stack is LIFO.
        auto args = currentBlock->getArguments();
        for (auto it = args.rbegin(); it != args.rend(); ++it) {
          BlockArgument arg = *it;
          int idx = allocator.getLocalIndex(arg);
          if (idx >= 0) {
            // Set the value to local (consumes from stack)
            LocalSetOp::create(
                builder, arg.getLoc(), static_cast<uint32_t>(idx),
                allocator.getLocalType(static_cast<unsigned>(idx)));
          } else {
            // No local allocated - this arg is used only once immediately
            // Keep it on stack by marking as emitted
            emittedToStack.insert(arg);
          }
        }
      }

      // Emit all operations EXCEPT the terminator
      for (Operation &op : currentBlock->without_terminator()) {
        if (failed)
          break;
        emitOperationAndDropUnused(&op);
        if (failed)
          break;
      }

      if (failed)
        break;

      // Handle terminator and get next block to process
      Operation *terminator = currentBlock->getTerminator();
      if (terminator) {
        currentBlock = emitTerminatorAndGetNext(terminator, /*isInLoop=*/false);
      } else {
        currentBlock = nullptr;
      }
    }
  }

  return dstFunc;
}

void WasmStackEmitter::emitOperationAndDropUnused(Operation *op) {
  bool hasUnusedResults = op->getNumResults() > 0 && op->use_empty();
  emitOperation(op);
  if (failed)
    return;
  if (!hasUnusedResults)
    return;
  dropUnusedResults(op);
}

void WasmStackEmitter::dropUnusedResults(Operation *op) {
  Location loc = op->getLoc();
  // Pop in reverse result order so we consume current stack top first.
  for (Value result : llvm::reverse(op->getResults())) {
    if (!emittedToStack.contains(result))
      continue;
    DropOp::create(builder, loc, toWasmStackType(result.getType()));
    emittedToStack.erase(result);
  }
}

void WasmStackEmitter::fail(Operation *op, StringRef message) {
  if (failed)
    return;
  failed = true;
  op->emitError(message);
}

void WasmStackEmitter::materializeResult(Location loc, Value result) {
  int idx = allocator.getLocalIndex(result);

  // No local allocated: keep result on value stack.
  if (idx < 0) {
    emittedToStack.insert(result);
    return;
  }

  // Tee policy: keep stack value for first consumer while also storing it.
  if (needsTee.contains(result)) {
    LocalTeeOp::create(builder, loc, static_cast<uint32_t>(idx),
                       allocator.getLocalType(static_cast<unsigned>(idx)));
    emittedToStack.insert(result);
    return;
  }

  // Local-only policy: materialize into local and keep stack clean.
  LocalSetOp::create(builder, loc, static_cast<uint32_t>(idx),
                     allocator.getLocalType(static_cast<unsigned>(idx)));
  emittedToStack.erase(result);
}

void WasmStackEmitter::emitOperation(Operation *op) {
  if (failed)
    return;

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
    emitCompareOp<EqOp>(eqOp, eqOp.getLhs(), eqOp.getRhs(), eqOp.getResult());
  } else if (auto neOp = dyn_cast<wasmssa::NeOp>(op)) {
    emitCompareOp<NeOp>(neOp, neOp.getLhs(), neOp.getRhs(), neOp.getResult());
  } else if (auto ltSIOp = dyn_cast<wasmssa::LtSIOp>(op)) {
    emitCompareOp<LtSOp>(ltSIOp, ltSIOp.getLhs(), ltSIOp.getRhs(),
                         ltSIOp.getResult());
  } else if (auto ltUIOp = dyn_cast<wasmssa::LtUIOp>(op)) {
    emitCompareOp<LtUOp>(ltUIOp, ltUIOp.getLhs(), ltUIOp.getRhs(),
                         ltUIOp.getResult());
  } else if (auto leSIOp = dyn_cast<wasmssa::LeSIOp>(op)) {
    emitCompareOp<LeSOp>(leSIOp, leSIOp.getLhs(), leSIOp.getRhs(),
                         leSIOp.getResult());
  } else if (auto leUIOp = dyn_cast<wasmssa::LeUIOp>(op)) {
    emitCompareOp<LeUOp>(leUIOp, leUIOp.getLhs(), leUIOp.getRhs(),
                         leUIOp.getResult());
  } else if (auto gtSIOp = dyn_cast<wasmssa::GtSIOp>(op)) {
    emitCompareOp<GtSOp>(gtSIOp, gtSIOp.getLhs(), gtSIOp.getRhs(),
                         gtSIOp.getResult());
  } else if (auto gtUIOp = dyn_cast<wasmssa::GtUIOp>(op)) {
    emitCompareOp<GtUOp>(gtUIOp, gtUIOp.getLhs(), gtUIOp.getRhs(),
                         gtUIOp.getResult());
  } else if (auto geSIOp = dyn_cast<wasmssa::GeSIOp>(op)) {
    emitCompareOp<GeSOp>(geSIOp, geSIOp.getLhs(), geSIOp.getRhs(),
                         geSIOp.getResult());
  } else if (auto geUIOp = dyn_cast<wasmssa::GeUIOp>(op)) {
    emitCompareOp<GeUOp>(geUIOp, geUIOp.getLhs(), geUIOp.getRhs(),
                         geUIOp.getResult());
  }
  // Float comparison operations
  else if (auto ltOp = dyn_cast<wasmssa::LtOp>(op)) {
    emitCompareOp<FLtOp>(ltOp, ltOp.getLhs(), ltOp.getRhs(), ltOp.getResult());
  } else if (auto leOp = dyn_cast<wasmssa::LeOp>(op)) {
    emitCompareOp<FLeOp>(leOp, leOp.getLhs(), leOp.getRhs(), leOp.getResult());
  } else if (auto gtOp = dyn_cast<wasmssa::GtOp>(op)) {
    emitCompareOp<FGtOp>(gtOp, gtOp.getLhs(), gtOp.getRhs(), gtOp.getResult());
  } else if (auto geOp = dyn_cast<wasmssa::GeOp>(op)) {
    emitCompareOp<FGeOp>(geOp, geOp.getLhs(), geOp.getRhs(), geOp.getResult());
  }
  // Test operation
  else if (auto eqzOp = dyn_cast<wasmssa::EqzOp>(op)) {
    emitTestOp<EqzOp>(eqzOp, eqzOp.getInput(), eqzOp.getResult());
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
  // Type conversion operations
  else if (auto convertSOp = dyn_cast<wasmssa::ConvertSOp>(op)) {
    emitConvertOp(convertSOp, /*isSigned=*/true);
  } else if (auto convertUOp = dyn_cast<wasmssa::ConvertUOp>(op)) {
    emitConvertOp(convertUOp, /*isSigned=*/false);
  } else if (auto promoteOp = dyn_cast<wasmssa::PromoteOp>(op)) {
    emitPromoteOp(promoteOp);
  } else if (auto demoteOp = dyn_cast<wasmssa::DemoteOp>(op)) {
    emitDemoteOp(demoteOp);
  } else if (auto extendSI32Op = dyn_cast<wasmssa::ExtendSI32Op>(op)) {
    emitExtendI32Op(extendSI32Op, /*isSigned=*/true);
  } else if (auto extendUI32Op = dyn_cast<wasmssa::ExtendUI32Op>(op)) {
    emitExtendI32Op(extendUI32Op, /*isSigned=*/false);
  } else if (auto wrapOp = dyn_cast<wasmssa::WrapOp>(op)) {
    emitWrapOp(wrapOp);
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
  // WAMI type conversion operations
  else if (auto truncSOp = dyn_cast<wami::TruncSOp>(op)) {
    emitTruncOp(truncSOp, /*isSigned=*/true);
  } else if (auto truncUOp = dyn_cast<wami::TruncUOp>(op)) {
    emitTruncOp(truncUOp, /*isSigned=*/false);
  }
  // WAMI stack-switching operations
  else if (auto refFuncOp = dyn_cast<wami::RefFuncOp>(op)) {
    emitRefFunc(refFuncOp);
  } else if (auto refNullOp = dyn_cast<wami::RefNullOp>(op)) {
    emitRefNull(refNullOp);
  } else if (auto contNewOp = dyn_cast<wami::ContNewOp>(op)) {
    emitContNew(contNewOp);
  } else if (auto contBindOp = dyn_cast<wami::ContBindOp>(op)) {
    emitContBind(contBindOp);
  } else if (auto suspendOp = dyn_cast<wami::SuspendOp>(op)) {
    emitSuspend(suspendOp);
  } else if (auto resumeOp = dyn_cast<wami::ResumeOp>(op)) {
    emitResume(resumeOp);
  } else if (auto resumeThrowOp = dyn_cast<wami::ResumeThrowOp>(op)) {
    emitResumeThrow(resumeThrowOp);
  } else if (auto barrierOp = dyn_cast<wami::BarrierOp>(op)) {
    emitBarrier(barrierOp);
  }
  // WAMI select operation
  else if (auto selectOp = dyn_cast<wami::SelectOp>(op)) {
    // WebAssembly select stack order: [true_val, false_val, condition]
    // with condition on top.
    // Pops: condition (i32), false_val, true_val
    // Pushes: condition ? true_val : false_val
    emitOperandIfNeeded(selectOp.getTrueValue());
    if (failed)
      return;
    emitOperandIfNeeded(selectOp.getFalseValue());
    if (failed)
      return;
    emitOperandIfNeeded(selectOp.getCondition());
    if (failed)
      return;
    SelectOp::create(builder, loc, TypeAttr::get(selectOp.getType()));
    materializeResult(loc, selectOp.getResult());
  }
  // Global variable operations
  else if (auto globalGetOp = dyn_cast<wasmssa::GlobalGetOp>(op)) {
    emitGlobalGet(globalGetOp);
  }
  // Function call
  else if (auto callOp = dyn_cast<wasmssa::FuncCallOp>(op)) {
    emitCall(callOp);
  }
  // Canonicalized away in ideal pipelines, but keep explicit support so
  // stackification remains robust when reconciliation is incomplete.
  else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1) {
      fail(op, "unsupported unrealized_conversion_cast arity");
      return;
    }

    Value src = castOp.getOperand(0);
    Value dst = castOp.getResult(0);
    int idx = allocator.getLocalIndex(src);
    if (idx >= 0) {
      LocalGetOp::create(builder, loc, static_cast<uint32_t>(idx),
                         dst.getType());
      materializeResult(loc, dst);
      return;
    }

    emitOperandIfNeeded(src);
    if (failed)
      return;
    if (!isNoOpCastPair(src.getType(), dst.getType())) {
      fail(
          op,
          "unsupported unrealized_conversion_cast without local-backed source");
      return;
    }
    materializeResult(loc, dst);
  }
  // Control flow
  else if (auto returnOp = dyn_cast<wasmssa::ReturnOp>(op)) {
    // Emit return operands to ensure they're on the stack
    // (TreeWalker usually handles this, but we emit explicitly for
    // robustness)
    for (Value operand : returnOp.getOperands()) {
      emitOperandIfNeeded(operand);
      if (failed)
        return;
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
      if (failed)
        return;
    }
    // No explicit wasmstack instruction needed - values are now on stack
    // and control flows to the block's end
  } else {
    fail(op, "unhandled operation in stackification emitter");
  }
}

void WasmStackEmitter::emitConst(wasmssa::ConstOp constOp) {
  Location loc = constOp.getLoc();
  if (::mlir::failed(
          emitWasmStackConst(builder, loc, constOp.getValueAttr()))) {
    fail(constOp, "unsupported wasmssa.const type");
    return;
  }

  materializeResult(loc, constOp.getResult());
}

template <typename WasmStackOp>
void WasmStackEmitter::emitBinaryOp(Operation *srcOp, Value lhs, Value rhs,
                                    Value result) {
  Location loc = srcOp->getLoc();
  Type resultType = result.getType();

  // Emit operands if not already on stack
  // Special case: if both operands are the same value, we need to handle
  // the stack carefully - one may be on stack, but we need two copies
  if (lhs == rhs) {
    emitOperandIfNeeded(lhs);
    if (failed)
      return;
    // For the second operand with same value, always use local.get if
    // available
    int idx = allocator.getLocalIndex(lhs);
    if (idx >= 0) {
      LocalGetOp::create(builder, loc, static_cast<uint32_t>(idx),
                         lhs.getType());
    } else {
      fail(srcOp, "repeated operand requires a local-backed value");
      return;
    }
  } else {
    emitOperandIfNeeded(lhs);
    if (failed)
      return;
    emitOperandIfNeeded(rhs);
    if (failed)
      return;
  }

  // Emit the operation
  WasmStackOp::create(builder, loc, TypeAttr::get(resultType));
  materializeResult(loc, result);
}

void WasmStackEmitter::emitOperandIfNeeded(Value value) {
  if (failed)
    return;

  // If already emitted to stack, mark as consumed and return
  if (emittedToStack.contains(value)) {
    emittedToStack.erase(value); // Mark as consumed
    return;
  }

  // If it has a local, emit local.get
  int idx = allocator.getLocalIndex(value);
  if (idx >= 0) {
    LocalGetOp::create(builder, value.getLoc(), static_cast<uint32_t>(idx),
                       allocator.getLocalType(static_cast<unsigned>(idx)));
    return;
  }

  if (Operation *defOp = value.getDefiningOp()) {
    fail(defOp, "value is neither on stack nor local-backed");
    return;
  }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *contextOp = blockArg.getOwner()->getParentOp();
    fail(contextOp, "block argument is neither on stack nor local-backed");
    return;
  }
  fail(builder.getBlock()->getParentOp(),
       "operand is neither on stack nor local-backed");
}

void WasmStackEmitter::materializeEntryBlockArguments(Block &block) {
  // Block arguments represent values on stack at block entry.
  // Process in reverse because stack is LIFO.
  auto args = block.getArguments();
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    BlockArgument arg = *it;
    int idx = allocator.getLocalIndex(arg);
    if (idx >= 0) {
      LocalSetOp::create(builder, arg.getLoc(), static_cast<uint32_t>(idx),
                         allocator.getLocalType(static_cast<unsigned>(idx)));
    } else {
      emittedToStack.insert(arg);
    }
  }
}

std::string WasmStackEmitter::generateLabel(StringRef prefix) {
  return (prefix + "_" + Twine(labelCounter++)).str();
}

std::string WasmStackEmitter::getLabelForExitLevel(unsigned exitLevel,
                                                   Operation *contextOp) {
  if (exitLevel >= labelStack.size()) {
    fail(contextOp, "exit level exceeds label stack depth");
    return {};
  }
  // Labels are indexed from the top of the stack (innermost first)
  return labelStack[labelStack.size() - 1 - exitLevel].first;
}

void WasmStackEmitter::emitBlock(wasmssa::BlockOp blockOp) {
  Location loc = blockOp.getLoc();

  // 1. Emit input values to stack BEFORE entering the block
  // These become the block's parameters in WebAssembly
  for (Value input : blockOp.getInputs()) {
    emitOperandIfNeeded(input);
    if (failed)
      return;
  }

  // Generate label for this block
  std::string label = generateLabel("block");

  // 2. Extract param types from the block's inputs
  SmallVector<Attribute> paramTypes;
  for (Value input : blockOp.getInputs()) {
    paramTypes.push_back(TypeAttr::get(toWasmStackType(input.getType())));
  }

  // 3. Extract result types from the target successor block's arguments
  SmallVector<Attribute> resultTypes;
  Block *target = blockOp.getTarget();
  for (BlockArgument arg : target->getArguments()) {
    resultTypes.push_back(TypeAttr::get(toWasmStackType(arg.getType())));
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
      materializeEntryBlockArguments(*currentBlock);

      // Emit all operations EXCEPT the terminator
      for (Operation &op : currentBlock->without_terminator()) {
        emitOperationAndDropUnused(&op);
        if (failed)
          return;
      }

      // Handle terminator and get next block to process
      Operation *terminator = currentBlock->getTerminator();
      if (terminator) {
        currentBlock = emitTerminatorAndGetNext(terminator, /*isInLoop=*/false);
      } else {
        currentBlock = nullptr;
      }
      if (failed)
        return;
    }
  }
  // ScopedLabel destructor pops the label automatically
}

void WasmStackEmitter::emitLoop(wasmssa::LoopOp loopOp) {
  Location loc = loopOp.getLoc();

  // 1. Emit input values to stack BEFORE entering the loop
  // These become the loop's parameters in WebAssembly
  for (Value input : loopOp.getInputs()) {
    emitOperandIfNeeded(input);
    if (failed)
      return;
  }

  // Generate label for this loop
  std::string label = generateLabel("loop");

  // 2. Extract param types from the loop's inputs
  SmallVector<Attribute> paramTypes;
  for (Value input : loopOp.getInputs()) {
    paramTypes.push_back(TypeAttr::get(toWasmStackType(input.getType())));
  }

  // 3. Extract result types from the target successor block's arguments
  SmallVector<Attribute> resultTypes;
  Block *target = loopOp.getTarget();
  for (BlockArgument arg : target->getArguments()) {
    resultTypes.push_back(TypeAttr::get(toWasmStackType(arg.getType())));
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
      materializeEntryBlockArguments(*currentBlock);

      // Emit all operations EXCEPT the terminator
      for (Operation &op : currentBlock->without_terminator()) {
        emitOperationAndDropUnused(&op);
        if (failed)
          return;
      }

      // Handle terminator and get next block to process
      Operation *terminator = currentBlock->getTerminator();
      if (terminator) {
        currentBlock = emitTerminatorAndGetNext(terminator, /*isInLoop=*/true);
      } else {
        currentBlock = nullptr;
      }
      if (failed)
        return;
    }
  }
  // ScopedLabel destructor pops the label automatically
}

void WasmStackEmitter::emitIf(wasmssa::IfOp ifOp) {
  Location loc = ifOp.getLoc();

  // 1. Emit additional inputs to stack BEFORE the condition
  // (In WebAssembly, params are consumed before condition)
  for (Value input : ifOp.getInputs()) {
    emitOperandIfNeeded(input);
    if (failed)
      return;
  }

  // Emit the condition to the stack (always last, as it's popped first)
  emitOperandIfNeeded(ifOp.getCondition());
  if (failed)
    return;

  // 2. Extract param types from the if's inputs (not including condition)
  SmallVector<Attribute> paramTypes;
  for (Value input : ifOp.getInputs()) {
    paramTypes.push_back(TypeAttr::get(toWasmStackType(input.getType())));
  }

  // 3. Extract result types from the target successor block's arguments
  SmallVector<Attribute> resultTypes;
  Block *target = ifOp.getTarget();
  for (BlockArgument arg : target->getArguments()) {
    resultTypes.push_back(TypeAttr::get(toWasmStackType(arg.getType())));
  }

  std::string label = generateLabel("if");

  // Create WasmStack if with param and result types.
  auto wasmIf = IfOp::create(builder, loc, builder.getStringAttr(label),
                             builder.getArrayAttr(paramTypes),
                             builder.getArrayAttr(resultTypes));

  // Push an explicit if-frame label so branch depth resolution inside then/else
  // matches WebAssembly semantics.
  ScopedLabel labelGuard(*this, label, /*isLoop=*/false);

  auto emitIfRegion = [&](Region &srcRegion, Region &dstRegion,
                          bool requireEntryBlock) {
    if (srcRegion.empty()) {
      if (requireEntryBlock)
        dstRegion.push_back(new Block());
      return;
    }

    Block *entryBlock = new Block();
    dstRegion.push_back(entryBlock);

    OpBuilder::InsertionGuard guard(builder);
    ScopedStackState stackGuard(*this);
    builder.setInsertionPointToStart(entryBlock);

    Block *currentBlock = &srcRegion.front();
    llvm::DenseSet<Block *> processed;

    while (currentBlock && !processed.contains(currentBlock)) {
      processed.insert(currentBlock);
      materializeEntryBlockArguments(*currentBlock);

      for (Operation &op : currentBlock->without_terminator()) {
        emitOperationAndDropUnused(&op);
        if (failed)
          return;
      }

      Operation *terminator = currentBlock->getTerminator();
      if (terminator) {
        currentBlock = emitTerminatorAndGetNext(terminator, /*isInLoop=*/false);
      } else {
        currentBlock = nullptr;
      }

      if (failed)
        return;
    }
  };

  emitIfRegion(ifOp.getIf(), wasmIf.getThenBody(), /*requireEntryBlock=*/true);
  if (failed)
    return;

  emitIfRegion(ifOp.getElse(), wasmIf.getElseBody(),
               /*requireEntryBlock=*/false);
}

void WasmStackEmitter::emitBranchIf(wasmssa::BranchIfOp branchIfOp) {
  Location loc = branchIfOp.getLoc();

  // Emit the condition to the stack
  emitOperandIfNeeded(branchIfOp.getCondition());
  if (failed)
    return;

  // Resolve the exit level to the actual enclosing block/loop label
  unsigned exitLevel = branchIfOp.getExitLevel();
  std::string label =
      getLabelForExitLevel(exitLevel, branchIfOp.getOperation());
  if (failed)
    return;

  BrIfOp::create(builder, loc, builder.getAttr<FlatSymbolRefAttr>(label));
}

Block *WasmStackEmitter::emitTerminatorAndGetNext(Operation *terminator,
                                                  bool isInLoop) {
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
    // Stack order must be: [args..., condition] with condition on top.

    // 1. Emit exit args to stack first (they stay if branch not taken).
    for (Value arg : branchIfOp.getInputs()) {
      emitOperandIfNeeded(arg);
      if (failed)
        return nullptr;
    }

    // 2. Emit condition (must be on top of stack for br_if).
    emitOperandIfNeeded(branchIfOp.getCondition());
    if (failed)
      return nullptr;

    // 3. Emit br_if to the exit level
    unsigned exitLevel = branchIfOp.getExitLevel();
    std::string label =
        getLabelForExitLevel(exitLevel, branchIfOp.getOperation());
    if (failed)
      return nullptr;
    BrIfOp::create(builder, loc, builder.getAttr<FlatSymbolRefAttr>(label));

    // 4. Return the else successor to continue processing (fallthrough path)
    return branchIfOp.getElseSuccessor();
  }

  if (auto blockReturnOp = dyn_cast<wasmssa::BlockReturnOp>(terminator)) {
    // Emit return values to stack
    for (Value input : blockReturnOp.getInputs()) {
      emitOperandIfNeeded(input);
      if (failed)
        return nullptr;
    }

    // If inside a loop, emit br to continue the loop.
    // INVARIANT: labelStack.back() is always the correct loop label here.
    if (isInLoop) {
      if (labelStack.empty()) {
        fail(blockReturnOp.getOperation(),
             "in-loop block_return has no enclosing loop label");
        return nullptr;
      }
      if (!labelStack.back().second) {
        fail(blockReturnOp.getOperation(),
             "in-loop block_return resolved to non-loop label");
        return nullptr;
      }

      std::string loopLabel = labelStack.back().first;
      BrOp::create(builder, loc, builder.getAttr<FlatSymbolRefAttr>(loopLabel));
    }
    // Else: values on stack, control flows to block end naturally

    return nullptr; // Stop processing this CFG path
  }

  // Handle function return
  if (auto returnOp = dyn_cast<wasmssa::ReturnOp>(terminator)) {
    // Emit return operands to ensure they're on the stack
    for (Value operand : returnOp.getOperands()) {
      emitOperandIfNeeded(operand);
      if (failed)
        return nullptr;
    }
    ReturnOp::create(builder, loc);
    return nullptr; // Stop processing - function ends
  }

  // For other terminators, just return nullptr
  return nullptr;
}

template <typename WasmStackOp>
void WasmStackEmitter::emitCompareOp(Operation *srcOp, Value lhs, Value rhs,
                                     Value result) {
  Location loc = srcOp->getLoc();
  Type operandType = lhs.getType();

  if (lhs == rhs) {
    emitOperandIfNeeded(lhs);
    if (failed)
      return;
    int idx = allocator.getLocalIndex(lhs);
    if (idx >= 0) {
      LocalGetOp::create(builder, loc, static_cast<uint32_t>(idx),
                         lhs.getType());
    } else {
      fail(srcOp, "repeated compare operand requires a local-backed value");
      return;
    }
  } else {
    emitOperandIfNeeded(lhs);
    if (failed)
      return;
    emitOperandIfNeeded(rhs);
    if (failed)
      return;
  }

  WasmStackOp::create(builder, loc, TypeAttr::get(operandType));
  materializeResult(loc, result);
}

template <typename WasmStackOp>
void WasmStackEmitter::emitTestOp(Operation *srcOp, Value input, Value result) {
  Location loc = srcOp->getLoc();
  Type inputType = input.getType();

  emitOperandIfNeeded(input);
  if (failed)
    return;
  WasmStackOp::create(builder, loc, TypeAttr::get(inputType));
  materializeResult(loc, result);
}

template <typename WasmStackOp>
void WasmStackEmitter::emitUnaryOp(Operation *srcOp, Value input,
                                   Value result) {
  Location loc = srcOp->getLoc();
  Type resultType = result.getType();

  emitOperandIfNeeded(input);
  if (failed)
    return;
  WasmStackOp::create(builder, loc, TypeAttr::get(resultType));
  materializeResult(loc, result);
}

void WasmStackEmitter::emitConvertOp(Operation *srcOp, bool isSigned) {
  Location loc = srcOp->getLoc();
  Value input = srcOp->getOperand(0);
  Value result = srcOp->getResult(0);
  Type inputType = input.getType();
  Type resultType = result.getType();

  emitOperandIfNeeded(input);
  if (failed)
    return;

  // Emit the appropriate convert instruction based on types
  if (resultType.isF32()) {
    if (inputType.isInteger(32)) {
      if (isSigned)
        F32ConvertI32SOp::create(builder, loc, inputType, resultType);
      else
        F32ConvertI32UOp::create(builder, loc, inputType, resultType);
    } else if (inputType.isInteger(64)) {
      if (isSigned)
        F32ConvertI64SOp::create(builder, loc, inputType, resultType);
      else
        F32ConvertI64UOp::create(builder, loc, inputType, resultType);
    }
  } else if (resultType.isF64()) {
    if (inputType.isInteger(32)) {
      if (isSigned)
        F64ConvertI32SOp::create(builder, loc, inputType, resultType);
      else
        F64ConvertI32UOp::create(builder, loc, inputType, resultType);
    } else if (inputType.isInteger(64)) {
      if (isSigned)
        F64ConvertI64SOp::create(builder, loc, inputType, resultType);
      else
        F64ConvertI64UOp::create(builder, loc, inputType, resultType);
    }
  }
  materializeResult(loc, result);
}

void WasmStackEmitter::emitPromoteOp(wasmssa::PromoteOp promoteOp) {
  Location loc = promoteOp.getLoc();
  Value input = promoteOp.getInput();
  Value result = promoteOp.getResult();
  Type inputType = input.getType();
  Type resultType = result.getType();

  emitOperandIfNeeded(input);
  if (failed)
    return;
  F64PromoteF32Op::create(builder, loc, inputType, resultType);
  materializeResult(loc, result);
}

void WasmStackEmitter::emitDemoteOp(wasmssa::DemoteOp demoteOp) {
  Location loc = demoteOp.getLoc();
  Value input = demoteOp.getInput();
  Value result = demoteOp.getResult();
  Type inputType = input.getType();
  Type resultType = result.getType();

  emitOperandIfNeeded(input);
  if (failed)
    return;
  F32DemoteF64Op::create(builder, loc, inputType, resultType);
  materializeResult(loc, result);
}

void WasmStackEmitter::emitExtendI32Op(Operation *srcOp, bool isSigned) {
  Location loc = srcOp->getLoc();
  Value input = srcOp->getOperand(0);
  Value result = srcOp->getResult(0);
  Type inputType = input.getType();
  Type resultType = result.getType();

  emitOperandIfNeeded(input);
  if (failed)
    return;

  if (isSigned)
    I64ExtendI32SOp::create(builder, loc, inputType, resultType);
  else
    I64ExtendI32UOp::create(builder, loc, inputType, resultType);
  materializeResult(loc, result);
}

void WasmStackEmitter::emitWrapOp(wasmssa::WrapOp wrapOp) {
  Location loc = wrapOp.getLoc();
  Value input = wrapOp.getInput();
  Value result = wrapOp.getResult();
  Type inputType = input.getType();
  Type resultType = result.getType();

  emitOperandIfNeeded(input);
  if (failed)
    return;
  I32WrapI64Op::create(builder, loc, inputType, resultType);
  materializeResult(loc, result);
}

void WasmStackEmitter::emitTruncOp(Operation *srcOp, bool isSigned) {
  Location loc = srcOp->getLoc();
  Value input = srcOp->getOperand(0);
  Value result = srcOp->getResult(0);
  Type inputType = input.getType();
  Type resultType = result.getType();

  emitOperandIfNeeded(input);
  if (failed)
    return;

  // Emit appropriate truncation instruction based on types
  if (resultType.isInteger(32)) {
    if (inputType.isF32()) {
      if (isSigned)
        I32TruncF32SOp::create(builder, loc, inputType, resultType);
      else
        I32TruncF32UOp::create(builder, loc, inputType, resultType);
    } else if (inputType.isF64()) {
      if (isSigned)
        I32TruncF64SOp::create(builder, loc, inputType, resultType);
      else
        I32TruncF64UOp::create(builder, loc, inputType, resultType);
    }
  } else if (resultType.isInteger(64)) {
    if (inputType.isF32()) {
      if (isSigned)
        I64TruncF32SOp::create(builder, loc, inputType, resultType);
      else
        I64TruncF32UOp::create(builder, loc, inputType, resultType);
    } else if (inputType.isF64()) {
      if (isSigned)
        I64TruncF64SOp::create(builder, loc, inputType, resultType);
      else
        I64TruncF64UOp::create(builder, loc, inputType, resultType);
    }
  }
  materializeResult(loc, result);
}

void WasmStackEmitter::emitSourceLocalGet(wasmssa::LocalGetOp localGetOp) {
  Location loc = localGetOp.getLoc();
  Value localRef = localGetOp.getLocalVar();
  Value result = localGetOp.getResult();

  int idx = allocator.getLocalIndex(localRef);
  if (idx >= 0) {
    LocalGetOp::create(builder, loc, static_cast<uint32_t>(idx),
                       allocator.getLocalType(static_cast<unsigned>(idx)));
  } else {
    fail(localGetOp.getOperation(),
         "source local.get references a value without allocated local");
    return;
  }
  materializeResult(loc, result);
}

void WasmStackEmitter::emitSourceLocalSet(wasmssa::LocalSetOp localSetOp) {
  Location loc = localSetOp.getLoc();
  Value localRef = localSetOp.getLocalVar();
  Value value = localSetOp.getValue();

  emitOperandIfNeeded(value);
  if (failed)
    return;

  int idx = allocator.getLocalIndex(localRef);
  if (idx >= 0) {
    LocalSetOp::create(builder, loc, static_cast<uint32_t>(idx),
                       allocator.getLocalType(static_cast<unsigned>(idx)));
  } else {
    fail(localSetOp.getOperation(),
         "source local.set references a value without allocated local");
  }
}

void WasmStackEmitter::emitSourceLocalTee(wasmssa::LocalTeeOp localTeeOp) {
  Location loc = localTeeOp.getLoc();
  Value localRef = localTeeOp.getLocalVar();
  Value value = localTeeOp.getValue();
  Value result = localTeeOp.getResult();

  emitOperandIfNeeded(value);
  if (failed)
    return;

  int idx = allocator.getLocalIndex(localRef);
  if (idx >= 0) {
    LocalTeeOp::create(builder, loc, static_cast<uint32_t>(idx),
                       allocator.getLocalType(static_cast<unsigned>(idx)));
  } else {
    fail(localTeeOp.getOperation(),
         "source local.tee references a value without allocated local");
    return;
  }
  materializeResult(loc, result);
}

void WasmStackEmitter::emitLoad(wami::LoadOp loadOp) {
  Location loc = loadOp.getLoc();
  Value addr = loadOp.getAddress();
  Value result = loadOp.getResult();
  Type resultType = result.getType();

  emitOperandIfNeeded(addr);
  if (failed)
    return;

  // Emit appropriate load instruction based on type
  if (resultType.isInteger(32)) {
    I32LoadOp::create(builder, loc, builder.getI32IntegerAttr(0),
                      builder.getI32IntegerAttr(4), TypeAttr::get(resultType));
  } else if (resultType.isInteger(64)) {
    I64LoadOp::create(builder, loc, builder.getI32IntegerAttr(0),
                      builder.getI32IntegerAttr(8), TypeAttr::get(resultType));
  } else if (resultType.isF32()) {
    F32LoadOp::create(builder, loc, builder.getI32IntegerAttr(0),
                      builder.getI32IntegerAttr(4), TypeAttr::get(resultType));
  } else if (resultType.isF64()) {
    F64LoadOp::create(builder, loc, builder.getI32IntegerAttr(0),
                      builder.getI32IntegerAttr(8), TypeAttr::get(resultType));
  }
  materializeResult(loc, result);
}

void WasmStackEmitter::emitStore(wami::StoreOp storeOp) {
  Location loc = storeOp.getLoc();
  Value addr = storeOp.getAddress();
  Value value = storeOp.getValue();
  Type valueType = value.getType();

  emitOperandIfNeeded(addr);
  if (failed)
    return;
  emitOperandIfNeeded(value);
  if (failed)
    return;

  // Emit appropriate store instruction based on type
  if (valueType.isInteger(32)) {
    I32StoreOp::create(builder, loc, builder.getI32IntegerAttr(0),
                       builder.getI32IntegerAttr(4), TypeAttr::get(valueType));
  } else if (valueType.isInteger(64)) {
    I64StoreOp::create(builder, loc, builder.getI32IntegerAttr(0),
                       builder.getI32IntegerAttr(8), TypeAttr::get(valueType));
  } else if (valueType.isF32()) {
    F32StoreOp::create(builder, loc, builder.getI32IntegerAttr(0),
                       builder.getI32IntegerAttr(4), TypeAttr::get(valueType));
  } else if (valueType.isF64()) {
    F64StoreOp::create(builder, loc, builder.getI32IntegerAttr(0),
                       builder.getI32IntegerAttr(8), TypeAttr::get(valueType));
  }
}

void WasmStackEmitter::emitRefFunc(wami::RefFuncOp refFuncOp) {
  Location loc = refFuncOp.getLoc();
  RefFuncOp::create(builder, loc, refFuncOp.getFuncAttr());
  materializeResult(loc, refFuncOp.getResult());
}

void WasmStackEmitter::emitRefNull(wami::RefNullOp refNullOp) {
  Location loc = refNullOp.getLoc();
  RefNullOp::create(
      builder, loc,
      TypeAttr::get(toWasmStackType(refNullOp.getResult().getType())));
  materializeResult(loc, refNullOp.getResult());
}

void WasmStackEmitter::emitContNew(wami::ContNewOp contNewOp) {
  Location loc = contNewOp.getLoc();
  emitOperandIfNeeded(contNewOp.getFunc());
  if (failed)
    return;

  auto contType = contNewOp->getAttrOfType<FlatSymbolRefAttr>("cont_type");
  if (!contType) {
    fail(contNewOp, "missing cont_type attribute");
    return;
  }

  ContNewOp::create(builder, loc, contType);
  materializeResult(loc, contNewOp.getResult());
}

void WasmStackEmitter::emitContBind(wami::ContBindOp contBindOp) {
  Location loc = contBindOp.getLoc();
  ValueRange operands = contBindOp->getOperands();
  if (operands.empty()) {
    fail(contBindOp, "cont.bind requires a continuation operand");
    return;
  }

  emitOperandIfNeeded(operands.front());
  if (failed)
    return;

  ValueRange boundArgs = operands.drop_front();
  for (Value v : boundArgs) {
    emitOperandIfNeeded(v);
    if (failed)
      return;
  }

  auto srcType = contBindOp->getAttrOfType<FlatSymbolRefAttr>("src_cont_type");
  auto dstType = contBindOp->getAttrOfType<FlatSymbolRefAttr>("dst_cont_type");
  if (!srcType || !dstType) {
    fail(contBindOp, "missing src_cont_type/dst_cont_type attributes");
    return;
  }

  ContBindOp::create(builder, loc, srcType, dstType);
  materializeResult(loc, contBindOp.getResult());
}

void WasmStackEmitter::emitSuspend(wami::SuspendOp suspendOp) {
  Location loc = suspendOp.getLoc();
  for (Value v : suspendOp->getOperands()) {
    emitOperandIfNeeded(v);
    if (failed)
      return;
  }

  auto tag = suspendOp->getAttrOfType<FlatSymbolRefAttr>("tag");
  if (!tag) {
    fail(suspendOp, "missing tag attribute");
    return;
  }

  SuspendOp::create(builder, loc, tag);
  for (Value result : llvm::reverse(suspendOp->getResults()))
    materializeResult(loc, result);
}

void WasmStackEmitter::emitResume(wami::ResumeOp resumeOp) {
  Location loc = resumeOp.getLoc();
  ValueRange operands = resumeOp->getOperands();
  for (Value v : operands) {
    emitOperandIfNeeded(v);
    if (failed)
      return;
  }

  auto contType = resumeOp->getAttrOfType<FlatSymbolRefAttr>("cont_type");
  auto handlerAttrs = resumeOp->getAttrOfType<ArrayAttr>("handlers");
  if (!contType || !handlerAttrs) {
    fail(resumeOp, "missing cont_type/handlers attributes");
    return;
  }

  SmallVector<Attribute> handlers;
  handlers.reserve(handlerAttrs.size());
  for (Attribute attr : handlerAttrs) {
    if (auto onLabel = dyn_cast<wami::OnLabelHandlerAttr>(attr)) {
      int64_t level = onLabel.getLevel();
      if (level < 0) {
        fail(resumeOp, "on_label level must be non-negative");
        return;
      }

      std::string label =
          getLabelForExitLevel(static_cast<unsigned>(level), resumeOp);
      if (failed)
        return;

      SmallVector<Attribute> pair = {
          onLabel.getTag(),
          FlatSymbolRefAttr::get(builder.getContext(), label)};
      handlers.push_back(ArrayAttr::get(builder.getContext(), pair));
      continue;
    }

    if (auto onSwitch = dyn_cast<wami::OnSwitchHandlerAttr>(attr)) {
      SmallVector<Attribute> pair = {
          onSwitch.getTag(),
          FlatSymbolRefAttr::get(builder.getContext(), "switch")};
      handlers.push_back(ArrayAttr::get(builder.getContext(), pair));
      continue;
    }

    fail(resumeOp,
         "handlers must contain #wami.on_label or #wami.on_switch attributes");
    return;
  }

  ResumeOp::create(builder, loc, contType,
                   ArrayAttr::get(builder.getContext(), handlers));
  for (Value result : llvm::reverse(resumeOp->getResults()))
    materializeResult(loc, result);
}

void WasmStackEmitter::emitResumeThrow(wami::ResumeThrowOp resumeThrowOp) {
  Location loc = resumeThrowOp.getLoc();
  ValueRange operands = resumeThrowOp->getOperands();
  for (Value v : operands) {
    emitOperandIfNeeded(v);
    if (failed)
      return;
  }

  auto contType = resumeThrowOp->getAttrOfType<FlatSymbolRefAttr>("cont_type");
  auto handlerAttrs = resumeThrowOp->getAttrOfType<ArrayAttr>("handlers");
  if (!contType || !handlerAttrs) {
    fail(resumeThrowOp, "missing cont_type/handlers attributes");
    return;
  }

  SmallVector<Attribute> handlers;
  handlers.reserve(handlerAttrs.size());
  for (Attribute attr : handlerAttrs) {
    if (auto onLabel = dyn_cast<wami::OnLabelHandlerAttr>(attr)) {
      int64_t level = onLabel.getLevel();
      if (level < 0) {
        fail(resumeThrowOp, "on_label level must be non-negative");
        return;
      }

      std::string label =
          getLabelForExitLevel(static_cast<unsigned>(level), resumeThrowOp);
      if (failed)
        return;

      SmallVector<Attribute> pair = {
          onLabel.getTag(),
          FlatSymbolRefAttr::get(builder.getContext(), label)};
      handlers.push_back(ArrayAttr::get(builder.getContext(), pair));
      continue;
    }

    if (auto onSwitch = dyn_cast<wami::OnSwitchHandlerAttr>(attr)) {
      SmallVector<Attribute> pair = {
          onSwitch.getTag(),
          FlatSymbolRefAttr::get(builder.getContext(), "switch")};
      handlers.push_back(ArrayAttr::get(builder.getContext(), pair));
      continue;
    }

    fail(resumeThrowOp,
         "handlers must contain #wami.on_label or #wami.on_switch attributes");
    return;
  }

  ResumeThrowOp::create(builder, loc, contType,
                        ArrayAttr::get(builder.getContext(), handlers));
}

void WasmStackEmitter::emitBarrier(wami::BarrierOp barrierOp) {
  Location loc = barrierOp.getLoc();
  Region &body = barrierOp.getBody();
  if (body.empty()) {
    fail(barrierOp, "barrier body must contain one block");
    return;
  }

  Block &entry = body.front();
  for (Operation &op : entry.without_terminator()) {
    emitOperationAndDropUnused(&op);
    if (failed)
      return;
  }

  auto yieldOp = dyn_cast<wami::BarrierYieldOp>(entry.getTerminator());
  if (!yieldOp) {
    fail(barrierOp, "barrier body must terminate with wami.barrier.yield");
    return;
  }

  SmallVector<Type> inputTypes;
  inputTypes.reserve(yieldOp.getValues().size());
  for (Value v : yieldOp.getValues()) {
    emitOperandIfNeeded(v);
    if (failed)
      return;
    inputTypes.push_back(toWasmStackType(v.getType()));
  }

  SmallVector<Type> resultTypes;
  resultTypes.reserve(barrierOp.getResultTypes().size());
  for (Type t : barrierOp.getResultTypes())
    resultTypes.push_back(toWasmStackType(t));

  FunctionType barrierType =
      FunctionType::get(builder.getContext(), inputTypes, resultTypes);
  BarrierOp::create(builder, loc, TypeAttr::get(barrierType));
  for (Value result : llvm::reverse(barrierOp->getResults()))
    materializeResult(loc, result);
}

void WasmStackEmitter::emitGlobalGet(wasmssa::GlobalGetOp globalGetOp) {
  Location loc = globalGetOp.getLoc();
  Value result = globalGetOp.getResult();
  Type resultType = result.getType();

  // Get the global symbol name
  FlatSymbolRefAttr globalName = globalGetOp.getGlobalAttr();

  // Emit wasmstack.global.get operation
  GlobalGetOp::create(builder, loc, globalName, TypeAttr::get(resultType));
  materializeResult(loc, result);
}

void WasmStackEmitter::emitCall(wasmssa::FuncCallOp callOp) {
  Location loc = callOp.getLoc();

  // Emit all operands to the stack
  for (Value operand : callOp.getOperands()) {
    emitOperandIfNeeded(operand);
    if (failed)
      return;
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

  CallOp::create(builder, loc, callOp.getCalleeAttr(), TypeAttr::get(funcType));

  // WebAssembly pushes multi-value results with the last result on top.
  // Materialize in reverse so local-backed assignment preserves SSA order.
  for (Value result : llvm::reverse(callOp.getResults())) {
    materializeResult(loc, result);
  }
}

// Explicit template instantiations for all binary operations
template void WasmStackEmitter::emitBinaryOp<AddOp>(Operation *, Value, Value,
                                                    Value);
template void WasmStackEmitter::emitBinaryOp<SubOp>(Operation *, Value, Value,
                                                    Value);
template void WasmStackEmitter::emitBinaryOp<MulOp>(Operation *, Value, Value,
                                                    Value);
template void WasmStackEmitter::emitBinaryOp<FDivOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<DivSOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<DivUOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<RemSOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<RemUOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<AndOp>(Operation *, Value, Value,
                                                    Value);
template void WasmStackEmitter::emitBinaryOp<OrOp>(Operation *, Value, Value,
                                                   Value);
template void WasmStackEmitter::emitBinaryOp<XorOp>(Operation *, Value, Value,
                                                    Value);
template void WasmStackEmitter::emitBinaryOp<ShlOp>(Operation *, Value, Value,
                                                    Value);
template void WasmStackEmitter::emitBinaryOp<ShrSOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<ShrUOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<RotlOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<RotrOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<FMinOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<FMaxOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitBinaryOp<FCopysignOp>(Operation *, Value,
                                                          Value, Value);

// Explicit template instantiations for comparison operations
template void WasmStackEmitter::emitCompareOp<EqOp>(Operation *, Value, Value,
                                                    Value);
template void WasmStackEmitter::emitCompareOp<NeOp>(Operation *, Value, Value,
                                                    Value);
template void WasmStackEmitter::emitCompareOp<LtSOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<LtUOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<LeSOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<LeUOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<GtSOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<GtUOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<GeSOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<GeUOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<FLtOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<FLeOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<FGtOp>(Operation *, Value, Value,
                                                     Value);
template void WasmStackEmitter::emitCompareOp<FGeOp>(Operation *, Value, Value,
                                                     Value);

// Explicit template instantiations for test operations
template void WasmStackEmitter::emitTestOp<EqzOp>(Operation *, Value, Value);

// Explicit template instantiations for unary operations
template void WasmStackEmitter::emitUnaryOp<ClzOp>(Operation *, Value, Value);
template void WasmStackEmitter::emitUnaryOp<CtzOp>(Operation *, Value, Value);
template void WasmStackEmitter::emitUnaryOp<PopcntOp>(Operation *, Value,
                                                      Value);
template void WasmStackEmitter::emitUnaryOp<FAbsOp>(Operation *, Value, Value);
template void WasmStackEmitter::emitUnaryOp<FNegOp>(Operation *, Value, Value);
template void WasmStackEmitter::emitUnaryOp<FSqrtOp>(Operation *, Value, Value);
template void WasmStackEmitter::emitUnaryOp<FCeilOp>(Operation *, Value, Value);
template void WasmStackEmitter::emitUnaryOp<FFloorOp>(Operation *, Value,
                                                      Value);
template void WasmStackEmitter::emitUnaryOp<FTruncOp>(Operation *, Value,
                                                      Value);

} // namespace mlir::wasmstack
