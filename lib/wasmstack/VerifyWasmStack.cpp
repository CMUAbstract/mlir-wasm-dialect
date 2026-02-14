//===- VerifyWasmStack.cpp - WasmStack stack verification --------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a WebAssembly-spec compliant stack verification pass
// for the WasmStack dialect. It validates stack balance, type checking,
// control flow, and function signatures.
//
//===----------------------------------------------------------------------===//

#include "wasmstack/WasmStackDialect.h"
#include "wasmstack/WasmStackOps.h"
#include "wasmstack/WasmStackPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::wasmstack {

#define GEN_PASS_DEF_VERIFYWASMSTACK
#include "wasmstack/WasmStackPasses.h.inc"

//===----------------------------------------------------------------------===//
// Control Frame for Structured Control Flow
//===----------------------------------------------------------------------===//

namespace {

/// Represents a control frame for structured control flow (block, loop, if)
struct ControlFrame {
  enum Kind { Block, Loop, If, Else, Function };

  Kind kind;
  StringRef label;               // Label name (may be empty)
  SmallVector<Type> paramTypes;  // Types consumed on entry
  SmallVector<Type> resultTypes; // Types produced on exit
  size_t stackHeightAtEntry;     // Stack height when frame was entered
  bool unreachable = false;      // Frame entered unreachable state
  Operation *op = nullptr;       // The operation that created this frame

  /// Get the types expected when branching to this frame
  ArrayRef<Type> getBranchTypes() const {
    // For loops, branch goes to start (needs params)
    // For blocks/if/function, branch goes to end (needs results)
    return kind == Loop ? ArrayRef<Type>(paramTypes)
                        : ArrayRef<Type>(resultTypes);
  }
};

//===----------------------------------------------------------------------===//
// Stack Verifier
//===----------------------------------------------------------------------===//

class StackVerifier {
public:
  StackVerifier() = default;

  /// Verify a function
  LogicalResult verifyFunction(FuncOp funcOp);

private:
  /// Value stack (types from bottom to top)
  SmallVector<Type, 16> valueStack;

  /// Control frame stack
  SmallVector<ControlFrame, 8> controlStack;

  /// Map from label name to control frame index
  llvm::StringMap<size_t> labelToFrameIndex;

  /// Whether we're in unreachable code (polymorphic stack)
  bool unreachable = false;

  /// Current operation being verified (for error messages)
  Operation *currentOp = nullptr;

  /// MLIR context for creating locations
  MLIRContext *ctx = nullptr;

  // Helper methods
  void reset(MLIRContext *context);
  void pushValue(Type type);
  LogicalResult popValue(Type expected);
  LogicalResult popAnyValue(Type &actual);
  LogicalResult popValues(ArrayRef<Type> expected);
  LogicalResult checkStackHeight(size_t expected, StringRef context);
  bool isSubtype(Type actual, Type expected) const;
  Type getNullableContRefType(FlatSymbolRefAttr ref) const;
  Type getNonNullContRefType(FlatSymbolRefAttr ref) const;

  // Control flow helpers
  void pushControlFrame(ControlFrame frame);
  LogicalResult popControlFrame();
  ControlFrame *findLabelFrame(StringRef label);

  // Operation handlers
  LogicalResult verifyStackOp(Operation *op);
  LogicalResult handleBlockOp(BlockOp op);
  LogicalResult handleLoopOp(LoopOp op);
  LogicalResult handleIfOp(IfOp op);
  LogicalResult handleBrOp(BrOp op);
  LogicalResult handleBrIfOp(BrIfOp op);
  LogicalResult handleBrTableOp(BrTableOp op);
  LogicalResult handleReturnOp(ReturnOp op);
  LogicalResult handleUnreachableOp(UnreachableOp op);
  LogicalResult handleContNewOp(ContNewOp op);
  LogicalResult handleContBindOp(ContBindOp op);
  LogicalResult handleResumeOp(ResumeOp op);
  LogicalResult handleResumeThrowOp(ResumeThrowOp op);
  LogicalResult handleSuspendOp(SuspendOp op);
  LogicalResult handleSwitchOp(SwitchOp op);

  FailureOr<FunctionType> resolveTypeFunc(Operation *op, FlatSymbolRefAttr ref,
                                          StringRef context);
  FailureOr<FunctionType> resolveContSig(Operation *op, FlatSymbolRefAttr ref,
                                         StringRef context);
  FailureOr<FunctionType> resolveTagSig(Operation *op, FlatSymbolRefAttr ref,
                                        StringRef context);

  // Walk the body of a control frame
  LogicalResult walkBody(Region &region);

  // Error emission helpers
  InFlightDiagnostic emitError(StringRef message);
  std::string formatTypeList(ArrayRef<Type> types);
  std::string formatStack();
};

//===----------------------------------------------------------------------===//
// Stack Verifier Implementation
//===----------------------------------------------------------------------===//

void StackVerifier::reset(MLIRContext *context) {
  valueStack.clear();
  controlStack.clear();
  labelToFrameIndex.clear();
  unreachable = false;
  currentOp = nullptr;
  ctx = context;
}

void StackVerifier::pushValue(Type type) { valueStack.push_back(type); }

LogicalResult StackVerifier::popValue(Type expected) {
  // In unreachable (polymorphic) mode, we accept any type
  if (unreachable) {
    // Still pop from stack if available (for tracking purposes)
    if (!valueStack.empty() &&
        valueStack.size() > controlStack.back().stackHeightAtEntry) {
      valueStack.pop_back();
    }
    return success();
  }

  // Check for stack underflow (below current frame's entry height)
  if (valueStack.size() <= controlStack.back().stackHeightAtEntry) {
    return emitError("stack underflow: expected ")
           << expected << " but stack is empty at this frame level";
  }

  Type actual = valueStack.pop_back_val();

  // Type check
  if (!isSubtype(actual, expected)) {
    return emitError("type mismatch: expected ")
           << expected << " but got " << actual;
  }

  return success();
}

LogicalResult StackVerifier::popAnyValue(Type &actual) {
  if (unreachable) {
    if (!valueStack.empty() &&
        valueStack.size() > controlStack.back().stackHeightAtEntry) {
      actual = valueStack.pop_back_val();
    } else {
      actual = Type();
    }
    return success();
  }

  if (valueStack.size() <= controlStack.back().stackHeightAtEntry)
    return emitError("stack underflow: expected a value but stack is empty");

  actual = valueStack.pop_back_val();
  return success();
}

LogicalResult StackVerifier::popValues(ArrayRef<Type> expected) {
  // Pop in reverse order (rightmost type is on top of stack)
  for (Type type : llvm::reverse(expected)) {
    if (failed(popValue(type)))
      return failure();
  }
  return success();
}

LogicalResult StackVerifier::checkStackHeight(size_t expected,
                                              StringRef context) {
  if (unreachable)
    return success();

  size_t frameBase = controlStack.back().stackHeightAtEntry;
  size_t actualHeight = valueStack.size() - frameBase;

  if (actualHeight != expected) {
    return emitError("stack height mismatch at ")
           << context << ": expected " << expected << " values but got "
           << actualHeight;
  }
  return success();
}

bool StackVerifier::isSubtype(Type actual, Type expected) const {
  if (actual == expected)
    return true;

  auto actualNonNull = dyn_cast<ContRefNonNullType>(actual);
  auto expectedNullable = dyn_cast<ContRefType>(expected);
  if (actualNonNull && expectedNullable) {
    return actualNonNull.getTypeName() == expectedNullable.getTypeName();
  }

  return false;
}

Type StackVerifier::getNullableContRefType(FlatSymbolRefAttr ref) const {
  return ContRefType::get(ctx, ref);
}

Type StackVerifier::getNonNullContRefType(FlatSymbolRefAttr ref) const {
  return ContRefNonNullType::get(ctx, ref);
}

void StackVerifier::pushControlFrame(ControlFrame frame) {
  if (!frame.label.empty()) {
    labelToFrameIndex[frame.label] = controlStack.size();
  }
  controlStack.push_back(std::move(frame));
}

LogicalResult StackVerifier::popControlFrame() {
  if (controlStack.empty()) {
    return emitError("control frame underflow");
  }

  ControlFrame &frame = controlStack.back();

  // If not unreachable, verify stack matches expected results
  if (!frame.unreachable && !unreachable) {
    // Check we have exactly the result types on stack
    size_t frameBase = frame.stackHeightAtEntry;
    size_t actualHeight = valueStack.size() - frameBase;
    if (actualHeight != frame.resultTypes.size()) {
      return frame.op->emitError(
                 "stack height mismatch at frame exit: expected ")
             << frame.resultTypes.size() << " values but got " << actualHeight;
    }

    // Verify result types (but don't actually pop yet)
    for (size_t i = 0; i < frame.resultTypes.size(); ++i) {
      size_t stackIdx = valueStack.size() - frame.resultTypes.size() + i;
      if (!isSubtype(valueStack[stackIdx], frame.resultTypes[i])) {
        return frame.op->emitError("frame result type mismatch at index ")
               << i << ": expected " << frame.resultTypes[i] << " but got "
               << valueStack[stackIdx];
      }
    }
  }

  // Remove label mapping
  if (!frame.label.empty()) {
    labelToFrameIndex.erase(frame.label);
  }

  // Pop values down to frame entry, then push results
  valueStack.resize(frame.stackHeightAtEntry);
  for (Type type : frame.resultTypes) {
    valueStack.push_back(type);
  }

  // Restore unreachable state from parent frame
  controlStack.pop_back();
  unreachable = controlStack.empty() ? false : controlStack.back().unreachable;

  return success();
}

ControlFrame *StackVerifier::findLabelFrame(StringRef label) {
  auto it = labelToFrameIndex.find(label);
  if (it == labelToFrameIndex.end())
    return nullptr;
  return &controlStack[it->second];
}

InFlightDiagnostic StackVerifier::emitError(StringRef message) {
  if (currentOp) {
    return currentOp->emitError(message);
  }
  assert(ctx && "MLIRContext not initialized");
  return mlir::emitError(UnknownLoc::get(ctx), message);
}

std::string StackVerifier::formatTypeList(ArrayRef<Type> types) {
  std::string result = "[";
  llvm::interleave(
      types,
      [&](Type t) {
        llvm::raw_string_ostream os(result);
        t.print(os);
      },
      [&]() { result += ", "; });
  result += "]";
  return result;
}

std::string StackVerifier::formatStack() { return formatTypeList(valueStack); }

FailureOr<FunctionType> StackVerifier::resolveTypeFunc(Operation *op,
                                                       FlatSymbolRefAttr ref,
                                                       StringRef context) {
  auto typeFunc = SymbolTable::lookupNearestSymbolFrom<TypeFuncOp>(op, ref);
  if (!typeFunc) {
    op->emitError(context) << ": unknown wasmstack.type.func symbol " << ref;
    return failure();
  }
  return typeFunc.getType();
}

FailureOr<FunctionType> StackVerifier::resolveContSig(Operation *op,
                                                      FlatSymbolRefAttr ref,
                                                      StringRef context) {
  auto contType = SymbolTable::lookupNearestSymbolFrom<TypeContOp>(op, ref);
  if (!contType) {
    op->emitError(context) << ": unknown wasmstack.type.cont symbol " << ref;
    return failure();
  }
  return resolveTypeFunc(op, contType.getFuncTypeAttr(), context);
}

FailureOr<FunctionType> StackVerifier::resolveTagSig(Operation *op,
                                                     FlatSymbolRefAttr ref,
                                                     StringRef context) {
  auto tag = SymbolTable::lookupNearestSymbolFrom<TagOp>(op, ref);
  if (!tag) {
    op->emitError(context) << ": unknown wasmstack.tag symbol " << ref;
    return failure();
  }
  return tag.getType();
}

//===----------------------------------------------------------------------===//
// Operation Verification
//===----------------------------------------------------------------------===//

LogicalResult StackVerifier::verifyStackOp(Operation *op) {
  currentOp = op;

  // Handle control flow operations specially
  if (auto blockOp = dyn_cast<BlockOp>(op))
    return handleBlockOp(blockOp);
  if (auto loopOp = dyn_cast<LoopOp>(op))
    return handleLoopOp(loopOp);
  if (auto ifOp = dyn_cast<IfOp>(op))
    return handleIfOp(ifOp);
  if (auto brOp = dyn_cast<BrOp>(op))
    return handleBrOp(brOp);
  if (auto brIfOp = dyn_cast<BrIfOp>(op))
    return handleBrIfOp(brIfOp);
  if (auto brTableOp = dyn_cast<BrTableOp>(op))
    return handleBrTableOp(brTableOp);
  if (auto returnOp = dyn_cast<ReturnOp>(op))
    return handleReturnOp(returnOp);
  if (auto unreachableOp = dyn_cast<UnreachableOp>(op))
    return handleUnreachableOp(unreachableOp);
  if (auto contNewOp = dyn_cast<ContNewOp>(op))
    return handleContNewOp(contNewOp);
  if (auto contBindOp = dyn_cast<ContBindOp>(op))
    return handleContBindOp(contBindOp);
  if (auto resumeOp = dyn_cast<ResumeOp>(op))
    return handleResumeOp(resumeOp);
  if (auto resumeThrowOp = dyn_cast<ResumeThrowOp>(op))
    return handleResumeThrowOp(resumeThrowOp);
  if (auto suspendOp = dyn_cast<SuspendOp>(op))
    return handleSuspendOp(suspendOp);
  if (auto switchOp = dyn_cast<SwitchOp>(op))
    return handleSwitchOp(switchOp);

  // Handle operations with StackEffectOpInterface
  if (auto stackOp = dyn_cast<StackEffectOpInterface>(op)) {
    // Pop input types
    auto inputTypes = stackOp.getStackInputTypes();
    if (failed(popValues(inputTypes)))
      return failure();

    // Push output types
    auto outputTypes = stackOp.getStackOutputTypes();
    for (Type type : outputTypes) {
      pushValue(type);
    }
    return success();
  }

  // Operations without stack effects (e.g., locals declarations, nop)
  return success();
}

LogicalResult StackVerifier::handleBlockOp(BlockOp op) {
  // Pop block parameters from stack
  SmallVector<Type> paramTypes;
  for (auto attr : op.getParamTypes()) {
    paramTypes.push_back(cast<TypeAttr>(attr).getValue());
  }
  if (failed(popValues(paramTypes)))
    return failure();

  // Get result types
  SmallVector<Type> resultTypes;
  for (auto attr : op.getResultTypes()) {
    resultTypes.push_back(cast<TypeAttr>(attr).getValue());
  }

  // Push control frame
  ControlFrame frame;
  frame.kind = ControlFrame::Block;
  frame.label = op.getLabel();
  frame.paramTypes = std::move(paramTypes);
  frame.resultTypes = std::move(resultTypes);
  frame.stackHeightAtEntry = valueStack.size();
  frame.op = op;
  pushControlFrame(std::move(frame));

  // Push parameters back onto stack (available inside block)
  for (auto attr : op.getParamTypes()) {
    pushValue(cast<TypeAttr>(attr).getValue());
  }

  // Verify block body
  if (failed(walkBody(op.getBody())))
    return failure();

  // Pop control frame and push results
  return popControlFrame();
}

LogicalResult StackVerifier::handleLoopOp(LoopOp op) {
  // Pop loop parameters from stack
  SmallVector<Type> paramTypes;
  for (auto attr : op.getParamTypes()) {
    paramTypes.push_back(cast<TypeAttr>(attr).getValue());
  }
  if (failed(popValues(paramTypes)))
    return failure();

  // Get result types
  SmallVector<Type> resultTypes;
  for (auto attr : op.getResultTypes()) {
    resultTypes.push_back(cast<TypeAttr>(attr).getValue());
  }

  // Push control frame
  ControlFrame frame;
  frame.kind = ControlFrame::Loop;
  frame.label = op.getLabel();
  frame.paramTypes = std::move(paramTypes);
  frame.resultTypes = std::move(resultTypes);
  frame.stackHeightAtEntry = valueStack.size();
  frame.op = op;
  pushControlFrame(std::move(frame));

  // Push parameters back onto stack (available inside loop)
  for (auto attr : op.getParamTypes()) {
    pushValue(cast<TypeAttr>(attr).getValue());
  }

  // Verify loop body
  if (failed(walkBody(op.getBody())))
    return failure();

  // Pop control frame and push results
  return popControlFrame();
}

LogicalResult StackVerifier::handleIfOp(IfOp op) {
  // Pop condition (i32)
  if (failed(popValue(IntegerType::get(op.getContext(), 32))))
    return failure();

  // Pop additional parameters
  SmallVector<Type> paramTypes;
  for (auto attr : op.getParamTypes()) {
    paramTypes.push_back(cast<TypeAttr>(attr).getValue());
  }
  if (failed(popValues(paramTypes)))
    return failure();

  // Get result types
  SmallVector<Type> resultTypes;
  for (auto attr : op.getResultTypes()) {
    resultTypes.push_back(cast<TypeAttr>(attr).getValue());
  }

  // Save state for then/else branches
  size_t stackHeight = valueStack.size();
  bool wasUnreachable = unreachable;

  // Verify then branch
  {
    ControlFrame frame;
    frame.kind = ControlFrame::If;
    auto label = op.getLabel();
    if (label.has_value())
      frame.label = label.value();
    frame.paramTypes = paramTypes;
    frame.resultTypes = resultTypes;
    frame.stackHeightAtEntry = stackHeight;
    frame.op = op;
    pushControlFrame(std::move(frame));

    // Push parameters for then branch
    for (Type type : paramTypes) {
      pushValue(type);
    }

    if (failed(walkBody(op.getThenBody())))
      return failure();

    if (failed(popControlFrame()))
      return failure();
  }

  // Verify else branch if present
  if (!op.getElseBody().empty()) {
    // Reset stack for else branch
    valueStack.resize(stackHeight);
    unreachable = wasUnreachable;

    ControlFrame frame;
    frame.kind = ControlFrame::Else;
    auto label = op.getLabel();
    if (label.has_value())
      frame.label = label.value();
    frame.paramTypes = paramTypes;
    frame.resultTypes = resultTypes;
    frame.stackHeightAtEntry = stackHeight;
    frame.op = op;
    pushControlFrame(std::move(frame));

    // Push parameters for else branch
    for (Type type : paramTypes) {
      pushValue(type);
    }

    if (failed(walkBody(op.getElseBody())))
      return failure();

    if (failed(popControlFrame()))
      return failure();
  } else if (!resultTypes.empty()) {
    // If has no else but produces results, that's an error
    // (unless then branch is unreachable)
    if (!wasUnreachable) {
      return op.emitError(
          "if without else cannot produce results unless unreachable");
    }
  }

  return success();
}

LogicalResult StackVerifier::handleBrOp(BrOp op) {
  StringRef target = op.getTargetAttr().getValue();
  ControlFrame *frame = findLabelFrame(target);
  if (!frame) {
    return op.emitError("branch target '") << target << "' not found";
  }

  // Verify branch types are on stack
  if (failed(popValues(frame->getBranchTypes())))
    return failure();

  // After unconditional branch, we're in unreachable code
  unreachable = true;
  if (!controlStack.empty()) {
    controlStack.back().unreachable = true;
  }

  return success();
}

LogicalResult StackVerifier::handleBrIfOp(BrIfOp op) {
  // Pop condition (i32)
  if (failed(popValue(IntegerType::get(op.getContext(), 32))))
    return failure();

  StringRef target = op.getTargetAttr().getValue();
  ControlFrame *frame = findLabelFrame(target);
  if (!frame) {
    return op.emitError("branch target '") << target << "' not found";
  }

  // For conditional branch, we need the branch types on stack,
  // but they stay on stack if branch is not taken
  ArrayRef<Type> branchTypes = frame->getBranchTypes();

  // Check that we have enough values and they match
  size_t frameBase = controlStack.back().stackHeightAtEntry;
  if (valueStack.size() - frameBase < branchTypes.size()) {
    return emitError("insufficient values for conditional branch: need ")
           << branchTypes.size() << " but only have "
           << (valueStack.size() - frameBase);
  }

  // Type check the top values (but don't pop - they're needed for fallthrough)
  for (size_t i = 0; i < branchTypes.size(); ++i) {
    size_t stackIdx = valueStack.size() - branchTypes.size() + i;
    if (!isSubtype(valueStack[stackIdx], branchTypes[i])) {
      return emitError("conditional branch type mismatch at index ")
             << i << ": expected " << branchTypes[i] << " but got "
             << valueStack[stackIdx];
    }
  }

  // Stack remains the same (branch types stay for fallthrough)
  return success();
}

LogicalResult StackVerifier::handleBrTableOp(BrTableOp op) {
  // Pop index (i32)
  if (failed(popValue(IntegerType::get(op.getContext(), 32))))
    return failure();

  // Verify default target
  StringRef defaultTarget = op.getDefaultTargetAttr().getValue();
  ControlFrame *defaultFrame = findLabelFrame(defaultTarget);
  if (!defaultFrame) {
    return op.emitError("default branch target '")
           << defaultTarget << "' not found";
  }

  ArrayRef<Type> expectedTypes = defaultFrame->getBranchTypes();

  // Verify all targets have same arity and types
  for (auto targetAttr : op.getTargets()) {
    auto target = cast<FlatSymbolRefAttr>(targetAttr);
    ControlFrame *frame = findLabelFrame(target.getValue());
    if (!frame) {
      return op.emitError("branch target '")
             << target.getValue() << "' not found";
    }

    ArrayRef<Type> targetTypes = frame->getBranchTypes();
    if (targetTypes.size() != expectedTypes.size()) {
      return op.emitError("branch table targets have inconsistent arities");
    }
    for (size_t i = 0; i < expectedTypes.size(); ++i) {
      if (!isSubtype(targetTypes[i], expectedTypes[i]) ||
          !isSubtype(expectedTypes[i], targetTypes[i])) {
        return op.emitError("branch table targets have inconsistent types");
      }
    }
  }

  // Pop branch values
  if (failed(popValues(expectedTypes)))
    return failure();

  // After br_table, we're unreachable
  unreachable = true;
  if (!controlStack.empty()) {
    controlStack.back().unreachable = true;
  }

  return success();
}

LogicalResult StackVerifier::handleReturnOp(ReturnOp op) {
  // Find function frame (should be first frame)
  if (controlStack.empty()) {
    return op.emitError("return outside of function");
  }

  // Get function return types from the outermost (function) frame
  ControlFrame &funcFrame = controlStack.front();
  if (funcFrame.kind != ControlFrame::Function) {
    return op.emitError("return: couldn't find function frame");
  }

  // Pop return values
  if (failed(popValues(funcFrame.resultTypes)))
    return failure();

  // After return, we're unreachable
  unreachable = true;
  if (!controlStack.empty()) {
    controlStack.back().unreachable = true;
  }

  return success();
}

LogicalResult StackVerifier::handleUnreachableOp(UnreachableOp op) {
  // After unreachable, stack becomes polymorphic
  unreachable = true;
  if (!controlStack.empty()) {
    controlStack.back().unreachable = true;
  }
  return success();
}

LogicalResult StackVerifier::handleContNewOp(ContNewOp op) {
  FailureOr<FunctionType> contSig =
      resolveContSig(op, op.getContTypeAttr(), "cont.new");
  if (failed(contSig))
    return failure();

  Type actual;
  if (failed(popAnyValue(actual)))
    return failure();

  auto funcRefType = dyn_cast<FuncRefType>(actual);
  if (!funcRefType) {
    return op.emitError("cont.new expects funcref on stack, got ") << actual;
  }

  // If the referenced function symbol is known, ensure signature compatibility.
  Operation *funcSym =
      SymbolTable::lookupNearestSymbolFrom(op, funcRefType.getTypeName());
  if (auto func = dyn_cast_or_null<FuncOp>(funcSym)) {
    if (func.getFuncType() != *contSig) {
      return op.emitError("funcref signature does not match continuation type");
    }
  } else if (auto import = dyn_cast_or_null<FuncImportOp>(funcSym)) {
    if (import.getFuncType() != *contSig) {
      return op.emitError(
          "imported funcref signature does not match continuation type");
    }
  }

  pushValue(getNonNullContRefType(op.getContTypeAttr()));
  return success();
}

LogicalResult StackVerifier::handleContBindOp(ContBindOp op) {
  FailureOr<FunctionType> srcSig =
      resolveContSig(op, op.getSrcContTypeAttr(), "cont.bind source");
  FailureOr<FunctionType> dstSig =
      resolveContSig(op, op.getDstContTypeAttr(), "cont.bind destination");
  if (failed(srcSig) || failed(dstSig))
    return failure();

  if (srcSig->getNumInputs() < dstSig->getNumInputs()) {
    return op.emitError(
        "destination continuation input arity exceeds source arity");
  }

  if (srcSig->getResults() != dstSig->getResults()) {
    return op.emitError(
        "source and destination continuation result types must match");
  }

  unsigned boundCount = srcSig->getNumInputs() - dstSig->getNumInputs();
  SmallVector<Type> expected;
  expected.push_back(getNonNullContRefType(op.getSrcContTypeAttr()));
  expected.append(srcSig->getInputs().begin(),
                  srcSig->getInputs().begin() + boundCount);

  if (failed(popValues(expected)))
    return failure();

  pushValue(getNonNullContRefType(op.getDstContTypeAttr()));
  return success();
}

static LogicalResult verifySwitchTagHasEmptyInputs(Operation *op,
                                                   FunctionType tagSig,
                                                   StringRef diagnostic) {
  if (!tagSig.getInputs().empty())
    return op->emitError(diagnostic);
  return success();
}

LogicalResult StackVerifier::handleResumeOp(ResumeOp op) {
  FailureOr<FunctionType> contSig =
      resolveContSig(op, op.getContTypeAttr(), "resume");
  if (failed(contSig))
    return failure();

  Type contRefType = getNonNullContRefType(op.getContTypeAttr());
  SmallVector<Type> expected;
  expected.append(contSig->getInputs().begin(), contSig->getInputs().end());
  expected.push_back(contRefType);
  if (failed(popValues(expected)))
    return failure();

  // Validate handler clauses.
  for (Attribute attr : op.getHandlers()) {
    auto pair = dyn_cast<ArrayAttr>(attr);
    if (!pair || pair.size() != 2)
      return op.emitError("invalid handler format");
    auto tag = dyn_cast<FlatSymbolRefAttr>(pair[0]);
    auto label = dyn_cast<FlatSymbolRefAttr>(pair[1]);
    if (!tag || !label)
      return op.emitError("handler must be (tag -> label) pair");

    FailureOr<FunctionType> tagSig = resolveTagSig(op, tag, "resume handler");
    if (failed(tagSig))
      return failure();

    // "switch" is the wasmstack sentinel for on-switch handlers.
    if (label.getValue() == "switch") {
      if (failed(verifySwitchTagHasEmptyInputs(
              op, *tagSig, "switch handler tag must have empty inputs")))
        return failure();
      continue;
    }

    ControlFrame *target = findLabelFrame(label.getValue());
    if (!target)
      return op.emitError("unknown handler label ") << label;

    ArrayRef<Type> targetTypes = target->getBranchTypes();
    size_t expectedPayloadCount = tagSig->getInputs().size();
    if (targetTypes.size() != expectedPayloadCount + 1) {
      return op.emitError("handler label ")
             << label << " expects " << targetTypes.size()
             << " values but handler passes " << (expectedPayloadCount + 1);
    }

    for (size_t idx = 0; idx < expectedPayloadCount; ++idx) {
      Type targetType = targetTypes[idx];
      Type expectedType = tagSig->getInput(idx);
      if (!isSubtype(expectedType, targetType)) {
        return op.emitError("handler label type mismatch at index ")
               << idx << " for " << label << ": expected " << targetType
               << " but handler provides " << expectedType;
      }
    }

    Type targetContType = targetTypes.back();
    FlatSymbolRefAttr targetContRef;
    if (auto nonNull = dyn_cast<ContRefNonNullType>(targetContType)) {
      targetContRef = nonNull.getTypeName();
    } else if (auto nullable = dyn_cast<ContRefType>(targetContType)) {
      targetContRef = nullable.getTypeName();
    } else {
      return op.emitError("handler label ")
             << label << " must end with continuation reference type";
    }

    FailureOr<FunctionType> targetContSig =
        resolveContSig(op, targetContRef, "resume handler continuation");
    if (failed(targetContSig))
      return failure();

    if (targetContSig->getInputs() != tagSig->getResults() ||
        targetContSig->getResults() != contSig->getResults()) {
      return op.emitError("handler continuation type mismatch for ")
             << label << ": expected (" << tagSig->getResults() << ") -> ("
             << contSig->getResults() << ") but got " << *targetContSig;
    }
  }

  for (Type result : contSig->getResults())
    pushValue(result);
  return success();
}

LogicalResult StackVerifier::handleResumeThrowOp(ResumeThrowOp op) {
  FailureOr<FunctionType> contSig =
      resolveContSig(op, op.getContTypeAttr(), "resume_throw");
  if (failed(contSig))
    return failure();

  Type contRefType = getNonNullContRefType(op.getContTypeAttr());
  SmallVector<Type> expected;
  expected.append(contSig->getInputs().begin(), contSig->getInputs().end());
  expected.push_back(contRefType);
  if (failed(popValues(expected)))
    return failure();

  for (Attribute attr : op.getHandlers()) {
    auto pair = dyn_cast<ArrayAttr>(attr);
    if (!pair || pair.size() != 2)
      return op.emitError("invalid handler format");
    auto tag = dyn_cast<FlatSymbolRefAttr>(pair[0]);
    auto label = dyn_cast<FlatSymbolRefAttr>(pair[1]);
    if (!tag || !label)
      return op.emitError("handler must be (tag -> label) pair");

    FailureOr<FunctionType> tagSig =
        resolveTagSig(op, tag, "resume_throw handler");
    if (failed(tagSig))
      return failure();

    if (label.getValue() == "switch") {
      if (failed(verifySwitchTagHasEmptyInputs(
              op, *tagSig, "switch handler tag must have empty inputs")))
        return failure();
      continue;
    }

    ControlFrame *target = findLabelFrame(label.getValue());
    if (!target)
      return op.emitError("unknown handler label ") << label;

    ArrayRef<Type> targetTypes = target->getBranchTypes();
    size_t expectedPayloadCount = tagSig->getInputs().size();
    if (targetTypes.size() != expectedPayloadCount + 1) {
      return op.emitError("handler label ")
             << label << " expects " << targetTypes.size()
             << " values but handler passes " << (expectedPayloadCount + 1);
    }

    for (size_t idx = 0; idx < expectedPayloadCount; ++idx) {
      Type targetType = targetTypes[idx];
      Type expectedType = tagSig->getInput(idx);
      if (!isSubtype(expectedType, targetType)) {
        return op.emitError("handler label type mismatch at index ")
               << idx << " for " << label << ": expected " << targetType
               << " but handler provides " << expectedType;
      }
    }

    Type targetContType = targetTypes.back();
    FlatSymbolRefAttr targetContRef;
    if (auto nonNull = dyn_cast<ContRefNonNullType>(targetContType)) {
      targetContRef = nonNull.getTypeName();
    } else if (auto nullable = dyn_cast<ContRefType>(targetContType)) {
      targetContRef = nullable.getTypeName();
    } else {
      return op.emitError("handler label ")
             << label << " must end with continuation reference type";
    }

    FailureOr<FunctionType> targetContSig =
        resolveContSig(op, targetContRef, "resume_throw handler continuation");
    if (failed(targetContSig))
      return failure();

    if (targetContSig->getInputs() != tagSig->getResults() ||
        targetContSig->getResults() != contSig->getResults()) {
      return op.emitError("handler continuation type mismatch for ")
             << label << ": expected (" << tagSig->getResults() << ") -> ("
             << contSig->getResults() << ") but got " << *targetContSig;
    }
  }

  return success();
}

LogicalResult StackVerifier::handleSuspendOp(SuspendOp op) {
  FailureOr<FunctionType> tagSig =
      resolveTagSig(op, op.getTagAttr(), "suspend");
  if (failed(tagSig))
    return failure();

  if (failed(popValues(tagSig->getInputs())))
    return failure();

  for (Type result : tagSig->getResults())
    pushValue(result);
  return success();
}

LogicalResult StackVerifier::handleSwitchOp(SwitchOp op) {
  FailureOr<FunctionType> contSig =
      resolveContSig(op, op.getContTypeAttr(), "switch");
  FailureOr<FunctionType> tagSig = resolveTagSig(op, op.getTagAttr(), "switch");
  if (failed(contSig) || failed(tagSig))
    return failure();
  if (failed(verifySwitchTagHasEmptyInputs(
          op, *tagSig, "switch tag must have empty inputs")))
    return failure();

  SmallVector<Type> expected;
  expected.append(contSig->getInputs().begin(), contSig->getInputs().end());
  expected.append(tagSig->getInputs().begin(), tagSig->getInputs().end());
  expected.push_back(getNonNullContRefType(op.getContTypeAttr()));
  if (failed(popValues(expected)))
    return failure();

  return success();
}

LogicalResult StackVerifier::walkBody(Region &region) {
  if (region.empty())
    return success();

  // WasmStack uses single-block regions with NoTerminator
  Block &block = region.front();
  for (Operation &op : block) {
    if (failed(verifyStackOp(&op)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Function Verification
//===----------------------------------------------------------------------===//

LogicalResult StackVerifier::verifyFunction(FuncOp funcOp) {
  reset(funcOp.getContext());

  // Get function type
  FunctionType funcType = funcOp.getFuncType();

  // Push function frame
  ControlFrame frame;
  frame.kind = ControlFrame::Function;
  frame.paramTypes = SmallVector<Type>(funcType.getInputs());
  frame.resultTypes = SmallVector<Type>(funcType.getResults());
  frame.stackHeightAtEntry = 0;
  frame.op = funcOp;
  pushControlFrame(std::move(frame));

  // Function parameters are NOT on the stack - they're in locals
  // (In WasmStack, parameters are accessed via local.get, not on the stack)

  // Verify function body
  if (failed(walkBody(funcOp.getBody())))
    return failure();

  // Pop function frame
  return popControlFrame();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct VerifyWasmStackPass
    : public impl::VerifyWasmStackBase<VerifyWasmStackPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    StackVerifier verifier;

    // Verify each function in the module (top-level WasmStack functions)
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      if (failed(verifier.verifyFunction(funcOp))) {
        signalPassFailure();
        return;
      }
    }

    // Also check WasmStack modules (wasmstack.module containers)
    for (auto wasmModule : moduleOp.getOps<wasmstack::ModuleOp>()) {
      for (auto funcOp : wasmModule.getOps<FuncOp>()) {
        if (failed(verifier.verifyFunction(funcOp))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::wasmstack
