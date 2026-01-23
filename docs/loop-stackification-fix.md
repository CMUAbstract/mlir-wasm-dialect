# Fix Plan: Incomplete Loop Bodies in Stackification Pass

## Problem Summary

The stackification pass only processes the **first block** (`.front()`) of multi-block regions. Loop regions have multiple blocks:
- `^bb0`: Exit condition check + `branch_if`
- `^bb1`: Loop body + `block_return` (continue)

The fix must **linearize the CFG** during emission by following terminator successors.

---

## WasmSSA Structure (Input)

```mlir
wasmssa.loop(%arg0, %arg1) : i32, i32 : {
^bb0(%arg2: i32, %arg3: i32):
  %6 = wasmssa.ge_si %arg2 %1 : i32 -> i32
  wasmssa.branch_if %6 to level 1 with args(%arg3 : i32) else ^bb1
^bb1:
  %7 = wasmssa.add %arg3 %4 : i32
  %8 = wasmssa.add %arg2 %2 : i32
  wasmssa.block_return %8, %7 : i32, i32
}
```

## Expected WasmStack Output

```mlir
wasmstack.loop @loop_0 : ([i32, i32]) -> [] {
  // From ^bb0:
  wasmstack.ge_s : i32
  wasmstack.br_if @block_0
  // From ^bb1 (fallthrough after br_if):
  wasmstack.add : i32         // acc + 1
  wasmstack.add : i32         // counter + step
  wasmstack.br @loop_0        // continue loop
}
```

---

## Implementation Plan

### Step 1: Modify `emitLoop` to Process CFG (lines 700-759)

Replace the simple `.front()` iteration with CFG traversal:

```cpp
void emitLoop(wasmssa::LoopOp loopOp) {
  // ... existing setup code (lines 700-740) ...

  // NEW: CFG linearization
  if (!loopOp.getBody().empty()) {
    Block *currentBlock = &loopOp.getBody().front();
    llvm::DenseSet<Block *> processed;

    while (currentBlock && !processed.contains(currentBlock)) {
      processed.insert(currentBlock);

      // Mark block arguments as available on stack
      for (BlockArgument arg : currentBlock->getArguments()) {
        emittedToStack.insert(arg);
      }

      // Emit all operations EXCEPT terminator
      for (Operation &op : currentBlock->without_terminator()) {
        emitOperation(&op);
      }

      // Handle terminator and get next block
      currentBlock = emitTerminatorAndGetNext(currentBlock->getTerminator(),
                                               /*isInLoop=*/true);
    }
  }

  labelStack.pop_back();
}
```

### Step 2: Add `emitTerminatorAndGetNext` Helper Function

This function handles terminators and returns the next block to process (or nullptr to stop):

```cpp
Block *emitTerminatorAndGetNext(Operation *terminator, bool isInLoop) {
  Location loc = terminator->getLoc();

  if (auto branchIfOp = dyn_cast<wasmssa::BranchIfOp>(terminator)) {
    // branch_if %cond to level N with args(...) else ^successor

    // 1. Emit exit args to stack (they stay if branch not taken)
    for (Value arg : branchIfOp.getExitArgs()) {
      emitOperandIfNeeded(arg);
    }

    // 2. Emit condition
    emitOperandIfNeeded(branchIfOp.getCondition());

    // 3. Emit br_if
    unsigned exitLevel = branchIfOp.getExitLevel();
    std::string label = getLabelForExitLevel(exitLevel);
    BrIfOp::create(builder, loc, builder.getAttr<FlatSymbolRefAttr>(label));

    // 4. Return the else successor to continue processing
    return branchIfOp.getSuccessor(0);  // The else/fallthrough block
  }

  if (auto blockReturnOp = dyn_cast<wasmssa::BlockReturnOp>(terminator)) {
    // Emit return values to stack
    for (Value input : blockReturnOp.getInputs()) {
      emitOperandIfNeeded(input);
    }

    // If inside a loop, emit br to continue the loop
    if (isInLoop) {
      std::string loopLabel = labelStack.back().first;
      BrOp::create(builder, loc, builder.getAttr<FlatSymbolRefAttr>(loopLabel));
    }
    // Else: values on stack, control flows to block end

    return nullptr;  // Stop processing
  }

  // Handle other terminators (branch, return) as needed
  if (auto branchOp = dyn_cast<wasmssa::BranchOp>(terminator)) {
    if (branchOp.targetsLevel()) {
      // Unconditional exit
      unsigned exitLevel = branchOp.getExitLevel();
      std::string label = getLabelForExitLevel(exitLevel);
      BrOp::create(builder, loc, builder.getAttr<FlatSymbolRefAttr>(label));
      return nullptr;
    } else {
      // Unconditional branch to another block (fallthrough)
      return branchOp.getSuccessor(0);
    }
  }

  return nullptr;
}
```

### Step 3: Apply Same Pattern to `emitBlock` (lines 660-697)

The `emitBlock` function has the same issue and needs the same CFG traversal pattern.

### Step 4: Update `emitBranchIf` (lines 838-849)

The current `emitBranchIf` is only called when processing single blocks. With CFG traversal, it may no longer be needed as a separate function - the logic moves into `emitTerminatorAndGetNext`. However, keep it for backward compatibility with single-block regions.

---

## Edge Cases to Handle

### 1. Values Used Across Blocks

**Problem:** Values defined in `^bb0` (like `%arg2`, `%arg3`) are used in `^bb1`.

**Solution:** `emitOperandIfNeeded` already handles this:
- If value is in `emittedToStack`, it's available
- If value needs a local, emit `local.get`

The key is that block arguments are marked in `emittedToStack` at the start of processing each block.

### 2. Stack State After `br_if`

**WebAssembly semantics:**
- `br_if` with N result values: if true, pops N values and branches; if false, **does NOT pop**, falls through
- So after fallthrough, the exit args are still on the stack

**Solution:** The exit args emitted before `br_if` remain available for the fallthrough block. The `emittedToStack` tracking handles subsequent uses.

### 3. Nested Control Flow

**Example:** `if` inside a `loop`

**Solution:** The existing recursive `emitIf` handles this. When we call `emitOperation` on an `IfOp`, it recursively processes the if's regions. The CFG traversal only handles the outer loop structure.

### 4. Multiple Successors from `branch_if`

**Current pattern:** `branch_if %cond to level N else ^bb1`
- Only ONE successor block (`^bb1`)
- The "level N" is not a block, it's a structured exit

**Solution:** The pattern always has exactly one fallthrough successor. Process it linearly.

### 5. Loop Continuation Branch

**Problem:** `block_return` inside a loop means "continue with new values"

**Solution:** Emit `br @loop_label` after pushing return values:
```cpp
if (isInLoop) {
  BrOp::create(builder, loc, loopLabel);
}
```

---

## Files to Modify

| File | Location | Change |
|------|----------|--------|
| `lib/wasmstack/WasmStackPasses.cpp` | ~700-759 | Rewrite `emitLoop` with CFG traversal |
| `lib/wasmstack/WasmStackPasses.cpp` | ~660-697 | Apply same pattern to `emitBlock` |
| `lib/wasmstack/WasmStackPasses.cpp` | New function | Add `emitTerminatorAndGetNext` helper |
| `lib/wasmstack/WasmStackPasses.cpp` | ~490-498 | Update `BlockReturnOp` handling (or remove if using helper) |

---

## Verification

```bash
# Run the full pipeline
./build/bin/wasm-opt --wami-convert-scf --wami-convert-arith --wami-convert-func \
  --reconcile-unrealized-casts --convert-to-wasmstack \
  test/wasmstack/control-flow-conversion.mlir

# Expected output for for_loop_sum:
wasmstack.func @for_loop_sum : () -> i32 {
  wasmstack.i32.const 5
  wasmstack.i32.const 1
  wasmstack.i32.const 1
  wasmstack.i32.const 0
  wasmstack.i32.const 0
  wasmstack.block @block_0 : ([i32, i32]) -> [i32] {
    wasmstack.loop @loop_1 : ([i32, i32]) -> [] {
      wasmstack.ge_s : i32
      wasmstack.br_if @block_0
      wasmstack.add : i32     // NEW: acc + 1
      wasmstack.add : i32     // NEW: counter + step
      wasmstack.br @loop_1    // NEW: continue loop
    }
  }
}

# Run existing tests
cmake --build build --target check-wasm
```
