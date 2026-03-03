# Plan: `hoist-from-scf-if` Pass

## Context

After `wami-convert-memref` lowers memref operations, it generates address arithmetic (`arith.constant`, `arith.muli`, `arith.addi`) inline inside whatever `scf.if` region the original memref op lived in. MLIR's built-in CSE and LICM cannot eliminate these redundancies because each `scf.if` region is a separate SSA scope. This causes 8 extra `i32.mul` per inner-loop iteration in benchmarks like nussinov, accounting for a significant performance gap vs LLVM.

The fix: a new pass that hoists pure operations with externally-defined operands out of `scf.if` regions, exposing them to subsequent CSE and LICM.

## Files to modify

1. **`include/Transforms/TransformsPasses.td`** — Add TableGen pass definition
2. **`lib/Transforms/HoistFromScfIf.cpp`** — New file, pass implementation
3. **`lib/Transforms/CMakeLists.txt`** — Add source file to build
4. **`toolchain/compile.sh`** — Insert pass in WAMI pipeline
5. **`test/Transforms/hoist-from-scf-if.mlir`** — New file, lit tests

## Step 1: TableGen definition

Add to `include/Transforms/TransformsPasses.td` after existing passes:

```tablegen
def HoistFromScfIf : Pass<"hoist-from-scf-if", "ModuleOp"> {
  let summary = "Hoist pure operations out of scf.if regions";
  let description = [{
    Moves pure (side-effect-free) operations whose operands are all defined
    outside the scf.if region to before the scf.if in the parent block.
    Processes bottom-up (innermost scf.if first) to handle nesting.
    Exposes redundant computations to subsequent CSE and LICM.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}
```

## Step 2: Implementation (`lib/Transforms/HoistFromScfIf.cpp`)

Follow `StrengthReduce.cpp` conventions (namespace `mlir::transforms`, `GEN_PASS_DEF_HOISTFROMSCFIF` macro).

### Core algorithm (single forward pass per region)

```
runOnOperation():
  Collect all scf::IfOp via module.walk()
  Process in reverse (bottom-up, innermost first)
  For each IfOp: hoistFromRegion(thenRegion), hoistFromRegion(elseRegion)

hoistFromRegion(region, insertBeforeOp):
  Snapshot ops list from region's block
  For each op in forward order:
    Skip if terminator (scf.yield)
    Skip if not isMemoryEffectFree(op)
    Skip if any operand's parent region is inside this region
    Otherwise: op->moveBefore(insertBeforeOp)
```

**Why single forward pass works**: Operations are processed in block order. When `arith.constant` (no operands) is hoisted first, it leaves the region. When we then check `arith.muli` whose operand was that constant, `operand.getParentRegion()` now returns the parent region (not the scf.if region), so the muli becomes hoistable too. The chain unravels naturally in one forward scan.

**Correctness**: Pure ops with all-external operands can always be safely moved before the `scf.if`. The hoisted value still dominates all uses inside the region (since it now dominates the entire `scf.if`). No terminator or side-effecting op is ever moved.

## Step 3: CMakeLists.txt

Add `HoistFromScfIf.cpp` to `lib/Transforms/CMakeLists.txt` source list. No new link dependencies needed (already links `MLIRSCFDialect`).

## Step 4: Pipeline insertion in `compile.sh`

Insert `hoist-from-scf-if` after `wami-convert-memref, canonicalize` and before the existing `sccp, cse, loop-invariant-code-motion` block:

```
wami-convert-memref, \
canonicalize, \
hoist-from-scf-if, \       ← NEW
sccp, \
cse, \
loop-invariant-code-motion, \
...
```

This way: (1) hoist-from-scf-if moves pure ops out of scf.if, (2) CSE merges duplicates now in the same scope, (3) LICM hoists loop-invariant ones to preheader.

## Step 5: Tests

Create `test/Transforms/hoist-from-scf-if.mlir` with:

1. **Basic**: constant inside scf.if → hoisted before scf.if
2. **Chain**: constant + muli + addi chain inside scf.if → all hoisted
3. **Both regions**: same constant in then and else → both hoisted (CSE can merge later)
4. **Nested scf.if**: inner if inside outer if → ops reach outer scope
5. **Negative: side effects**: `wami.load` stays inside scf.if
6. **Negative: internal operand**: op depending on block argument stays
7. **scf.if with results**: hoisted op used by scf.yield still works

## Verification

1. `cmake --build build --target check-wasm` — all existing tests pass
2. Run the nussinov comparison:
   ```bash
   ./toolchain/compile.sh -i benchmark/polybench/medium/nussinov.mlir \
     -o /tmp/wami-nussinov --compiler=wami --binaryen-opt-flags="-O4" --skip-build
   wasm2wat /tmp/wami-nussinov.wasm -o /tmp/wami-nussinov-final.wat
   ```
   Count `i32.mul` in hot region — should drop from 9 to ~3 (matching LLVM)
3. Dump intermediate IR after the new pass to verify constants/muls are hoisted out of scf.if regions
