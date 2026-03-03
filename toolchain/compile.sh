#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values for input, output, and flags
ADD_DEBUG_FUNCTIONS=false
BINARYEN_OPT_FLAGS=""
COMPILER=""
CLEAN=false
LLVM_OPT_FLAGS=""
SKIP_BUILD=false

# Function to display usage information
usage() {
    echo "Usage: $0 -i <input_mlir_file> -o <output_base_name> [--binaryen-opt-flags]"
    echo "  -i, --input      Input MLIR file"
    echo "  -o, --output     Output base name"
    echo "  --compiler       Compiler to use (wami or llvm)"
    echo "  --binaryen-opt-flags    Perform WebAssembly optimization (optional)"
    echo "  --llvm-opt-flags        Perform LLVM optimization (optional, only supported in --compiler=llvm)"
    echo "  --add-debug-functions   Add debug functions to the output (optional, only supported in --compiler=wami)"
    echo "  --clean                 Remove temporary files after completion"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_MLIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --compiler=*)
            COMPILER="${1#*=}"
            shift
            ;;
        --binaryen-opt-flags=*)
            BINARYEN_OPT_FLAGS="${1#*=}"
            shift
            ;;
        --llvm-opt-flags=*)
            LLVM_OPT_FLAGS="${1#*=}"
            shift
            ;;
        --add-debug-functions)
            ADD_DEBUG_FUNCTIONS=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

# Check if input and output are provided
if [[ -z "$INPUT_MLIR" || -z "$OUTPUT_BASE" ]]; then
    echo "Error: input and output are required."
    usage
fi

if [[ -z "$COMPILER" ]]; then
    echo "Error: compiler option is required. Use --compiler to specify 'wami' or 'llvm'."
    usage
elif [[ "$COMPILER" != "wami" && "$COMPILER" != "llvm" ]]; then
    echo "Error: Invalid compiler option '$COMPILER'. It must be either 'wami' or 'llvm'."
    usage
fi

# Check if WASI_SDK_PATH environment variable is set (needed for linking libc)
if [[ -z "$WASI_SDK_PATH" ]]; then
    echo "Error: Environment variable 'WASI_SDK_PATH' is not defined."
    echo "Please set it before running."
    exit 1
fi

# ============================================================
# Common MLIR pass pipeline fragments (shared by both compilers)
# ============================================================

# Affine-level optimizations (on affine.for / affine.load / affine.store).
# Accumulator promotion must precede unrolling for best pattern detection.
AFFINE_OPTS="\
affine-scalrep, \
affine-promote-accumulators, \
affine-loop-invariant-code-motion, \
affine-loop-normalize"

# Post-affine optimizations (on scf.for / memref.load / memref.store).
POST_AFFINE_OPTS="\
symbol-dce, \
canonicalize, \
cse, \
scf-loop-unroll{unroll-factor=4}"

# Standard optimization cleanup sequence (used after each lowering phase).
OPT_CLEANUP="\
sccp, \
loop-invariant-code-motion, \
loop-invariant-subset-hoisting, \
cse, \
control-flow-sink"

# Full shared preprocessing: standard MLIR → optimized SCF/memref MLIR.
# Both compilers apply this pipeline via wasm-opt (which has the custom
# accumulator promotion passes registered).
SHARED_PREPROCESS="\
inline, \
canonicalize, \
func.func($AFFINE_OPTS), \
lower-affine, \
$POST_AFFINE_OPTS, \
$OPT_CLEANUP"

# ============================================================
# Build
# ============================================================

# Final output file
OUTPUT_WASM="${OUTPUT_BASE}.wasm"
OUTPUT_BEFOREOPT_WASM=""

if [ "$SKIP_BUILD" = true ]; then
    echo "Skipping project build (--skip-build)."
else
    echo "Building the project..."
    cmake --build "$REPO_ROOT/build"
    echo "Building the project... done"
fi

# ============================================================
# Find wasm-ld binary (shared by both compilers)
# ============================================================

find_wasm_ld() {
    WASM_LD_BIN="${WASM_LD_BIN:-}"
    if [[ -z "$WASM_LD_BIN" ]]; then
        if command -v wasm-ld > /dev/null 2>&1; then
            WASM_LD_BIN="$(command -v wasm-ld)"
        else
            WASM_LD_BIN="$WASI_SDK_PATH/bin/wasm-ld"
        fi
    fi
    echo "Using wasm-ld binary: $WASM_LD_BIN"
}

# ============================================================
# Compiler-specific pipelines
# ============================================================

if [[ "$COMPILER" == "wami" ]]; then
    OUTPUT_WASMSTACK_MLIR="${OUTPUT_BASE}-wasmstack-1.mlir"
    OUTPUT_OBJ="${OUTPUT_BASE}-2.o" # relocatable object file
    OUTPUT_WAT="${OUTPUT_BASE}-obj-3.wat"
    OUTPUT_BEFOREOPT_WASM="${OUTPUT_BASE}-nobinaryen-4.wasm"
    OUTPUT_BEFOREOPT_WAT="${OUTPUT_BASE}-nobinaryen-5.wat"

    if $ADD_DEBUG_FUNCTIONS; then
        echo "Warning: --add-debug-functions is not supported in the new wami->wasmstack pipeline. Ignoring."
    fi

    # Full WAMI pipeline: shared preprocessing + WAMI-specific lowering.
    # Runs entirely in wasm-opt (single invocation).
    #   Phase 1-2: Shared preprocessing (affine opts, accumulator promotion, unrolling)
    #   Phase 3:   Lower memref (address math emitted as arith ops)
    #   Phase 4:   Optimize address computations + strength reduction
    #   Phase 5:   Lower remaining dialects (scf, arith, math, func) to WasmStack
    echo "Converting $INPUT_MLIR to WasmStack..."
    "$REPO_ROOT/build/bin/wasm-opt" \
    --pass-pipeline="builtin.module( \
      $SHARED_PREPROCESS, \
      wami-convert-memref, \
      canonicalize, \
      hoist-from-scf-if, \
      $OPT_CLEANUP, \
      remainder-to-and, \
      strength-reduce, \
      canonicalize, \
      cse, \
      wami-convert-scf, \
      wami-convert-arith, \
      wami-convert-math, \
      wami-convert-func, \
      reconcile-unrealized-casts, \
      convert-to-wasmstack, \
      verify-wasmstack \
    )" \
        "$INPUT_MLIR" \
        -o "${OUTPUT_WASMSTACK_MLIR}"

    echo "Emitting relocatable object from $OUTPUT_WASMSTACK_MLIR..."
    "$REPO_ROOT/build/bin/wasm-emit" "$OUTPUT_WASMSTACK_MLIR" --mlir-to-wasm --relocatable -o "$OUTPUT_OBJ"

    echo "Converting $OUTPUT_OBJ to WAT format..."
    wasm2wat "$OUTPUT_OBJ" -o "$OUTPUT_WAT"

    echo "Linking the object file with stdlib using wasm-ld..."
    find_wasm_ld
    "$WASM_LD_BIN" --no-entry --allow-undefined \
    --export-memory --export=main --export=malloc --export=free \
    --export=__heap_end \
    -L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
    -O3 --lto-CGO3 --lto-O3 -o "$OUTPUT_BEFOREOPT_WASM" "$OUTPUT_OBJ"

    wasm2wat "$OUTPUT_BEFOREOPT_WASM" -o "$OUTPUT_BEFOREOPT_WAT"

elif [[ "$COMPILER" == "llvm" ]]; then
    OUTPUT_PREPROCESSED="${OUTPUT_BASE}-preprocessed-0.mlir" # after shared opts
    OUTPUT_LLVM_MLIR="${OUTPUT_BASE}-llvm-1.mlir" # MLIR LLVM IR dialect
    OUTPUT_LL="${OUTPUT_BASE}-2.ll" # LLVM IR
    OUTPUT_OBJ="${OUTPUT_BASE}-3.o" # object file
    OUTPUT_WAT="${OUTPUT_BASE}-obj-4.wat"
    OUTPUT_BEFOREOPT_WASM="${OUTPUT_BASE}-nobinaryen-5.wasm"
    OUTPUT_BEFOREOPT_WAT="${OUTPUT_BASE}-nobinaryen-6.wat"

    # Step 1: Shared preprocessing via wasm-opt.
    # Uses wasm-opt (not mlir-opt) because the custom accumulator promotion
    # passes are registered there.
    echo "Preprocessing $INPUT_MLIR (shared optimizations)..."
    "$REPO_ROOT/build/bin/wasm-opt" \
    --pass-pipeline="builtin.module($SHARED_PREPROCESS)" \
        "$INPUT_MLIR" \
        -o "$OUTPUT_PREPROCESSED"

    # Step 2: LLVM-specific lowering via mlir-opt.
    # Input is already optimized SCF/memref MLIR from step 1.
    echo "Converting $OUTPUT_PREPROCESSED to LLVM dialect..."
    mlir-opt "$OUTPUT_PREPROCESSED" \
    --pass-pipeline="builtin.module( \
      convert-scf-to-cf, \
      convert-arith-to-llvm{index-bitwidth=32}, \
      convert-func-to-llvm{index-bitwidth=32}, \
      memref-expand, \
      expand-strided-metadata, \
      finalize-memref-to-llvm{index-bitwidth=32}, \
      convert-cf-to-llvm{index-bitwidth=32}, \
      canonicalize, \
      $OPT_CLEANUP, \
      convert-to-llvm, \
      reconcile-unrealized-casts, \
      canonicalize, \
      $OPT_CLEANUP \
    )" \
    -o "$OUTPUT_LLVM_MLIR"

    echo "Translating $OUTPUT_LLVM_MLIR to LLVM IR (.ll)..."
    mlir-translate "$OUTPUT_LLVM_MLIR" --mlir-to-llvmir -o "$OUTPUT_LL"

    OUTPUT_OPT_LL="${OUTPUT_BASE}-opt-2b.ll"
    echo "Running LLVM middle-end optimizations on $OUTPUT_LL..."
    opt -O3 -S "$OUTPUT_LL" -o "$OUTPUT_OPT_LL"

    echo "Lowering $OUTPUT_OPT_LL to object file (.o)..."
    llc $LLVM_OPT_FLAGS -filetype=obj -mtriple=wasm32-wasi "$OUTPUT_OPT_LL" -o "$OUTPUT_OBJ"

    echo "Converting $OUTPUT_OBJ to WAT format..."
    wasm2wat "$OUTPUT_OBJ" -o "$OUTPUT_WAT"

    echo "Linking the object file with stdlib using wasm-ld..."
    find_wasm_ld
    # We always use -O3 optimization level
    "$WASM_LD_BIN" --no-entry --allow-undefined \
    --export-memory --export=main --export=malloc --export=free \
    --export=__heap_end --export=__data_base \
    -L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
    -O3 --lto-CGO3 --lto-O3 -o "$OUTPUT_BEFOREOPT_WASM" "$OUTPUT_OBJ"
fi

if [[ -n "$BINARYEN_OPT_FLAGS" ]]; then
    echo "Optimizing the WebAssembly output..."
    wasm-opt "$OUTPUT_BEFOREOPT_WASM" $BINARYEN_OPT_FLAGS -o "$OUTPUT_WASM"
else
    echo "Skipping WebAssembly optimization..."
    cp "$OUTPUT_BEFOREOPT_WASM" "$OUTPUT_WASM"
fi

# Clean up temporary files if --clean flag is set
if $CLEAN; then
    echo "Cleaning up temporary files..."
    rm -f "$OUTPUT_WASMSTACK_MLIR" "$OUTPUT_BEFOREOPT_WASM" "$OUTPUT_BEFOREOPT_WAT" \
          "$OUTPUT_LLVM_MLIR" "$OUTPUT_LL" "$OUTPUT_OPT_LL" "$OUTPUT_OBJ" "$OUTPUT_WAT" \
          "$OUTPUT_PREPROCESSED"
fi

# Print the produced files
echo "Produced files:"
if [[ "$COMPILER" == "wami" ]]; then
    echo "  - $OUTPUT_WASMSTACK_MLIR (WasmStack MLIR)"
    echo "  - $OUTPUT_OBJ (Relocatable object file)"
    echo "  - $OUTPUT_WAT (WAT format)"
elif [[ "$COMPILER" == "llvm" ]]; then
    echo "  - $OUTPUT_PREPROCESSED (Preprocessed MLIR after shared optimizations)"
    echo "  - $OUTPUT_LLVM_MLIR (LLVM dialect MLIR)"
    echo "  - $OUTPUT_LL (LLVM IR)"
    echo "  - $OUTPUT_OPT_LL (LLVM IR after opt -O3)"
    echo "  - $OUTPUT_OBJ (Object file)"
    echo "  - $OUTPUT_WAT (WAT format)"
fi

# Final optimized (or unoptimized) WebAssembly file
echo "  - $OUTPUT_WASM (Final WebAssembly output)"

echo "Conversion completed!"
