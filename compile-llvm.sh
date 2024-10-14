#!/bin/bash

WASI_SDK_PATH=/Users/byeongje/wasm/wasi-sdk-22.0

# Default values for input and output
OPTIMIZE=false

# Function to display usage information
usage() {
    echo "Usage: $0 -i <input_mlir_file> -o <output_base_name> [--optimize]"
    echo "  -i, --input      Input MLIR file"
    echo "  -o, --output     Output base name"
    echo "  --optimize       Perform WebAssembly optimization (optional)"
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
        --optimize)
            OPTIMIZE=true
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

# File paths based on output base name
OUTPUT_LLVM_MLIR="${OUTPUT_BASE}-llvm.mlir"
OUTPUT_LL="${OUTPUT_BASE}.ll"
OUTPUT_OBJ="${OUTPUT_BASE}.o"
OUTPUT_WAT="${OUTPUT_BASE}.wat"
OUTPUT_LINKED_WASM="${OUTPUT_BASE}-linked.wasm"
OUTPUT_LINKED_WAT="${OUTPUT_BASE}-linked.wat"

# Step 1: Convert the MLIR file to LLVM dialect
echo "Converting $INPUT_MLIR to LLVM dialect..."
mlir-opt "$INPUT_MLIR" --convert-scf-to-cf --lower-affine \
    --convert-arith-to-llvm="index-bitwidth=32" \
    --convert-func-to-llvm="index-bitwidth=32" \
    --memref-expand --expand-strided-metadata \
    --finalize-memref-to-llvm="index-bitwidth=32" \
    --convert-to-llvm --reconcile-unrealized-casts -o "$OUTPUT_LLVM_MLIR"

# Step 2: Translate the LLVM dialect MLIR file to LLVM IR (.ll)
echo "Translating $OUTPUT_LLVM_MLIR to LLVM IR (.ll)..."
mlir-translate "$OUTPUT_LLVM_MLIR" --mlir-to-llvmir -o "$OUTPUT_LL"

# Step 3: Use `llc` to lower the LLVM IR (.ll) to an object file
echo "Lowering $OUTPUT_LL to object file (.o)..."
if $OPTIMIZE; then
    llc -O3 -filetype=obj -mtriple=wasm32-wasi "$OUTPUT_LL" -o "$OUTPUT_OBJ"
else
    llc -O0 -filetype=obj -mtriple=wasm32-wasi "$OUTPUT_LL" -o "$OUTPUT_OBJ"
fi

# Step 4: Convert the object file to WAT format using `wasm2wat`
echo "Converting $OUTPUT_OBJ to WAT format..."
wasm2wat "$OUTPUT_OBJ" -o "$OUTPUT_WAT"


# Step 5: Link the Wasm object file with the standard library using `wasm-ld`
echo "Linking the object file with stdlib using wasm-ld..."
if $OPTIMIZE; then
    $WASI_SDK_PATH/bin/wasm-ld --no-entry \
    --export-memory --export=_mlir_ciface_main --export=malloc --export=free \
    -L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
    -O3 --lto-CGO3 --lto-O3 -o "$OUTPUT_LINKED_WASM" "$OUTPUT_OBJ"
else
    $WASI_SDK_PATH/bin/wasm-ld --no-entry \
    --export-memory --export=_mlir_ciface_main --export=malloc --export=free \
    --global-base=0 -L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
    -o "$OUTPUT_LINKED_WASM" "$OUTPUT_OBJ"
fi

if $OPTIMIZE; then
    echo "Optimizing the WebAssembly output..."
    wasm-opt "$OUTPUT_LINKED_WASM" -O4 -o "$OUTPUT_LINKED_WASM"
else
    echo "Skipping WebAssembly optimization..."
fi



# Step 6: Convert the linked Wasm file to WAT format
echo "Converting the linked Wasm file to formatted WAT format..."
wasm2wat "$OUTPUT_LINKED_WASM" -o "$OUTPUT_LINKED_WAT"

echo "Compilation completed! The output files are:"
echo "  LLVM MLIR: $OUTPUT_LLVM_MLIR"
echo "  LLVM IR: $OUTPUT_LL"
echo "  Object file: $OUTPUT_OBJ"
echo "  WAT: $OUTPUT_WAT"
echo "  Linked WASM: $OUTPUT_LINKED_WASM"
echo "  Linked WAT: $OUTPUT_LINKED_WAT"
