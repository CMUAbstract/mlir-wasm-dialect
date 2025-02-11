#!/bin/bash

# Check if WASI_SDK_PATH environment variable is set
if [ -z "$WASI_SDK_PATH" ]; then
    echo "Error: Environment variable 'WASI_SDK_PATH' is not defined. Please set it before running this script."
    exit 1
fi

# Default values for input, output, and flags
ADD_DEBUG_FUNCTIONS=false
BINARYEN_OPT_FLAGS=""
COMPILER=""
CLEAN=false

# Function to display usage information
usage() {
    echo "Usage: $0 -i <input_mlir_file> -o <output_base_name> [--binaryen-opt-flags]"
    echo "  -i, --input      Input MLIR file"
    echo "  -o, --output     Output base name"
    echo "  --compiler       Compiler to use (mlir or llvm)"
    echo "  --binaryen-opt-flags    Perform WebAssembly optimization (optional)"
    echo "  --llvm-opt-flags        Perform LLVM optimization (optional, only supported in --compiler=llvm)"
    echo "  --add-debug-functions   Add debug functions to the output (optional, only supported in --compiler=mlir)"
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
    echo "Error: compiler option is required. Use --compiler to specify 'mlir' or 'llvm'."
    usage
elif [[ "$COMPILER" != "mlir" && "$COMPILER" != "llvm" ]]; then
    echo "Error: Invalid compiler option '$COMPILER'. It must be either 'mlir' or 'llvm'."
    usage
fi

# Final output file 
OUTPUT_WASM="${OUTPUT_BASE}.wasm" 

OUTPUT_BEFOREOPT_WASM=""

# Build the project
echo "Building the project..."
cmake --build build 
echo "Building the project... done"

if [[ "$COMPILER" == "mlir" ]]; then
    OUTPUT_MLIR="${OUTPUT_BASE}-wasm-1.mlir" # MLIR wasm dialect
    OUTPUT_RAW_WAT="${OUTPUT_BASE}-raw-2.wat" # 1 followed by by wasm-translate
    OUTPUT_BEFOREOPT_WASM="${OUTPUT_BASE}-nobinaryen-3.wasm" # 2 followed by wat2wasm
    OUTPUT_BEFOREOPT_WAT="${OUTPUT_BASE}-nobinaryen-4.wat" # 3 followed by wasm2wat

    # Convert MLIR file to the Wasm dialect
    echo "Converting $INPUT_MLIR to Wasm dialect..."
    build/bin/wasm-opt --convert-to-ssawasm --reconcile-unrealized-casts --ssawasm-data-to-local --introduce-locals --convert-ssawasm-to-wasm "$INPUT_MLIR" -o "${OUTPUT_MLIR}"

    # Translate the resulting MLIR file to a .wat file
    echo "Translating $OUTPUT_MLIR to .wat format..."
    if $ADD_DEBUG_FUNCTIONS; then
        build/bin/wasm-translate "$OUTPUT_MLIR" --mlir-to-wat --add-debug-functions -o "${OUTPUT_RAW_WAT}"
    else
        build/bin/wasm-translate "$OUTPUT_MLIR" --mlir-to-wat -o "${OUTPUT_RAW_WAT}"
    fi

    echo "Converting $OUTPUT_RAW_WAT to .wasm format..."
    wat2wasm "$OUTPUT_RAW_WAT" -o "$OUTPUT_BEFOREOPT_WASM"

    wasm2wat "$OUTPUT_BEFOREOPT_WASM" -o "$OUTPUT_BEFOREOPT_WAT"

elif [[ "$COMPILER" == "llvm" ]]; then
    OUTPUT_LLVM_MLIR="${OUTPUT_BASE}-llvm-1.mlir" # MLIR LLVM IR dialect
    OUTPUT_LL="${OUTPUT_BASE}-2.ll" # LLVM IR
    OUTPUT_OBJ="${OUTPUT_BASE}-3.o" # object file
    OUTPUT_WAT="${OUTPUT_BASE}-obj-4.wat" 
    OUTPUT_BEFOREOPT_WASM="${OUTPUT_BASE}-nobinaryen-5.wasm"
    OUTPUT_BEFOREOPT_WAT="${OUTPUT_BASE}-nobinaryen-6.wat"

    echo "Converting $INPUT_MLIR to LLVM dialect..."
    mlir-opt "$INPUT_MLIR" --convert-scf-to-cf --lower-affine \
        --convert-arith-to-llvm="index-bitwidth=32" \
        --convert-func-to-llvm="index-bitwidth=32" \
        --memref-expand --expand-strided-metadata \
        --finalize-memref-to-llvm="index-bitwidth=32" \
        --convert-to-llvm --reconcile-unrealized-casts -o "$OUTPUT_LLVM_MLIR"

    echo "Translating $OUTPUT_LLVM_MLIR to LLVM IR (.ll)..."
    mlir-translate "$OUTPUT_LLVM_MLIR" --mlir-to-llvmir -o "$OUTPUT_LL"

    echo "Lowering $OUTPUT_LL to object file (.o)..."
    llc $LLVM_OPT_FLAGS -filetype=obj -mtriple=wasm32-wasi "$OUTPUT_LL" -o "$OUTPUT_OBJ"

    echo "Converting $OUTPUT_OBJ to WAT format..."
    wasm2wat "$OUTPUT_OBJ" -o "$OUTPUT_WAT"

    echo "Linking the object file with stdlib using wasm-ld..."
    # We always use -O3 optimization level
    $WASI_SDK_PATH/bin/wasm-ld --no-entry \
    --export-memory --export=_mlir_ciface_main --export=malloc --export=free \
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
    rm -f "$OUTPUT_MLIR" "$OUTPUT_RAW_WAT" "$OUTPUT_BEFOREOPT_WASM" "$OUTPUT_BEFOREOPT_WAT" "$OUTPUT_LLVM_MLIR" "$OUTPUT_LL" "$OUTPUT_OBJ" "$OUTPUT_WAT"
fi

# Print the produced files
echo "Produced files:"
if [[ "$COMPILER" == "mlir" ]]; then
    echo "  - $OUTPUT_MLIR (MLIR Wasm dialect)"
    echo "  - $OUTPUT_RAW_WAT (Raw WAT format)"
    echo "  - $OUTPUT_BEFOREOPT_WASM (Unoptimized WebAssembly)"
    echo "  - $OUTPUT_BEFOREOPT_WAT (Unoptimized WAT format)"
elif [[ "$COMPILER" == "llvm" ]]; then
    echo "  - $OUTPUT_LLVM_MLIR (LLVM dialect MLIR)"
    echo "  - $OUTPUT_LL (LLVM IR)"
    echo "  - $OUTPUT_OBJ (Object file)"
    echo "  - $OUTPUT_WAT (WAT format)"
fi

# Final optimized (or unoptimized) WebAssembly file
echo "  - $OUTPUT_WASM (Final WebAssembly output)"

echo "Conversion completed!"
