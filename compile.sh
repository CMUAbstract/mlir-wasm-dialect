#!/bin/bash

WASI_SDK_PATH=/Users/byeongje/wasm/wasi-sdk-22.0

# Default values for input and output
OPTIMIZE=false
ADD_DEBUG_FUNCTIONS=false

# Function to display usage information
usage() {
    echo "Usage: $0 -i <input_mlir_file> -o <output_base_name> [--optimize]"
    echo "  -i, --input      Input MLIR file"
    echo "  -o, --output     Output base name"
    echo "  --optimize       Perform WebAssembly optimization (optional)"
    echo "  --add-debug-functions    Add debug functions to the output (optional)"
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
        --add-debug-functions)
            ADD_DEBUG_FUNCTIONS=true
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
OUTPUT_MLIR="${OUTPUT_BASE}.mlir"
OUTPUT_WAT="${OUTPUT_BASE}.wat"
OUTPUT_WASM="${OUTPUT_BASE}.wasm"
OUTPUT_FORMATTED_WAT="${OUTPUT_BASE}-formatted.wat"
OUTPUT_OPTIMIZED_WAT="${OUTPUT_BASE}-optimized.wat"

# Convert MLIR file to the Wasm dialect
echo "Converting $INPUT_MLIR to Wasm dialect..."
build/bin/wasm-opt --convert-to-wasm --reconcile-unrealized-casts --wasm-finalize "$INPUT_MLIR" -o "$OUTPUT_MLIR"

# Translate the resulting MLIR file to a .wat file
echo "Translating $OUTPUT_MLIR to .wat format..."
if $ADD_DEBUG_FUNCTIONS; then
    build/bin/wasm-translate "$OUTPUT_MLIR" --mlir-to-wat --add-debug-functions -o "$OUTPUT_WAT"
else
    build/bin/wasm-translate "$OUTPUT_MLIR" --mlir-to-wat -o "$OUTPUT_WAT"
fi

# Improve .wat readability by converting to .wasm and back to .wat
# This is for debugging purposes only
echo "Improving .wat readability by converting to .wasm and back to .wat..."
wat2wasm "$OUTPUT_WAT" -o "$OUTPUT_WASM"
wasm2wat "$OUTPUT_WASM" -o "$OUTPUT_FORMATTED_WAT"




# Conditionally optimize the WebAssembly output using wasm-opt from Binaryen
if $OPTIMIZE; then
    echo "Optimizing the WebAssembly output..."
    wasm-opt "$OUTPUT_WASM" -O4 -o "$OUTPUT_WASM"
    # Produce formatted .wat file of the optimized wasm
    echo "Print the wat file..."
    wasm2wat "$OUTPUT_WASM" -o "$OUTPUT_OPTIMIZED_WAT"
else
    echo "Skipping WebAssembly optimization..."
fi



echo "Conversion completed! The output files are:"
echo "  MLIR: $OUTPUT_MLIR"
echo "  WAT: $OUTPUT_WAT"
echo "  Formatted WAT: $OUTPUT_FORMATTED_WAT"
echo "  WASM: $OUTPUT_WASM"
if $OPTIMIZE; then
echo "  Optimized WAT: $OUTPUT_OPTIMIZED_WAT"
fi
