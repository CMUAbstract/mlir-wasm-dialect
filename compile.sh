#!/bin/bash

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
OUTPUT_MLIR="${OUTPUT_BASE}.mlir"
OUTPUT_WAT="${OUTPUT_BASE}.wat"
OUTPUT_OBJ="${OUTPUT_BASE}.o"
OUTPUT_FORMATTED_WAT="${OUTPUT_BASE}-formatted.wat"
OUTPUT_OPTIMIZED_OBJ="${OUTPUT_BASE}-optimized.obj"
OUTPUT_LINKED_WASM="${OUTPUT_BASE}-linked.wasm"
OUTPUT_LINKED_FORMATTED_WAT="${OUTPUT_BASE}-linked-formatted.wat"

# Convert MLIR file to the Wasm dialect
echo "Converting $INPUT_MLIR to Wasm dialect..."
build/bin/wasm-opt --convert-to-wasm --reconcile-unrealized-casts --wasm-finalize "$INPUT_MLIR" -o "$OUTPUT_MLIR"

# Translate the resulting MLIR file to a .wat file
echo "Translating $OUTPUT_MLIR to .wat format..."
build/bin/wasm-translate "$OUTPUT_MLIR" --mlir-to-wat -o "$OUTPUT_WAT"

# Improve .wat readability by converting to .wasm and back to .wat
# This is for debugging purposes only
echo "Improving .wat readability by converting to .wasm and back to .wat..."
wat2wasm --relocatable "$OUTPUT_WAT" -o "$OUTPUT_OBJ"
wasm2wat "$OUTPUT_OBJ" -o "$OUTPUT_FORMATTED_WAT"

# Conditionally optimize the WebAssembly output using wasm-opt from Binaryen
if $OPTIMIZE; then
    echo "Optimizing the WebAssembly output..."
    wasm-opt "$OUTPUT_OBJ" -O4 -o "$OUTPUT_OPTIMIZED_OBJ"
    FINAL_OBJ="$OUTPUT_OPTIMIZED_OBJ"
else
    echo "Skipping WebAssembly optimization..."
    FINAL_OBJ="$OUTPUT_OBJ"
fi

# Link the object file with the standard library using wasm-ld
echo "Linking the object file with stdlib..."
$WASI_SDK_PATH/bin/wasm-ld --no-entry \
--export-memory --export=main --export=malloc --export=free \
-L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
-o "$OUTPUT_LINKED_WASM" "$FINAL_OBJ"



# Produce formatted .wat file of the linked Wasm file
echo "Formatting the linked Wasm file..."
wasm2wat "$OUTPUT_LINKED_WASM" -o "$OUTPUT_LINKED_FORMATTED_WAT"

echo "Conversion completed! The output files are:"
echo "  MLIR: $OUTPUT_MLIR"
echo "  WAT: $OUTPUT_WAT"
echo "  Formatted WAT: $OUTPUT_FORMATTED_WAT"
if $OPTIMIZE; then
    echo "  Optimized WASM: $OUTPUT_OPTIMIZED_WASM"
fi
echo "  Linked WASM: $OUTPUT_LINKED_WASM"
echo "  Linked Formatted WAT: $OUTPUT_LINKED_FORMATTED_WAT"
