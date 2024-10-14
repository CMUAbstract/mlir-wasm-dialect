#!/bin/bash

# Configure this
ZEPHYRPROJECT="/Users/byeongje/zephyrproject/zephyr"

# Check if the minimum number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <mlir_file> --type=[mlir|llvm] [--optimize] -- <aot_flags>"
    exit 1
fi

# Initialize variables
MLIR_FILE=""
COMPILATION_TYPE="mlir"  # Default type is mlir
OPTIMIZE_FLAG=false
AOT_FLAGS=""
EXECUTION_TYPE=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --type=*)
            COMPILATION_TYPE="${1#*=}"
            shift
            ;;
        --optimize)
            OPTIMIZE_FLAG=true
            shift
            ;;
        --)
            shift
            AOT_FLAGS="$1"
            break
            ;;
        *)
            MLIR_FILE=$1
            shift
            ;;
    esac
done

# Check for valid values of --type
if [[ "$COMPILATION_TYPE" != "mlir" && "$COMPILATION_TYPE" != "llvm" ]]; then
    echo "Error: --type must be either 'mlir' or 'llvm'."
    exit 1
fi

# Check if MLIR file is provided
if [ -z "$MLIR_FILE" ]; then
    echo "Error: MLIR or LLVM file not provided."
    exit 1
fi

# Check if AOT flags were provided
if [ -z "$AOT_FLAGS" ]; then
    echo "Error: AOT compilation flags not provided."
    exit 1
fi

# Get the base name of the input file (without directory and extension)
BASENAME=$(basename "$MLIR_FILE" .mlir)

# Step 1: Create a temporary directory with the current datetime
TEMP_DIR=$(mktemp -d "./tmp_${BASENAME}_$(date +%Y%m%d_%H%M%S)")

echo "Temporary directory created: $TEMP_DIR"

# Step 2: Compile input file to Wasm (depending on the type)
if [ "$COMPILATION_TYPE" = "mlir" ]; then
    COMPILE_CMD="./compile.sh -i $MLIR_FILE -o $TEMP_DIR/$BASENAME"
else
    COMPILE_CMD="./compile-llvm.sh -i $MLIR_FILE -o $TEMP_DIR/$BASENAME"
fi

# Add optimization flag if true
if [ "$OPTIMIZE_FLAG" = true ]; then
    COMPILE_CMD="$COMPILE_CMD --optimize"
fi

echo "Compiling $COMPILATION_TYPE to Wasm with command: $COMPILE_CMD"
eval "$COMPILE_CMD"

# Check if the Wasm file was created successfully
if [ ! -f "$TEMP_DIR/$BASENAME.wasm" ]; then
    echo "Error: Wasm file not found in $TEMP_DIR"
    exit 1
fi

# Step 3: Compile Wasm to AOT
AOT_COMPILE_CMD="./mcu-wasm-executor/aot-compiler/compile_aot.sh -i $TEMP_DIR/$BASENAME.wasm -o $TEMP_DIR/$BASENAME.aot -- $AOT_FLAGS"

echo "Compiling Wasm to AOT with command: $AOT_COMPILE_CMD"
eval "$AOT_COMPILE_CMD"

# Check if the AOT file was created successfully
if [ ! -f "$TEMP_DIR/$BASENAME.aot" ]; then
    echo "Error: AOT file not found in $TEMP_DIR"
    exit 1
fi

# Step 4: Run AOT on device
echo "Running AOT on device..."

# Activate the virtual environment for Zephyr
source $ZEPHYRPROJECT/.venv/bin/activate

if [ "$COMPILATION_TYPE" = "llvm" ]; then
    EXECUTION_TYPE=1
fi

# Move to the MCU Wasm Executor directory and prepare to run the AOT binary
(cd mcu-wasm-executor && \
    xxd -i -n wasm_file "../$TEMP_DIR/$BASENAME.aot" src/wasm.h && \
    west build . -b apollo4p_blue_kxr_evb -p -- -DEXECUTION_TYPE=$EXECUTION_TYPE && \
    west flash)

# Check if the build and flash were successful
if [ $? -eq 0 ]; then
    echo "AOT binary successfully flashed and executed on the device."
else
    echo "Error: Failed to build or flash the AOT binary."
    exit 1
fi
