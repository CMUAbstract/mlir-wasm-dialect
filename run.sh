#!/bin/bash

# Configure this
ZEPHYRPROJECT="/Users/byeongje/zephyrproject/zephyr"

CMDARGS="$@"

# Check if the minimum number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <mlir_file> --type=[mlir|llvm] [--optimize] [--use-aot=<true|false>] -- <aot_flags>"
    exit 1
fi

# Initialize variables
MLIR_FILE=""
COMPILER="mlir"  # Default type is mlir
BINARYEN_OPT_FLAGS=""
USE_AOT=true  # Default is to use AOT
AOT_FLAGS=""
EXECUTION_TYPE=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --compiler=*)
            COMPILER="${1#*=}"
            shift
            ;;
        --binaryen-opt-flags=*)
            BINARYEN_OPT_FLAGS="${1#*=}"
            shift
            ;;
        --use-aot=*)
            USE_AOT="${1#*=}"
            shift
            ;;
        --)
            shift
            AOT_FLAGS="$@"
            break
            ;;
        *)
            MLIR_FILE=$1
            shift
            ;;
    esac
done

echo "BIN FLAGS: $BINARYEN_OPT_FLAGS"
# Check for valid values of --type
if [[ "$COMPILER" != "mlir" && "$COMPILER" != "llvm" ]]; then
    echo "Error: --type must be either 'mlir' or 'llvm'."
    exit 1
fi

# Check if MLIR file is provided
if [ -z "$MLIR_FILE" ]; then
    echo "Error: MLIR or LLVM file not provided."
    exit 1
fi

# Get the base name of the input file (without directory and extension)
BASENAME=$(basename "$MLIR_FILE" .mlir)

# Step 1: Create a temporary directory with the current datetime
TEMP_DIR=$(mktemp -d "./tmp_${BASENAME}_$(date +%Y%m%d_%H%M%S)")

echo "Temporary directory created: $TEMP_DIR"

echo $CMDARGS > $TEMP_DIR/cmdargs

# Step 2: Compile input file to Wasm (depending on the type)
if [ "$COMPILER" = "mlir" ]; then
    COMPILE_CMD="./compile.sh -i $MLIR_FILE -o $TEMP_DIR/$BASENAME --binaryen-opt-flags=\"$BINARYEN_OPT_FLAGS\""
else
    COMPILE_CMD="./compile-llvm.sh -i $MLIR_FILE -o $TEMP_DIR/$BASENAME --binaryen-opt-flags=\"$BINARYEN_OPT_FLAGS\""
fi

echo "Compiling $COMPILER to Wasm with command: $COMPILE_CMD"
eval "$COMPILE_CMD"

# Check if the Wasm file was created successfully
if [ ! -f "$TEMP_DIR/$BASENAME.wasm" ]; then
    echo "Error: Wasm file not found in $TEMP_DIR"
    exit 1
fi

# Step 3: Conditionally compile Wasm to AOT based on --use-aot
if [ "$USE_AOT" = true ]; then
    AOT_COMPILE_CMD="./mcu-wasm-executor/aot-compiler/compile_aot.sh -i $TEMP_DIR/$BASENAME.wasm -o $TEMP_DIR/$BASENAME.aot -- $AOT_FLAGS"

    echo "Compiling Wasm to AOT with command: $AOT_COMPILE_CMD"
    eval "$AOT_COMPILE_CMD"

    # Check if the AOT file was created successfully
    if [ ! -f "$TEMP_DIR/$BASENAME.aot" ]; then
        echo "Error: AOT file not found in $TEMP_DIR"
        exit 1
    fi

    # Set the file to be used for execution (AOT)
    EXEC_FILE="$TEMP_DIR/$BASENAME.aot"
else
    # Set the file to be used for execution (Wasm)
    EXEC_FILE="$TEMP_DIR/$BASENAME.wasm"
fi

# Set the execution type for LLVM or MLIR
if [ "$COMPILER" = "llvm" ]; then
    EXECUTION_TYPE=1
fi

# Step 4: Function to run the compiled file on the device
run_on_device() {
    local file=$1

    echo "Running on device using $file..."

    # Activate the virtual environment for Zephyr
    source $ZEPHYRPROJECT/.venv/bin/activate

    # Move to the MCU Wasm Executor directory and prepare to run the binary
    (cd mcu-wasm-executor && \
        xxd -i -n wasm_file "../$file" src/wasm.h && \
        west build . -b apollo4p_blue_kxr_evb -p -- -DEXECUTION_TYPE=$EXECUTION_TYPE && \
        west flash)

    # Check if the build and flash were successful
    if [ $? -eq 0 ]; then
        echo "Binary successfully flashed and executed on the device."
    else
        echo "Error: Failed to build or flash the binary."
        exit 1
    fi
}

# Step 5: Run the compiled file (either AOT or Wasm) on the device
run_on_device "$EXEC_FILE"
