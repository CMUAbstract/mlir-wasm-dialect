#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CMDARGS="$@"

# Check if the minimum number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <mlir_file> --compiler=[wami|llvm] [--use-aot=<true|false>] [--silent] -- <aot_flags>"
    exit 1
fi

# Initialize variables
MLIR_FILE=""
COMPILER="wami"  # Default type is wami
DEVICE="mcu"  # Default device is mcu
LLVM_OPT_FLAGS=""
BINARYEN_OPT_FLAGS=""
USE_AOT=false  # Default is to use interpreter
AOT_FLAGS=""
SILENT=false
ITERATIONS=1
WARMUP=0
SKIP_BUILD=false
WAMI_PREPROCESS=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --compiler=*)
            COMPILER="${1#*=}"
            shift
            ;;
        --llvm-opt-flags=*)
            LLVM_OPT_FLAGS="${1#*=}"
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
        --iterations=*)
            ITERATIONS="${1#*=}"
            shift
            ;;
        --warmup=*)
            WARMUP="${1#*=}"
            shift
            ;;
        --silent)
            SILENT=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --wami-preprocess)
            WAMI_PREPROCESS=true
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

# Check for valid values of --compiler
if [[ "$COMPILER" != "wami" && "$COMPILER" != "llvm" ]]; then
    echo "Error: --compiler must be either 'wami' or 'llvm'."
    exit 1
fi

# Check if MLIR file is provided
if [ -z "$MLIR_FILE" ]; then
    echo "Error: MLIR or LLVM file not provided."
    exit 1
fi

# Get the base name of the input file (without directory and extension)
BASENAME=$(basename "$MLIR_FILE" .mlir)

# Step 1: Create a temporary directory with a random hash
RANDOM_HASH=$(openssl rand -hex 4)  # generates 8-character random hex string
TEMP_DIR="./tmp/tmp_${BASENAME}_${RANDOM_HASH}"
mkdir -p "$TEMP_DIR"

echo "Temporary directory created: $TEMP_DIR"

echo $CMDARGS > $TEMP_DIR/cmdargs

# Step 2: Compile input file to Wasm (depending on the type)
SKIP_BUILD_FLAG=""
if [ "$SKIP_BUILD" = true ]; then
    SKIP_BUILD_FLAG="--skip-build"
fi
WAMI_PREPROCESS_FLAG=""
if [ "$WAMI_PREPROCESS" = true ]; then
    WAMI_PREPROCESS_FLAG="--wami-preprocess"
fi
COMPILE_CMD="\"$SCRIPT_DIR/compile.sh\" -i $MLIR_FILE -o $TEMP_DIR/$BASENAME --compiler=$COMPILER --llvm-opt-flags=\"$LLVM_OPT_FLAGS\"  --binaryen-opt-flags=\"$BINARYEN_OPT_FLAGS\" $SKIP_BUILD_FLAG $WAMI_PREPROCESS_FLAG"

echo "Compiling $COMPILER to Wasm with command: $COMPILE_CMD"
eval "$COMPILE_CMD"

# Check if the Wasm file was created successfully
if [ ! -f "$TEMP_DIR/$BASENAME.wasm" ]; then
    echo "Error: Wasm file not found in $TEMP_DIR"
    exit 1
fi

# Step 3: Conditionally compile Wasm to AOT based on --use-aot
if [ "$USE_AOT" = true ] && [ "$DEVICE" = "mcu" -o "$DEVICE" = "local_wamr" ]; then
    AOT_COMPILE_CMD="\"$SCRIPT_DIR/aot-compiler/compile_aot.sh\" -i $TEMP_DIR/$BASENAME.wasm -o $TEMP_DIR/$BASENAME.aot -- $AOT_FLAGS"

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

run_local_wasmtime() {
    local file=$1
    local abs_file
    abs_file="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"

    echo "Running on mac using $file..."

    if [ "$USE_AOT" = true ]; then
        local MODE="aot"
    else
        local MODE="interpreter"
    fi

    if [ "$SKIP_BUILD" = true ]; then
        COMMAND_GROUP='
            "$SCRIPT_DIR/wasmtime-executor/target/release/run_wasm_bin" --mode '"$MODE"' --quiet --iterations '"$ITERATIONS"' --warmup '"$WARMUP"' --input "'"$abs_file"'"
        '
    else
        COMMAND_GROUP='
            cd "$SCRIPT_DIR/wasmtime-executor" && \
            cargo run --release -- --mode '"$MODE"' --quiet --iterations '"$ITERATIONS"' --warmup '"$WARMUP"' --input "'"$abs_file"'"
        '
    fi

    if [ "$SILENT" = true ]; then
        (eval "$COMMAND_GROUP") > /dev/null 2>&1
    else
        (eval "$COMMAND_GROUP")
    fi



}

run_local_wamr() {
    local file=$1
    local abs_file
    abs_file="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"

    echo "Running on mac using $file..."

    COMMAND_GROUP='
        cd "$SCRIPT_DIR/local-executor" && \
        xxd -i -n wasm_file "$abs_file" src/wasm.h && \
        cmake -S . -B build -DBENCH_WARMUP='"$WARMUP"' -DBENCH_ITERATIONS='"$ITERATIONS"' && \
        cmake --build build && \
        ./build/app
    '

    if [ "$SILENT" = true ]; then
        (eval "$COMMAND_GROUP") > /dev/null 2>&1
    else
        (eval "$COMMAND_GROUP")
    fi



}

run_local_node() {
    local file=$1
    local abs_file
    abs_file="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"

    echo "Running on mac using $file..."

    COMMAND_GROUP='
        node "$SCRIPT_DIR/node-executor/run_wasm.mjs" \
            --quiet --iterations '"$ITERATIONS"' --warmup '"$WARMUP"' --input "'"$abs_file"'"
    '

    if [ "$SILENT" = true ]; then
        (eval "$COMMAND_GROUP") > /dev/null 2>&1
    else
        (eval "$COMMAND_GROUP")
    fi
}

# Step 4: Function to run the compiled file on the device
run_on_device() {
    local file=$1
    local abs_file
    abs_file="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"

    if [ -z "$ZEPHYRPROJECT" ]; then
        echo "Error: ZEPHYRPROJECT environment variable is not set (required for --device=mcu)."
        exit 1
    fi

    echo "Running on device using $file..."

    # Activate the virtual environment for Zephyr
    source $ZEPHYRPROJECT/.venv/bin/activate

    # Move to the MCU Wasm Executor directory and prepare to run the binary
    # Define your command group
    COMMAND_GROUP='
        cd "$SCRIPT_DIR/mcu-wasm-executor" && \
        xxd -i -n wasm_file "$abs_file" src/wasm.h && \
        west build . -b apollo4p_blue_kxr_evb -p && \
        west flash
    '

    # Execute with or without silencing
    if [ "$SILENT" = true ]; then
        (eval "$COMMAND_GROUP") > /dev/null 2>&1
    else
        (eval "$COMMAND_GROUP")
    fi

    # Check if the build and flash were successful
    if [ $? -eq 0 ]; then
        echo "Binary successfully flashed and executed on the device."
    else
        echo "Error: Failed to build or flash the binary."
        exit 1
    fi
}

# Step 5: Run the compiled file (either AOT or Wasm) on the device
if [ "$DEVICE" = "local_wamr" ]; then
    run_local_wamr "$EXEC_FILE"
elif [ "$DEVICE" = "local_wasmtime" ]; then
    run_local_wasmtime "$EXEC_FILE"
elif [ "$DEVICE" = "local_node" ]; then
    run_local_node "$EXEC_FILE"
elif [ "$DEVICE" = "mcu" ]; then
    run_on_device "$EXEC_FILE"
else
    echo "Error: Invalid device specified. Use 'local' or 'mcu'."
fi
