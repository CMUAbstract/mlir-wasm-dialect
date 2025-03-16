#!/bin/bash

DOCKER_IMAGE="byeongjeecmu/wamr:latest" 

INPUT_FILE=""
OUTPUT_FILE=""
OPTIMIZATION_FLAGS=""

# Function to display help message
usage() {
    echo "Usage: $0 -i <input_file> -o <output_file> -- [optimization_flags]"
    exit 1
}

# Parse the input arguments using getopts
while getopts ":i:o:" opt; do
    case ${opt} in
        i )
            INPUT_FILE=$OPTARG
            ;;
        o )
            OUTPUT_FILE=$OPTARG
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Option -$OPTARG requires an argument." 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# After the input and output options, the remaining arguments are the docker command and flags
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ] || [ $# -eq 0 ]; then
    usage
fi

OPTIMIZATION_FLAGS="$@"


INPUT_DIR=$(dirname "$INPUT_FILE")
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
INPUT_DIR=$(cd "$INPUT_DIR" && pwd)
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
INPUT_BASENAME=$(basename "$INPUT_FILE")
OUTPUT_BASENAME=$(basename "$OUTPUT_FILE")

# Copy the input wasm file into the container and perform the computation
docker run --rm \
    -v "$INPUT_DIR":/input_dir \
    -v "$OUTPUT_DIR":/output_dir \
    -w /input_dir \
    "$DOCKER_IMAGE" /bin/bash -c "
    cp $INPUT_BASENAME /workspace/input.wasm &&
    /workspace/wasm-micro-runtime/wamr-compiler/build/wamrc \
    -o /workspace/output.aot $OPTIMIZATION_FLAGS /workspace/input.wasm \
    && cp /workspace/output.aot /output_dir/$OUTPUT_BASENAME"

# Step 3: Check if the result was copied successfully
if [ -f "$OUTPUT_FILE" ]; then
    echo "Computation completed. The result has been saved to $OUTPUT_FILE."
else
    echo "Computation failed or result not found."
fi

