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



# Copy the input wasm file into the container and perform the computation
docker run --rm -v "$(pwd)":/host -w /host "$DOCKER_IMAGE" /bin/bash -c "
    cp $INPUT_FILE /workspace/input.wasm &&
    /workspace/wasm-micro-runtime/wamr-compiler/build/wamrc \
    -o /workspace/output.aot $OPTIMIZATION_FLAGS /workspace/input.wasm \
    && cp /workspace/output.aot /host/$OUTPUT_FILE"

# Step 3: Check if the result was copied successfully
if [ -f "$OUTPUT_FILE" ]; then
    echo "Computation completed. The result has been saved to $OUTPUT_FILE."
else
    echo "Computation failed or result not found."
fi

