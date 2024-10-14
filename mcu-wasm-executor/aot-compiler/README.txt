# AOT compiler

We can use `compile_aot.sh` to compile wasm files into aot files.
Since WAMR has limited support for Mac, we use docker to build and use the wamr
compiler.
The script currently uses the docker image `byeongjeecmu/wamr:latest`, which is
compiled from the Dockerfile (taken from WAMR devcontainer).
