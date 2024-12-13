# Example (atax)

# llvm

```sh
# LLVM compiler (with -O3 opt), interpreter mode, no binaryen optimization
./run-mcu.sh polybench/atax_256.mlir --compiler=llvm --testcase=ATAX_LLVM --use-aot=false --llvm-opt-flags="-O3"

# LLVM compiler (with -O3 opt), interpreter mode, -O4 binaryen optimization
./run-mcu.sh polybench/atax_256.mlir --compiler=llvm --testcase=ATAX_LLVM --use-aot=false --llvm-opt-flags="-O3" --binaryen-opt-flags="-O4"

# LLVM compiler (with -O3 opt), aot mode (with -O3 opt), no binaryen optimization
./run-mcu.sh polybench/atax_256.mlir --compiler=llvm --testcase=ATAX_LLVM --use-aot=true --llvm-opt-flags="-O3" -- --opt-level=3 --target=thumbv7em --target-abi=eabihf --cpu=cortex-m4

# LLVM compiler (with -O3 opt), aot mode (with -O3 opt), -O4 binaryen optimization
./run-mcu.sh polybench/atax_256.mlir --compiler=llvm --testcase=ATAX_LLVM --use-aot=true --llvm-opt-flags="-O3" --binaryen-opt-flags="-O4" -- --opt-level=3 --target=thumbv7em --target-abi=eabihf --cpu=cortex-m4
```

# mlir

```sh
# MLIR-based compiler, interpreter mode, no binaryen optimization
./run-mcu.sh polybench/atax_256.mlir --compiler=mlir --testcase=ATAX_MLIR --use-aot=false

# MLIR-based compiler, interpreter mode, -O4 binaryen optimization
./run-mcu.sh polybench/atax_256.mlir --compiler=mlir --testcase=ATAX_MLIR --use-aot=false --binaryen-opt-flags="-O4"

# MLIR-based compiler, aot mode (with -O3 opt), no binaryen optimization
./run-mcu.sh polybench/atax_256.mlir --compiler=mlir --testcase=ATAX_MLIR --use-aot=true -- --opt-level=3 --target=thumbv7em --target-abi=eabihf --cpu=cortex-m4

# MLIR-based compiler, aot mode (with -O3 opt), -O4 binaryen optimization
./run-mcu.sh polybench/atax_256.mlir --compiler=mlir --testcase=ATAX_MLIR --use-aot=true --binaryen-opt-flags="-O4" -- --opt-level=3 --target=thumbv7em --target-abi=eabihf --cpu=cortex-m4
```