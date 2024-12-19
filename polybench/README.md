# PolyBench Benchmark Suite on Apollo4 Blue Plus

This directory provides instructions for testing an MLIR-based port of the
PolyBench benchmark suite on the Apollo4 Blue Plus device. It allows 
configuration of compilers, optimization levels, and evaluation setups.

## Key Features:
- Supports LLVM or MLIR (WASM dialect-based) compilers
- Configurable LLVM optimization levels (-O0, -O1, etc.)
- Configurable Binaryen optimization levels (-O0, -O1, etc.)
- Configurable AOT (Ahead-Of-Time) compilation optimization levels (1, 2, 3)

## Prerequisites
1.	Connect the Apollo4 Blue Plus board and Saleae Logic device to the host machine.
2.	Install the Logic2 software: https://www.saleae.com/pages/downloads
3.	In Logic2, open Settings and enable the Automation Server feature.

## Running the Benchmarks
1. Setup virtualenv
	- It is recommended to install pipenv and then run `pipenv install; pipenv shell` in this directory.
2.	Execute the command below to view the preferred configuration:
```sh
./gencmds.py
```
3. To run specific benchmarks, pipe the output of gencmds.py through jq to filter by tag or compiler, then pipe that into runcmds.py. For example:
```sh
./gencmds.py | jq -c 'select(.tag == "atax" and .compiler == "mlir")' | ./runcmds.py
```
4. To run all tests at once:
```sh
./gencmds.py | ./runcmds.py
```

## Caution
Please manually verify that the Wasm files run successfully without errors (for
example, by using minicom), and confirm that you are measuring the intended
behavior.
