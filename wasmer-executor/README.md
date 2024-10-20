# Interpreter

Simple WebAssembly interpreter to test and debug wasm files.
For now, we use a hard-coded input from MNIST dataset.

## Usage
To test wasm/wat files produced from the `convert-to-wasm` pass, run the following:
```
cargo run -- -i <filename>
```
To test wasm/wat files produced from the LLVM backend, run the following:
```
cargo run -- -i <filename> --indirect-tensor-pointer
```
`<filename>` can be a path to either a `.wasm` or `.wat` file.

## Debugging
This interpreter implements host-side functions `log_i32` and `log_f32` to make
debugging wasm code easier.
The wasm/wat code may import these functions (from `env`) and use them to print
stack values.