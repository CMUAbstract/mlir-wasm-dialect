# mcu-wasm-executor

This is a testing tool to run wasm/aot files on Apollo4 microcontrollers.
For now, we use a hard-coded input from MNIST dataset.


## Run
```
export ZEPHYRPROJECT=/Users/byeongje/zephyrproject

source $ZEPHYRPROJECT/.venv/bin/activate

xxd -i -n wasm_file filename src/wasm.h

west build . -b apollo4p_blue_kxr_evb -p -- -DEXECUTION_TYPE=N
west flash
```

- `filename` can be a path to either an `wasm` or `aot` file.
This does not support the `wat` file extenstion.

- `EXECUTION_TYPE` can be either `0`, `1`, or `2`.

    - 0 : For wasm files produced by llvm backend. The input is a struct with
    base_ptr, data, offset, sizes, strides, and the entry function is
    `_mlir_ciface_main`
    - 1 : For wasm files produced by `convert-to-wasm` pass. The input is a pointer
    to the tensor data, assuming that it has the canonical layout, and the entry
    function is `main`
    - 2 : For wasm files produced by tflm compilation pipeline (see tflm-wasm
    repo). The input data is hard-coded in the wasm file and the entry function
    is `main_argc_argv`


## Monitoring Output
This example uses `printf` to produce output, which can be read from the serial
console.
I use `minicom` to monitor the console.
First, install `minicom`:
```sh
brew install minicom #  macOS
sudo apt install minicom # ubunbu
```

To find the console device, run:
```sh
ls /dev/tty.*
```
If the board is connected, an USB device of this form should be listed: `/dev/tty.usbmodemXXX`.

To monitor the device, run:
```sh
minicom -D /dev/tty.usbmodemXXX
```
