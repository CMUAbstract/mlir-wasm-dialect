# mcu-wasm-executor

This is a testing tool to run wasm/aot files on Apollo4 microcontrollers.
For now, we use a hard-coded input from MNIST dataset.


## Run
```
export ZEPHYRPROJECT=/Users/byeongje/zephyrproject

source $ZEPHYRPROJECT/.venv/bin/activate

xxd -i -n wasm_file filename src/wasm.h

west build . -b apollo4p_blue_kxr_evb -p -- -D{MODE}=1
west flash
```

- `filename` can be a path to either an `wasm` or `aot` file.
This does not support the `wat` file extenstion.

- `MODE` sets the scaffolding code that should be used to run and initialize the
wasm code.

    - MNIST_LLVM : For wasm files produced by llvm backend from
    `../test/conv2d.mlir` or `../test/lenet.mlir`. This uses the function
    defined in `src/mnist_llvm.h`.
    - MNIST_MLIR : For wasm files produced by `convert-to-wasm` pass from
    `../test/conv2d.mlir` or `../test/lenet.mlir`.. This uses the function
    defined in `src/mnist_llvm.h`. 


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
