# Run
```
export ZEPHYRPROJECT=/Users/byeongje/zephyrproject

source $ZEPHYRPROJECT/.venv/bin/activate

xxd -i -n wasm_file filename src/wasm.h

west build . -b apollo4p_blue_kxr_evb -p
west flash
```

`filename` can be a path to either an `wasm` or `aot` file.
This does not support the `wat` file extenstion.


# Monitoring Output
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
