use aligned_array::{Aligned, A32};
use bytemuck;
use std::time::Instant;
use wasmtime::{Engine, Instance, Module, Result, Store};

mod mnist;

fn main() -> Result<()> {
    // Currently, we use WASI because enabling WASI is the easiest way to link
    // libc functions (malloc and free) at compile time. However, we need to
    // eliminate the WASI dependency in the future since we are not using any
    // WASI functions.
    let engine = Engine::default();
    let module = Module::from_file(&engine, "../test/conv2d-linked.wat")?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory not found");

    let malloc_fn = instance
        .get_func(&mut store, "malloc")
        .expect("malloc not found")
        .typed::<i32, i32>(&store)?;

    let main_fn = instance
        .get_func(&mut store, "_mlir_ciface_main")
        .expect("_mlir_ciface_main not found")
        .typed::<(i32, i32), ()>(&store)?;

    let scaled_data: Vec<_> = mnist::SAMPLE_DATA
        .into_iter()
        .map(|x| (x as f32) / 255.0)
        .collect();

    let tensor_ptr = malloc_fn.call(&mut store, mnist::INPUT_TENSOR_SIZE as i32)?;
    memory.data_mut(&mut store)
        [tensor_ptr as usize..tensor_ptr as usize + mnist::INPUT_TENSOR_SIZE]
        .copy_from_slice(bytemuck::cast_slice(&scaled_data));

    let input_ptr = malloc_fn.call(&mut store, mnist::INPUT_SIZE as i32)?;
    let input = mnist::Input {
        base_ptr: tensor_ptr,
        data: tensor_ptr,
        offset: 0,
        sizes: [1, 28, 28],
        strides: [28 * 28, 28, 1],
    };
    memory.data_mut(&mut store)[input_ptr as usize..input_ptr as usize + mnist::INPUT_SIZE]
        .copy_from_slice(bytemuck::bytes_of(&input));

    let output_tensor_ptr = malloc_fn.call(&mut store, mnist::OUTPUT_TENSOR_SIZE as i32)?;
    let output_ptr = malloc_fn.call(&mut store, mnist::OUTPUT_SIZE as i32)?;
    let output = mnist::Output {
        base_ptr: output_tensor_ptr,
        data: output_tensor_ptr,
        offset: 0,
        sizes: [1, 10],
        strides: [10, 1],
    };
    memory.data_mut(&mut store)[output_ptr as usize..output_ptr as usize + mnist::OUTPUT_SIZE]
        .copy_from_slice(bytemuck::bytes_of(&output));

    let now = Instant::now();
    main_fn.call(&mut store, (output_ptr, input_ptr))?;
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    let mut output_buffer = [0u8; mnist::OUTPUT_SIZE];
    memory.read(&store, output_ptr as usize, &mut output_buffer)?;

    let output: mnist::Output = bytemuck::cast(output_buffer);

    // We need aligned array because
    // we are converting byte array to f32 array
    //
    let mut output_tensor_buffer: Aligned<A32, [u8; 40]> =
        Aligned([0u8; mnist::OUTPUT_TENSOR_SIZE]);
    memory.read(
        &store,
        output.data as usize,
        output_tensor_buffer.as_mut_slice(),
    )?;

    for (i, score) in bytemuck::cast_slice::<u8, f32>(output_tensor_buffer.as_slice())
        .iter()
        .enumerate()
    {
        println!("{}: {}", i, score);
    }
    // expected output

    //    Elapsed: 28.46µs
    //    0: 8.469532
    //    1: 0
    //    2: 0
    //    3: 0
    //    4: 0
    //    5: 0
    //    6: 0
    //    7: 0
    //    8: 0
    //    9: 0
    Ok(())
}
