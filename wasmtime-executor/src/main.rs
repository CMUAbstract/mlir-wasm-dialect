use aligned_array::{Aligned, A32, A8};
use bytemuck;
use run_wasm_lib::mnist;
use std::fs;
use std::{path::PathBuf, time::Instant};
use structopt::StructOpt;
use wasmtime::{Config, Engine, Linker, Module, OptLevel, Result, Store};

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long)]
    input: PathBuf,

    #[structopt(long)]
    indirect_tensor_pointer: bool,
    // TODO: Take input tensor as argument
    #[structopt(long)]
    aot: bool,
    #[structopt(long)]
    optimize: bool,
}

// test code for wasm code that takes an input tensor and returns output tensor
// (10xf32)
fn main() -> Result<()> {
    let opt = Opt::from_args();

    let mut config = Config::new();
    config.cranelift_opt_level(if opt.optimize {
        OptLevel::Speed
    } else {
        OptLevel::None
    });

    let engine = Engine::new(&config)?;
    let wasm_bytes = fs::read(opt.input)?;

    let module = if opt.aot {
        let compiled_wasm_bytes = engine.precompile_module(&wasm_bytes)?;
        unsafe { Module::deserialize(&engine, &compiled_wasm_bytes)? }
    } else {
        Module::from_binary(&engine, &wasm_bytes)?
    };
    let mut linker = Linker::new(&engine);
    // these log functions are useful to debug wasm code
    linker.func_wrap("env", "log_i32", |x: i32| -> i32 {
        println!("log_i32: {}", x);
        x
    })?;
    linker.func_wrap("env", "log_f32", |x: f32| -> f32 {
        println!("log_f32: {}", x);
        x
    })?;

    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory not found");

    let malloc_fn = instance
        .get_func(&mut store, "malloc")
        .expect("malloc not found")
        .typed::<i32, i32>(&store)?;

    let scaled_data: Vec<_> = mnist::SAMPLE_DATA
        .into_iter()
        .map(|x| (x as f32) / 255.0)
        .collect();

    let tensor_ptr = malloc_fn.call(&mut store, mnist::INPUT_TENSOR_SIZE as i32)?;
    memory.data_mut(&mut store)
        [tensor_ptr as usize..tensor_ptr as usize + mnist::INPUT_TENSOR_SIZE]
        .copy_from_slice(bytemuck::cast_slice(&scaled_data));

    if opt.indirect_tensor_pointer {
        let main_fn = instance
            .get_func(&mut store, "_mlir_ciface_main")
            .expect("_mlir_ciface_main not found")
            .typed::<(i32, i32), ()>(&store)?;

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
    } else {
        let main_fn = instance
            .get_func(&mut store, "main")
            .expect("main not found")
            .typed::<i32, i32>(&store)?;

        let now = Instant::now();
        let output_ptr = main_fn.call(&mut store, tensor_ptr)?;
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);

        let mut output_tensor_buffer: Aligned<A8, [u8; 40]> = Aligned([0u8; 40]);
        memory.read(
            &store,
            output_ptr as usize,
            output_tensor_buffer.as_mut_slice(),
        )?;

        for (i, score) in bytemuck::cast_slice::<u8, f32>(output_tensor_buffer.as_slice())
            .iter()
            .enumerate()
        {
            println!("{}: {}", i, score);
        }
    }

    Ok(())
}
