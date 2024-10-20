use aligned_array::{Aligned, A32, A8};
use bytemuck;
use std::fs;
use std::{path::PathBuf, str::FromStr, time::Instant};
use structopt::StructOpt;
use wasmer::{imports, Function, Instance, Module, Store, Value};

use run_wasm_lib::mnist;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long)]
    input: PathBuf,

    #[structopt(long)]
    indirect_tensor_pointer: bool,

    #[structopt(long)]
    backend: Backend,

    #[structopt(long)]
    optimize: bool,
}

#[derive(Debug, StructOpt)]
enum Backend {
    Singlepass,
    Cranelift,
    LLVM,
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "singlepass" => Ok(Backend::Singlepass),
            "cranelift" => Ok(Backend::Cranelift),
            "llvm" => Ok(Backend::LLVM),
            _ => Err(format!("Unknown backend: {}", s)),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    // Create a store with Cranelift compiler
    let mut store = match opt.backend {
        Backend::Singlepass => {
            let compiler = wasmer_compiler_singlepass::Singlepass::new();
            Store::new(compiler)
        }
        Backend::Cranelift => {
            let mut compiler = wasmer_compiler_cranelift::Cranelift::default();
            if opt.optimize {
                compiler.opt_level(wasmer_compiler_cranelift::CraneliftOptLevel::Speed);
            }
            Store::new(compiler)
        }
        Backend::LLVM => {
            unimplemented!()
        }
    };

    // Read the WebAssembly bytes
    let wasm_bytes = fs::read(opt.input)?;

    // Create a module
    let module = Module::new(&store, &wasm_bytes)?;

    // Create import object and define imported functions
    let import_object = create_imports(&mut store)?;

    // Instantiate the module
    let instance = Instance::new(&mut store, &module, &import_object)?;

    // Get the memory export
    let memory = instance.exports.get_memory("memory")?;

    // Get the malloc function
    let malloc_fn: &Function = instance.exports.get_function("malloc")?;

    let scaled_data: Vec<f32> = mnist::SAMPLE_DATA
        .iter()
        .map(|&x| (x as f32) / 255.0)
        .collect();

    // Call malloc to allocate space for the input tensor
    let tensor_ptr = malloc_fn.call(&mut store, &[Value::I32(mnist::INPUT_TENSOR_SIZE as i32)])?[0]
        .i32()
        .unwrap() as u64;

    // First get the memory view before any further use of the store
    // Copy the scaled input tensor to the WebAssembly memory
    memory
        .view(&store)
        .write(tensor_ptr, &bytemuck::cast_slice(&scaled_data))?;

    if opt.indirect_tensor_pointer {
        let main_fn = instance.exports.get_function("_mlir_ciface_main")?;

        let input_ptr = malloc_fn.call(&mut store, &[Value::I32(mnist::INPUT_SIZE as i32)])?[0]
            .i32()
            .unwrap() as u64;
        let input = mnist::Input {
            base_ptr: tensor_ptr as i32,
            data: tensor_ptr as i32,
            offset: 0,
            sizes: [1, 28, 28],
            strides: [28 * 28, 28, 1],
        };

        // Ensure the memory view is fetched before mutably borrowing the store
        memory
            .view(&store)
            .write(input_ptr, &bytemuck::bytes_of(&input))?;

        let output_tensor_ptr = malloc_fn
            .call(&mut store, &[Value::I32(mnist::OUTPUT_TENSOR_SIZE as i32)])?[0]
            .i32()
            .unwrap() as u64;
        let output_ptr = malloc_fn.call(&mut store, &[Value::I32(mnist::OUTPUT_SIZE as i32)])?[0]
            .i32()
            .unwrap() as u64;
        let output = mnist::Output {
            base_ptr: output_tensor_ptr as i32,
            data: output_tensor_ptr as i32,
            offset: 0,
            sizes: [1, 10],
            strides: [10, 1],
        };

        memory
            .view(&store)
            .write(output_ptr, &bytemuck::bytes_of(&output))?;

        let now = Instant::now();
        main_fn.call(
            &mut store,
            &[Value::I32(output_ptr as i32), Value::I32(input_ptr as i32)],
        )?;
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);

        let mut output_buffer: [u8; mnist::OUTPUT_SIZE] = [0; mnist::OUTPUT_SIZE];
        memory.view(&store).read(output_ptr, &mut output_buffer)?;

        let output: mnist::Output = bytemuck::cast(output_buffer);

        let mut output_tensor_buffer: Aligned<A32, [u8; mnist::OUTPUT_TENSOR_SIZE]> =
            Aligned([0u8; mnist::OUTPUT_TENSOR_SIZE]);
        memory
            .view(&store)
            .read(output.data as u64, output_tensor_buffer.as_mut_slice())?;

        for (i, score) in bytemuck::cast_slice::<u8, f32>(output_tensor_buffer.as_slice())
            .iter()
            .enumerate()
        {
            println!("{}: {}", i, score);
        }
    } else {
        let main_fn = instance.exports.get_function("main")?;

        let now = Instant::now();
        let output_ptr = main_fn.call(&mut store, &[Value::I32(tensor_ptr as i32)])?[0]
            .i32()
            .unwrap() as u64;
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);

        let mut output_tensor_buffer: Aligned<A8, [u8; 40]> = Aligned([0u8; 40]);
        memory
            .view(&store)
            .read(output_ptr, output_tensor_buffer.as_mut_slice())?;

        for (i, score) in bytemuck::cast_slice::<u8, f32>(output_tensor_buffer.as_slice())
            .iter()
            .enumerate()
        {
            println!("{}: {}", i, score);
        }
    }

    Ok(())
}

fn create_imports(store: &mut Store) -> Result<wasmer::Imports, Box<dyn std::error::Error>> {
    let log_i32 = Function::new_typed(store, |x: i32| {
        println!("log_i32: {}", x);
    });

    let log_f32 = Function::new_typed(store, |x: f32| {
        println!("log_f32: {}", x);
    });

    let import_object = imports! {
        "env" => {
            "log_i32" => log_i32,
            "log_f32" => log_f32,
        }
    };

    Ok(import_object)
}
