use std::fs;
use std::{path::PathBuf, time::Instant};
use structopt::StructOpt;
use wasmtime::{Config, Engine, Linker, Module, OptLevel, Result, Store};

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long)]
    input: PathBuf,
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
    linker.func_wrap("env", "log", || {
        println!("hi");
    })?;

    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    let main_fn = instance
        .get_func(&mut store, "main")
        .expect("main not found")
        .typed::<(), f32>(&store)?;

    let now = Instant::now();
    let result = main_fn.call(&mut store, ())?;
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("Result: {}", result);

    Ok(())
}
