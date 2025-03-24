use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use wasmtime::{Caller, Config, Engine, Linker, Module, OptLevel, Result, Store};

const NUM_WARMUP_ITERATIONS: usize = 1;
const NUM_ITERATIONS: usize = 5;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long)]
    input: PathBuf,
    #[structopt(short, long)]
    use_aot: bool,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();

    let mut config = Config::new();
    config.cranelift_opt_level(OptLevel::Speed);
    if !opt.use_aot {
        config.target("pulley64")?;
    }

    let engine = Engine::new(&config)?;

    // Use Lazy static instead of raw static mut
    static INTERVALS: Lazy<Mutex<Vec<Duration>>> =
        Lazy::new(|| Mutex::new(Vec::with_capacity(100)));
    static LAST_TOGGLE: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));

    for i in 0..NUM_WARMUP_ITERATIONS + NUM_ITERATIONS {
        println!("Iteration {}", i);
        // Compile module
        let module = Module::from_file(&engine, opt.input.clone())?;

        // Create a new store for each iteration
        let mut store = Store::new(&engine, ());

        // Create a new linker for each iteration
        let mut linker = Linker::new(&engine);

        linker.func_wrap(
            "env",
            "print_i32",
            |_caller: Caller<'_, ()>, _value: i32| {
                // println!("Host function print_i32 called with value={}", value);
            },
        )?;

        linker.func_wrap("env", "toggle_gpio", move |_caller: Caller<'_, ()>| {
            // Only measure time for non-warmup iterations
            if i >= NUM_WARMUP_ITERATIONS {
                let now = Instant::now();
                let mut last = LAST_TOGGLE.lock().unwrap();
                match *last {
                    None => {
                        // println!("First toggle");
                        *last = Some(now);
                    }
                    Some(previous) => {
                        // println!("Second toggle");
                        INTERVALS.lock().unwrap().push(now.duration_since(previous));
                        *last = None;
                    }
                }
            }
        })?;

        // Instantiate and run
        let instance = linker.instantiate(&mut store, &module)?;
        let mlir_main_fn = instance
            .get_func(&mut store, "main")
            .expect("main not found")
            .typed::<(), i32>(&store);
        let llvm_main_fn = instance
            .get_func(&mut store, "main")
            .expect("main not found")
            .typed::<(i32, i32), i32>(&store);

        if mlir_main_fn.is_ok() {
            mlir_main_fn?.call(&mut store, ())?;
        } else if llvm_main_fn.is_ok() {
            llvm_main_fn?.call(&mut store, (1, 2))?;
        } else {
            panic!("main not found");
        }
    }

    // println!("Done");
    // println!("Intervals collected: {}", INTERVALS.lock().unwrap().len());

    // Calculate and print statistics
    let intervals = INTERVALS.lock().unwrap();
    if !intervals.is_empty() {
        let avg = intervals.iter().sum::<Duration>() / intervals.len() as u32;
        let min = intervals.iter().min().unwrap();
        let max = intervals.iter().max().unwrap();

        println!("[execution time] {:?} miliseconds", avg.as_micros());
        println!("[min execution time] {:?} miliseconds", min.as_micros());
        println!("[max execution time] {:?} miliseconds", max.as_micros());
    }

    Ok(())
}
