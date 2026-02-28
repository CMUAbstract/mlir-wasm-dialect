use crate::cli::Cli;
use crate::host_env::{register, HostState};
use crate::result_check::RunReport;
use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};
use wasmtime::{Config, Engine, Linker, Module, OptLevel, Store};

#[derive(Debug)]
pub enum RunnerError {
    Module(String),
    MissingEntry(String),
    Signature(String),
    Trap(String),
    Expectation {
        expected: i32,
        return_val: i32,
        iteration: usize,
    },
    InvalidArgs(String),
}

impl RunnerError {
    pub fn exit_code(&self) -> i32 {
        match self {
            RunnerError::Module(_) => 2,
            RunnerError::MissingEntry(_) => 3,
            RunnerError::Signature(_) | RunnerError::InvalidArgs(_) => 4,
            RunnerError::Trap(_) => 5,
            RunnerError::Expectation { .. } => 6,
        }
    }
}

impl Display for RunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RunnerError::Module(msg) => write!(f, "module error: {}", msg),
            RunnerError::MissingEntry(name) => {
                write!(f, "entry function '{}' was not found", name)
            }
            RunnerError::Signature(msg) => write!(f, "entry signature error: {}", msg),
            RunnerError::Trap(msg) => write!(f, "execution trap: {}", msg),
            RunnerError::Expectation {
                expected,
                return_val,
                iteration,
            } => write!(
                f,
                "result mismatch at iteration {}: expected {}, got {}",
                iteration, expected, return_val
            ),
            RunnerError::InvalidArgs(msg) => write!(f, "invalid arguments: {}", msg),
        }
    }
}

pub fn run(cli: &Cli) -> Result<RunReport, RunnerError> {
    if cli.iterations == 0 {
        return Err(RunnerError::InvalidArgs(
            "--iterations must be greater than zero".to_string(),
        ));
    }

    let use_aot = cli.mode == "aot";

    let mut config = Config::new();
    if use_aot {
        config.cranelift_opt_level(OptLevel::Speed);
    } else {
        config
            .target("pulley64")
            .map_err(|e| RunnerError::Module(e.to_string()))?;
    }
    let engine = Engine::new(&config).map_err(|e| RunnerError::Module(e.to_string()))?;
    let module = if use_aot {
        let wasm_bytes =
            std::fs::read(&cli.input).map_err(|e| RunnerError::Module(e.to_string()))?;
        let serialized = engine
            .precompile_module(&wasm_bytes)
            .map_err(|e| RunnerError::Module(e.to_string()))?;
        unsafe {
            Module::deserialize(&engine, &serialized)
                .map_err(|e| RunnerError::Module(e.to_string()))?
        }
    } else {
        Module::from_file(&engine, &cli.input).map_err(|e| RunnerError::Module(e.to_string()))?
    };

    let total_iterations = cli.warmup + cli.iterations;
    let mut durations: Vec<Duration> = Vec::with_capacity(cli.iterations);
    let mut measured_return_val = 0;
    let mut measured_print_count = 0;
    let mut measured_print_hash = 0;

    for iter in 0..total_iterations {
        let started = Instant::now();
        let result = run_once(&engine, &module, cli)?;
        let elapsed = result.toggle_duration.unwrap_or_else(|| started.elapsed());

        if iter >= cli.warmup {
            measured_return_val = result.return_val;
            measured_print_count = result.print_count;
            measured_print_hash = result.print_hash;
            durations.push(elapsed);

            if let Some(expected) = cli.expect_i32 {
                if result.return_val != expected {
                    return Err(RunnerError::Expectation {
                        expected,
                        return_val: result.return_val,
                        iteration: iter - cli.warmup,
                    });
                }
            }
        }
    }

    let (avg_ms, min_ms, max_ms) = summarize_ms(&durations);
    Ok(RunReport {
        expected: cli.expect_i32,
        return_val: measured_return_val,
        pass: cli
            .expect_i32
            .map(|v| v == measured_return_val)
            .unwrap_or(true),
        iterations: cli.iterations,
        warmup: cli.warmup,
        avg_ms,
        min_ms,
        max_ms,
        print_count: measured_print_count,
        print_hash: measured_print_hash,
    })
}

struct IterationResult {
    return_val: i32,
    toggle_duration: Option<Duration>,
    print_count: u64,
    print_hash: u64,
}

fn run_once(engine: &Engine, module: &Module, cli: &Cli) -> Result<IterationResult, RunnerError> {
    let mut store = Store::new(
        engine,
        HostState::new(cli.quiet, cli.print_mode.is_hash(), cli.print_hash_seed),
    );
    let mut linker = Linker::new(engine);
    register(&mut linker).map_err(|e| RunnerError::Module(e.to_string()))?;

    let instance = linker
        .instantiate(&mut store, module)
        .map_err(|e| RunnerError::Module(e.to_string()))?;

    let entry = instance
        .get_func(&mut store, &cli.entry)
        .ok_or_else(|| RunnerError::MissingEntry(cli.entry.clone()))?;

    let return_val = if let Ok(typed) = entry.typed::<(), i32>(&store) {
        typed
            .call(&mut store, ())
            .map_err(|e| RunnerError::Trap(e.to_string()))?
    } else if let Ok(typed) = entry.typed::<(i32, i32), i32>(&store) {
        typed
            .call(&mut store, (0, 0))
            .map_err(|e| RunnerError::Trap(e.to_string()))?
    } else {
        return Err(RunnerError::Signature(
            "expected entry signature () -> i32 or (i32, i32) -> i32".to_string(),
        ));
    };

    let print_count = store.data().print_count();
    let print_hash = store.data().print_hash();

    Ok(IterationResult {
        return_val,
        toggle_duration: store.data().latest_toggle_duration(),
        print_count,
        print_hash,
    })
}

fn summarize_ms(samples: &[Duration]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut min = samples[0];
    let mut max = samples[0];
    let mut total = Duration::ZERO;

    for sample in samples {
        if *sample < min {
            min = *sample;
        }
        if *sample > max {
            max = *sample;
        }
        total += *sample;
    }

    let avg = total.as_secs_f64() * 1000.0 / samples.len() as f64;
    (avg, min.as_secs_f64() * 1000.0, max.as_secs_f64() * 1000.0)
}
