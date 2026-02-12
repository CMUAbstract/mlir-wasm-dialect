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
        actual: i32,
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
                actual,
                iteration,
            } => write!(
                f,
                "result mismatch at iteration {}: expected {}, got {}",
                iteration, expected, actual
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

    let mut config = Config::new();
    config.cranelift_opt_level(OptLevel::Speed);
    let engine = Engine::new(&config).map_err(|e| RunnerError::Module(e.to_string()))?;
    let module =
        Module::from_file(&engine, &cli.input).map_err(|e| RunnerError::Module(e.to_string()))?;

    let total_iterations = cli.warmup + cli.iterations;
    let mut durations: Vec<Duration> = Vec::with_capacity(cli.iterations);
    let mut measured_actual = 0;

    for iter in 0..total_iterations {
        let started = Instant::now();
        let result = run_once(&engine, &module, cli)?;
        let elapsed = result.toggle_duration.unwrap_or_else(|| started.elapsed());

        if iter >= cli.warmup {
            measured_actual = result.actual;
            durations.push(elapsed);

            if let Some(expected) = cli.expect_i32 {
                if result.actual != expected {
                    return Err(RunnerError::Expectation {
                        expected,
                        actual: result.actual,
                        iteration: iter - cli.warmup,
                    });
                }
            }
        }
    }

    let (avg_ms, min_ms, max_ms) = summarize_ms(&durations);
    Ok(RunReport {
        expected: cli.expect_i32,
        actual: measured_actual,
        pass: cli.expect_i32.map(|v| v == measured_actual).unwrap_or(true),
        iterations: cli.iterations,
        warmup: cli.warmup,
        avg_ms,
        min_ms,
        max_ms,
    })
}

struct IterationResult {
    actual: i32,
    toggle_duration: Option<Duration>,
}

fn run_once(engine: &Engine, module: &Module, cli: &Cli) -> Result<IterationResult, RunnerError> {
    let mut store = Store::new(engine, HostState::new(cli.quiet));
    let mut linker = Linker::new(engine);
    register(&mut linker).map_err(|e| RunnerError::Module(e.to_string()))?;

    let instance = linker
        .instantiate(&mut store, module)
        .map_err(|e| RunnerError::Module(e.to_string()))?;

    let entry = instance
        .get_func(&mut store, &cli.entry)
        .ok_or_else(|| RunnerError::MissingEntry(cli.entry.clone()))?;

    let typed = entry
        .typed::<(), i32>(&store)
        .map_err(|e| RunnerError::Signature(e.to_string()))?;

    let actual = typed
        .call(&mut store, ())
        .map_err(|e| RunnerError::Trap(e.to_string()))?;

    Ok(IterationResult {
        actual,
        toggle_duration: store.data().latest_toggle_duration(),
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
