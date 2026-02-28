use anyhow::Result;
use std::time::{Duration, Instant};
use wasmtime::{Caller, Linker};

const FNV1A64_PRIME: u64 = 1099511628211;

#[derive(Debug, Clone)]
pub struct HostState {
    pub quiet: bool,
    print_hash_only: bool,
    toggle_started_at: Option<Instant>,
    toggle_durations: Vec<Duration>,
    print_count: u64,
    print_hash: u64,
}

impl HostState {
    pub fn new(quiet: bool, print_hash_only: bool, print_hash_seed: u64) -> Self {
        Self {
            quiet,
            print_hash_only,
            toggle_started_at: None,
            toggle_durations: Vec::new(),
            print_count: 0,
            print_hash: print_hash_seed,
        }
    }

    fn on_toggle(&mut self) {
        if let Some(started_at) = self.toggle_started_at.take() {
            self.toggle_durations.push(started_at.elapsed());
            return;
        }
        self.toggle_started_at = Some(Instant::now());
    }

    pub fn latest_toggle_duration(&self) -> Option<Duration> {
        self.toggle_durations.last().copied()
    }

    fn on_print_i32(&mut self, value: i32) {
        self.print_count += 1;
        for byte in value.to_le_bytes() {
            self.print_hash ^= u64::from(byte);
            self.print_hash = self.print_hash.wrapping_mul(FNV1A64_PRIME);
        }

        if !self.print_hash_only && !self.quiet {
            println!("print_i32: {}", value);
        }
    }

    pub fn print_count(&self) -> u64 {
        self.print_count
    }

    pub fn print_hash(&self) -> u64 {
        self.print_hash
    }
}

pub fn register(linker: &mut Linker<HostState>) -> Result<()> {
    linker.func_wrap("env", "toggle_gpio", |mut caller: Caller<'_, HostState>| {
        caller.data_mut().on_toggle();
    })?;

    linker.func_wrap(
        "env",
        "print_i32",
        |mut caller: Caller<'_, HostState>, value: i32| {
            caller.data_mut().on_print_i32(value);
        },
    )?;

    Ok(())
}
