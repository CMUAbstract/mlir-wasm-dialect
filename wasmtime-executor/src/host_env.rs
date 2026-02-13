use anyhow::{anyhow, Result};
use std::time::{Duration, Instant};
use wasmtime::{Caller, Extern, Linker, Memory};

const WASM_PAGE_SIZE: u64 = 65536;
const HEAP_ALIGNMENT: u64 = 8;
const FNV1A64_PRIME: u64 = 1099511628211;

#[derive(Debug, Clone)]
pub struct HostState {
    pub quiet: bool,
    print_hash_only: bool,
    heap_ptr: Option<u64>,
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
            heap_ptr: None,
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

    linker.func_wrap(
        "env",
        "free",
        |_caller: Caller<'_, HostState>, _ptr: i32| {},
    )?;

    linker.func_wrap(
        "env",
        "malloc",
        |mut caller: Caller<'_, HostState>, size: i32| -> Result<i32> {
            if size < 0 {
                return Err(anyhow!("malloc size must be non-negative, got {}", size));
            }

            let bytes = size as u64;
            let memory = resolve_memory(&mut caller)?;
            let current_memory_end = memory.data_size(&caller) as u64;

            let next_ptr = caller
                .data()
                .heap_ptr
                .unwrap_or_else(|| align_up(current_memory_end, HEAP_ALIGNMENT));

            let ptr = align_up(next_ptr, HEAP_ALIGNMENT);
            let end = ptr
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("malloc overflow for size {}", size))?;

            ensure_capacity(&memory, &mut caller, end)?;
            caller.data_mut().heap_ptr = Some(end);

            if ptr > i32::MAX as u64 {
                return Err(anyhow!("allocated pointer out of i32 range: {}", ptr));
            }
            Ok(ptr as i32)
        },
    )?;

    Ok(())
}

fn resolve_memory(caller: &mut Caller<'_, HostState>) -> Result<Memory> {
    caller
        .get_export("memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| anyhow!("module does not export memory required by env.malloc"))
}

fn ensure_capacity(
    memory: &Memory,
    caller: &mut Caller<'_, HostState>,
    needed_end: u64,
) -> Result<()> {
    let current = memory.data_size(&*caller) as u64;
    if needed_end <= current {
        return Ok(());
    }

    let needed_bytes = needed_end - current;
    let pages = (needed_bytes + WASM_PAGE_SIZE - 1) / WASM_PAGE_SIZE;
    memory.grow(&mut *caller, pages)?;
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}
