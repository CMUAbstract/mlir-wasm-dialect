#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { performance } from 'node:perf_hooks';

const FNV1A64_PRIME = 1099511628211n;
const U64_MASK = 0xFFFFFFFFFFFFFFFFn;

const { values: args } = parseArgs({
  options: {
    input:               { type: 'string' },
    entry:               { type: 'string', default: 'main' },
    iterations:          { type: 'string', default: '1' },
    warmup:              { type: 'string', default: '0' },
    quiet:               { type: 'boolean', default: false },
    json:                { type: 'boolean', default: false },
    'print-mode':        { type: 'string', default: 'normal' },
    'print-hash-seed':   { type: 'string', default: '14695981039346656037' },
    'expect-i32':        { type: 'string' },
  },
});

const inputPath = args.input;
if (!inputPath) {
  process.stderr.write('Error: --input is required\n');
  process.exit(1);
}

const entryName = args.entry;
const iterations = parseInt(args.iterations, 10);
const warmupCount = parseInt(args.warmup, 10);
const quiet = args.quiet;
const jsonOutput = args.json;
const printMode = args['print-mode'];
const printHashSeed = BigInt(args['print-hash-seed']);
const expectI32 = args['expect-i32'] != null ? parseInt(args['expect-i32'], 10) : null;

if (iterations <= 0) {
  process.stderr.write('Error: --iterations must be greater than zero\n');
  process.exit(1);
}

const wasmBuffer = await readFile(inputPath);

const totalIterations = warmupCount + iterations;
const durations = [];
let measuredReturnVal = 0;
let measuredPrintCount = 0n;
let measuredPrintHash = 0n;

for (let iter = 0; iter < totalIterations; iter++) {
  // Per-iteration state (matches Rust: new store per iteration)
  let printCount = 0n;
  let printHash = printHashSeed;
  let toggleStartedAt = null;
  let lastToggleDuration = null;

  const importObject = {
    env: {
      toggle_gpio() {
        if (toggleStartedAt !== null) {
          lastToggleDuration = performance.now() - toggleStartedAt;
          toggleStartedAt = null;
        } else {
          toggleStartedAt = performance.now();
        }
      },
      print_i32(value) {
        printCount += 1n;
        const buf = new ArrayBuffer(4);
        new DataView(buf).setInt32(0, value, true); // little-endian
        const bytes = new Uint8Array(buf);
        for (const byte of bytes) {
          printHash ^= BigInt(byte);
          printHash = (printHash * FNV1A64_PRIME) & U64_MASK;
        }
        if (printMode !== 'hash' && !quiet) {
          console.log(`print_i32: ${value}`);
        }
      },
    },
  };

  const { instance } = await WebAssembly.instantiate(wasmBuffer, importObject);

  const fn = instance.exports[entryName];
  if (typeof fn !== 'function') {
    process.stderr.write(`Error: entry function '${entryName}' was not found\n`);
    process.exit(3);
  }

  // JS Wasm bindings default missing numeric args to 0, so fn() works for both
  // () -> i32 and (i32, i32) -> i32 signatures.
  const started = performance.now();
  const returnVal = fn();
  const wallElapsed = performance.now() - started;

  const elapsed = lastToggleDuration !== null ? lastToggleDuration : wallElapsed;

  if (iter >= warmupCount) {
    measuredReturnVal = returnVal;
    measuredPrintCount = printCount;
    measuredPrintHash = printHash;
    durations.push(elapsed);

    if (expectI32 !== null && returnVal !== expectI32) {
      process.stderr.write(
        `ERROR: result mismatch at iteration ${iter - warmupCount}: expected ${expectI32}, got ${returnVal}\n`
      );
      process.exit(6);
    }
  }
}

// Compute statistics
function summarizeMs(samples) {
  if (samples.length === 0) return { avg: 0, min: 0, max: 0, stddev: 0 };
  let min = samples[0];
  let max = samples[0];
  let total = 0;
  for (const s of samples) {
    if (s < min) min = s;
    if (s > max) max = s;
    total += s;
  }
  const avg = total / samples.length;
  const variance = samples.reduce((acc, s) => acc + (s - avg) ** 2, 0) / samples.length;
  return { avg, min, max, stddev: Math.sqrt(variance) };
}

const { avg, min, max, stddev } = summarizeMs(durations);
const pass = expectI32 !== null ? expectI32 === measuredReturnVal : true;
const expectedStr = expectI32 !== null ? String(expectI32) : 'none';
const hashStr = '0x' + measuredPrintHash.toString(16).padStart(16, '0');

if (jsonOutput) {
  const expectedJson = expectI32 !== null ? String(expectI32) : 'null';
  console.log(
    `{"pass":${pass},"expected":${expectedJson},"return_val":${measuredReturnVal},"iterations":${iterations},"warmup":${warmupCount},"ms_avg":${avg.toFixed(6)},"ms_min":${min.toFixed(6)},"ms_max":${max.toFixed(6)},"print_count":${measuredPrintCount},"print_hash":"${hashStr}"}`
  );
} else if (!quiet) {
  console.log(
    `RESULT status=${pass ? 'PASS' : 'FAIL'} expected=${expectedStr} return_val=${measuredReturnVal} iterations=${iterations} warmup=${warmupCount} ms_avg=${avg.toFixed(6)} ms_min=${min.toFixed(6)} ms_max=${max.toFixed(6)} print_count=${measuredPrintCount} print_hash=${hashStr}`
  );
}

console.log(`[execution time] ${avg.toFixed(3)} miliseconds`);
console.log(`[iterations] ${iterations}`);
console.log(`[warmup] ${warmupCount}`);
console.log(`[min] ${min.toFixed(3)} miliseconds`);
console.log(`[max] ${max.toFixed(3)} miliseconds`);
console.log(`[stddev] ${stddev.toFixed(3)} miliseconds`);
