use bytemuck;
use criterion::{criterion_group, criterion_main, Criterion};
use run_wasm_lib::mnist;
use std::fs;
use std::path::PathBuf;
use wasmtime::{Config, Engine, Instance, Linker, Memory, Module, OptLevel, Store, TypedFunc};

fn prepare_common(wasm_file: &str, optimize: bool) -> (Instance, Store<()>, Memory, i32) {
    let mut wasm_path = PathBuf::new();
    wasm_path.push("benches");
    wasm_path.push("wasm_files");
    wasm_path.push(wasm_file);

    let mut config = Config::new();
    config.cranelift_opt_level(if optimize {
        OptLevel::Speed
    } else {
        OptLevel::None
    });

    let engine = Engine::new(&config).unwrap();
    let wasm_bytes = fs::read(wasm_path).unwrap();

    // always do aot
    let module = {
        let compiled_wasm_bytes = engine.precompile_module(&wasm_bytes).unwrap();
        unsafe { Module::deserialize(&engine, &compiled_wasm_bytes).unwrap() }
    };

    let mut linker = Linker::new(&engine);

    // Optional logging functions for debugging
    linker
        .func_wrap("env", "log_i32", |x: i32| -> i32 {
            println!("log_i32: {}", x);
            x
        })
        .unwrap();
    linker
        .func_wrap("env", "log_f32", |x: f32| -> f32 {
            println!("log_f32: {}", x);
            x
        })
        .unwrap();

    // Prepare the input data
    let scaled_data: Vec<_> = mnist::SAMPLE_DATA
        .into_iter()
        .map(|x| (x as f32) / 255.0)
        .collect();

    // Create the instance and store outside the benchmark loop
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module).unwrap();
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory not found");

    let malloc_fn = instance
        .get_func(&mut store, "malloc")
        .expect("malloc not found")
        .typed::<i32, i32>(&store)
        .unwrap();

    // Allocate and initialize input tensor
    let tensor_ptr = malloc_fn
        .call(&mut store, mnist::INPUT_TENSOR_SIZE as i32)
        .unwrap();

    // Copy the input data to the tensor
    memory.data_mut(&mut store)
        [tensor_ptr as usize..tensor_ptr as usize + mnist::INPUT_TENSOR_SIZE]
        .copy_from_slice(bytemuck::cast_slice(&scaled_data));

    (instance, store, memory, tensor_ptr)
}

fn prepare_mlir(wasm_file: &str, optimize: bool) -> (TypedFunc<i32, i32>, Store<()>, i32) {
    let (instance, mut store, _, tensor_ptr) = prepare_common(wasm_file, optimize);

    let main_fn = instance
        .get_func(&mut store, "main")
        .expect("main not found")
        .typed::<i32, i32>(&store)
        .unwrap();

    return (main_fn, store, tensor_ptr);
}

fn prepare_llvm(
    wasm_file: &str,
    optimize: bool,
) -> (TypedFunc<(i32, i32), ()>, Store<()>, i32, i32) {
    let (instance, mut store, memory, tensor_ptr) = prepare_common(wasm_file, optimize);

    let main_fn = instance
        .get_func(&mut store, "_mlir_ciface_main")
        .expect("_mlir_ciface_main not found")
        .typed::<(i32, i32), ()>(&store)
        .unwrap();

    let malloc_fn = instance
        .get_func(&mut store, "malloc")
        .expect("malloc not found")
        .typed::<i32, i32>(&store)
        .unwrap();

    let (input_ptr, output_ptr) = {
        let input_ptr = malloc_fn
            .call(&mut store, mnist::INPUT_SIZE as i32)
            .unwrap();
        let input = mnist::Input {
            base_ptr: tensor_ptr,
            data: tensor_ptr,
            offset: 0,
            sizes: [1, 28, 28],
            strides: [28 * 28, 28, 1],
        };
        memory.data_mut(&mut store)[input_ptr as usize..input_ptr as usize + mnist::INPUT_SIZE]
            .copy_from_slice(bytemuck::bytes_of(&input));

        let output_tensor_ptr = malloc_fn
            .call(&mut store, mnist::OUTPUT_TENSOR_SIZE as i32)
            .unwrap();
        let output_ptr = malloc_fn
            .call(&mut store, mnist::OUTPUT_SIZE as i32)
            .unwrap();
        let output = mnist::Output {
            base_ptr: output_tensor_ptr,
            data: output_tensor_ptr,
            offset: 0,
            sizes: [1, 10],
            strides: [10, 1],
        };
        memory.data_mut(&mut store)[output_ptr as usize..output_ptr as usize + mnist::OUTPUT_SIZE]
            .copy_from_slice(bytemuck::bytes_of(&output));

        (input_ptr, output_ptr)
    };

    return (main_fn, store, input_ptr, output_ptr);
}

fn bench_conv2d_mlir_unopt(c: &mut Criterion) {
    let (main_fn, mut store, tensor_ptr) = prepare_mlir("conv2d-mlir.wasm", false);

    c.bench_function("conv2d_mlir_no_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, tensor_ptr).unwrap();
        })
    });
}

fn bench_conv2d_mlir_opt(c: &mut Criterion) {
    let (main_fn, mut store, tensor_ptr) = prepare_mlir("conv2d-mlir.wasm", true);

    c.bench_function("conv2d_mlir_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, tensor_ptr).unwrap();
        })
    });
}
fn bench_conv2d_unoptimized_llvm_unopt(c: &mut Criterion) {
    let (main_fn, mut store, input_ptr, output_ptr) =
        prepare_llvm("conv2d-unoptimized-llvm.wasm", false);

    c.bench_function("conv2d_unoptimized_llvm_no_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, (input_ptr, output_ptr)).unwrap();
        })
    });
}
fn bench_conv2d_unoptimized_llvm_opt(c: &mut Criterion) {
    let (main_fn, mut store, input_ptr, output_ptr) =
        prepare_llvm("conv2d-unoptimized-llvm.wasm", true);

    c.bench_function("conv2d_unoptimized_llvm_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, (input_ptr, output_ptr)).unwrap();
        })
    });
}

fn bench_conv2d_optimized_llvm_unopt(c: &mut Criterion) {
    let (main_fn, mut store, input_ptr, output_ptr) =
        prepare_llvm("conv2d-optimized-llvm.wasm", false);

    c.bench_function("conv2d_optimized_llvm_no_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, (input_ptr, output_ptr)).unwrap();
        })
    });
}
fn bench_conv2d_optimized_llvm_opt(c: &mut Criterion) {
    let (main_fn, mut store, input_ptr, output_ptr) =
        prepare_llvm("conv2d-optimized-llvm.wasm", true);

    c.bench_function("conv2d_optimized_llvm_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, (input_ptr, output_ptr)).unwrap();
        })
    });
}

fn bench_lenet_mlir_unopt(c: &mut Criterion) {
    let (main_fn, mut store, tensor_ptr) = prepare_mlir("lenet-mlir.wasm", false);

    c.bench_function("lenet_mlir_no_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, tensor_ptr).unwrap();
        })
    });
}

fn bench_lenet_mlir_opt(c: &mut Criterion) {
    let (main_fn, mut store, tensor_ptr) = prepare_mlir("lenet-mlir.wasm", true);

    c.bench_function("lenet_mlir_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, tensor_ptr).unwrap();
        })
    });
}
fn bench_lenet_unoptimized_llvm_unopt(c: &mut Criterion) {
    let (main_fn, mut store, input_ptr, output_ptr) =
        prepare_llvm("lenet-unoptimized-llvm.wasm", false);

    c.bench_function("lenet_unoptimized_llvm_no_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, (input_ptr, output_ptr)).unwrap();
        })
    });
}
fn bench_lenet_unoptimized_llvm_opt(c: &mut Criterion) {
    let (main_fn, mut store, input_ptr, output_ptr) =
        prepare_llvm("lenet-unoptimized-llvm.wasm", true);

    c.bench_function("lenet_unoptimized_llvm_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, (input_ptr, output_ptr)).unwrap();
        })
    });
}

fn bench_lenet_optimized_llvm_unopt(c: &mut Criterion) {
    let (main_fn, mut store, input_ptr, output_ptr) =
        prepare_llvm("lenet-optimized-llvm.wasm", false);

    c.bench_function("lenet_optimized_llvm_no_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, (input_ptr, output_ptr)).unwrap();
        })
    });
}
fn bench_lenet_optimized_llvm_opt(c: &mut Criterion) {
    let (main_fn, mut store, input_ptr, output_ptr) =
        prepare_llvm("lenet-optimized-llvm.wasm", true);

    c.bench_function("lenet_optimized_llvm_cranelift_opt", |b| {
        b.iter(|| {
            // Reset any mutable state here if necessary
            main_fn.call(&mut store, (input_ptr, output_ptr)).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_conv2d_mlir_unopt,
    bench_conv2d_mlir_opt,
    bench_conv2d_unoptimized_llvm_unopt,
    bench_conv2d_unoptimized_llvm_opt,
    bench_conv2d_optimized_llvm_unopt,
    bench_conv2d_optimized_llvm_opt,
    bench_lenet_mlir_unopt,
    bench_lenet_mlir_opt,
    bench_lenet_unoptimized_llvm_unopt,
    bench_lenet_unoptimized_llvm_opt,
    bench_lenet_optimized_llvm_unopt,
    bench_lenet_optimized_llvm_opt,
);
criterion_main!(benches);
