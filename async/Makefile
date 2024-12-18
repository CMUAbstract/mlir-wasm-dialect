# Makefile for this simple example
# Adjust these paths and variables based on your environment
WASI_SDK_PATH ?= /path/to/wasi-sdk
MLIR_OPT      ?= mlir-opt
MLIR_TRANSLATE?= mlir-translate
OPT           ?= opt
LLC           ?= llc
CLANG         ?= $(WASI_SDK_PATH)/bin/clang
WASM_LD       ?= $(WASI_SDK_PATH)/bin/wasm-ld
SYSROOT       ?= $(WASI_SDK_PATH)/share/wasi-sysroot
CFLAGS        ?= -O3 --target=wasm32-wasi --sysroot=$(SYSROOT)

# Input files
EXAMPLE_MLIR  = example.mlir
ASYNC_RUNTIME_C = async_runtime.c

# Intermediate files
EXAMPLE_MANUALEDIT_MLIR = example-manualedit.mlir
EXAMPLE_LL_MLIR         = example-ll.mlir
EXAMPLE_FIXED_LL_MLIR   = example-ll-fixed.mlir
EXAMPLE_LL              = example.ll
EXAMPLE_BC              = example-lowered.bc
EXAMPLE_O               = example.o
ASYNC_RUNTIME_O         = async_runtime.o
WASM_OUTPUT             = example.wasm

.PHONY: all clean

all: $(WASM_OUTPUT)

# Step 1: initial mlir-opt run and manual modification
$(EXAMPLE_MANUALEDIT_MLIR): $(EXAMPLE_MLIR)
	$(MLIR_OPT) --async-func-to-async-runtime --async-to-async-runtime \
	--convert-to-llvm $< -o $@.tmp
	# Manual modification: remove lines 2-4 and lines 64-68
	# Adjust line numbers if necessary.
	sed '2,4d;64,67d' $@.tmp > $@
	rm $@.tmp

# Step 2: second mlir-opt run
$(EXAMPLE_LL_MLIR): $(EXAMPLE_MANUALEDIT_MLIR)
	$(MLIR_OPT) --convert-async-to-llvm \
	--async-func-to-async-runtime \
	--reconcile-unrealized-casts \
	--convert-to-llvm \
	--canonicalize \
	$< -o $@.tmp
	# Manual modification: change all i64 to i32
	sed 's/i64/i32/g' $@.tmp > $@
	rm $@.tmp

# Step 3: mlir-translate to LLVM IR
$(EXAMPLE_LL): $(EXAMPLE_LL_MLIR)
	$(MLIR_TRANSLATE) $< --mlir-to-llvmir -o $@

# Step 4: opt passes
$(EXAMPLE_BC): $(EXAMPLE_LL)
	$(OPT) --passes="coro-early,function(coro-elide),coro-split,coro-cleanup" $< -o $@

# Step 5: llc to compile to object file
$(EXAMPLE_O): $(EXAMPLE_BC)
	$(LLC) -O0 -filetype=obj -mtriple=wasm32-wasi $< -o $@

# Step 6: compile async runtime
$(ASYNC_RUNTIME_O): $(ASYNC_RUNTIME_C)
	$(CLANG) $(CFLAGS) -c $< -o $@

# Step 7: link object files
$(WASM_OUTPUT): $(EXAMPLE_O) $(ASYNC_RUNTIME_O)
	$(WASM_LD) \
	  $^ \
	  --no-entry \
	  --allow-undefined \
	  --export=main \
	  $(SYSROOT)/lib/wasm32-wasi/libc.a \
	  -o $@

clean:
	rm -f $(EXAMPLE_MANUALEDIT_MLIR) $(EXAMPLE_LL_MLIR) $(EXAMPLE_LL) $(EXAMPLE_BC) $(EXAMPLE_O) $(ASYNC_RUNTIME_O) $(WASM_OUTPUT)