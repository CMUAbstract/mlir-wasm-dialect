#! /usr/bin/env python3
import argparse
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_data_from_file(filename):
    """Parse the JSON data from the file and extract execution time information."""
    data = []
    
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Extract data from each line (JSON object)
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    
                    # Extract relevant information
                    benchmark = entry.get('tag')
                    compiler = entry.get('compiler')
                    use_aot = entry.get('use_aot', False)
                    binaryen_opt_level = entry.get('binaryen_opt_level')
                    
                    # Extract execution time from stdout
                    stdout = entry.get('stdout', '')
                    match = re.search(r'\[execution time\] (\d+\.\d+) miliseconds', stdout)
                    if match:
                        execution_time = float(match.group(1))
                        
                        data.append({
                            'benchmark': benchmark,
                            'compiler': compiler,
                            'use_aot': use_aot,
                            'binaryen_opt_level': binaryen_opt_level,
                            'execution_time': execution_time
                        })
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"Error reading file {filename}: {e}", file=sys.stderr)
        sys.exit(1)
    
    return data

def filter_and_prepare_data(data, use_aot, binaryen_opt_level):
    """Filter the data based on the specified parameters and prepare for plotting."""
    # Convert binaryen_opt_level to integer
    try:
        binaryen_opt_level = int(binaryen_opt_level)
    except ValueError:
        print(f"Error: binaryen_opt_level must be an integer, got: {binaryen_opt_level}", file=sys.stderr)
        sys.exit(1)
    
    # Filter data based on use_aot and binaryen_opt_level
    filtered_data = [d for d in data if d['use_aot'] == use_aot and d['binaryen_opt_level'] == binaryen_opt_level]
    
    # Organize data by benchmark and compiler
    result = {}
    
    for entry in filtered_data:
        benchmark = entry['benchmark']
        compiler = entry['compiler']
        execution_time = entry['execution_time']
        
        if benchmark not in result:
            result[benchmark] = {'llvm': None, 'mlir': None}
        
        result[benchmark][compiler] = execution_time
    
    # Filter out benchmarks without both LLVM and MLIR data
    result = {k: v for k, v in result.items() if v['llvm'] is not None and v['mlir'] is not None}
    
    # Calculate speedup for each benchmark
    for benchmark, values in result.items():
        values['speedup'] = values['llvm'] / values['mlir']
    
    return result

def plot_data(data, use_aot, binaryen_opt_level, output_file=None, show_speedup=True, normalize=False, only_speedup=False):
    """Create bar plots for execution time and speedup."""
    # Convert data to lists for plotting
    benchmarks = list(data.keys())
    llvm_times = [data[b]['llvm'] for b in benchmarks]
    mlir_times = [data[b]['mlir'] for b in benchmarks]
    speedups = [data[b]['speedup'] for b in benchmarks]
    
    # Sort data by benchmark name
    sorted_indices = np.argsort(benchmarks)
    benchmarks = [benchmarks[i] for i in sorted_indices]
    llvm_times = [llvm_times[i] for i in sorted_indices]
    mlir_times = [mlir_times[i] for i in sorted_indices]
    speedups = [speedups[i] for i in sorted_indices]
    
    # Calculate geometric mean speedup
    # Use numpy.prod for multiplication and then take the nth root
    # Add a small epsilon to avoid issues with zero values
    epsilon = 1e-10
    geo_mean_speedup = np.prod(np.array(speedups) + epsilon) ** (1.0 / len(speedups)) - epsilon
    mlir_wins = len([s for s in speedups if s > 1])
    llvm_wins = len([s for s in speedups if s < 1])
    
    # Create figure - one or two subplots depending on show_speedup and only_speedup
    if only_speedup:
        fig, ax2 = plt.subplots(figsize=(15, 6))
    elif show_speedup:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(15, 6))
    
    # Normalize data if requested
    if normalize:
        # Normalize with respect to LLVM (set LLVM execution time as 1.0)
        norm_llvm_times = [1.0 for _ in range(len(benchmarks))]  # All LLVM times become 1.0
        norm_mlir_times = [mlir_times[i] / llvm_times[i] for i in range(len(benchmarks))]  # MLIR times relative to LLVM
        # Use normalized times
        plot_llvm_times = norm_llvm_times
        plot_mlir_times = norm_mlir_times
        y_label = 'Normalized Execution Time (relative to LLVM)'
    else:
        # Use raw times
        plot_llvm_times = llvm_times
        plot_mlir_times = mlir_times
        y_label = 'Execution Time (ms)'
    
    # Plot execution times on the first subplot (if not only showing speedup)
    if not only_speedup:
        x = np.arange(len(benchmarks))
        width = 0.35
        
        ax1.bar(x - width/2, plot_llvm_times, width, label='LLVM', color='blue', alpha=0.7)
        ax1.bar(x + width/2, plot_mlir_times, width, label='MLIR', color='green', alpha=0.7)
        
        ax1.set_ylabel(y_label)
        execution_mode = "Interpreter" if not use_aot else "AOT"
        ax1.set_title(f'LLVM vs MLIR Performance Comparison - {execution_mode} (Binaryen O{binaryen_opt_level})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax1.legend()
    
    # Plot speedup on the second subplot if requested
    if show_speedup or only_speedup:
        x = np.arange(len(benchmarks))
        width = 0.35
        
        # Green for MLIR better (speedup > 1), red for LLVM better (speedup < 1)
        colors = ['green' if s > 1 else 'red' for s in speedups]
        ax2.bar(x, speedups, width, color=colors)
        ax2.axhline(y=1, color='black', linestyle='--')
        
        ax2.set_ylabel('Speedup (LLVM/MLIR)')
        execution_mode = "Interpreter" if not use_aot else "AOT"
        if only_speedup:
            ax2.set_title(f'Speedup Ratio: LLVM / MLIR - {execution_mode} (Binaryen O{binaryen_opt_level})')
        else:
            ax2.set_title(f'Speedup Ratio: LLVM / MLIR (> 1 means MLIR is better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(benchmarks, rotation=45, ha='right')
        
        # Prepare summary statistics text
        summary_text = (f"Geometric Mean Speedup (LLVM/MLIR): {geo_mean_speedup:.3f}\n"
                       f"Benchmarks where MLIR outperforms LLVM: {mlir_wins} out of {len(speedups)}\n"
                       f"Benchmarks where LLVM outperforms MLIR: {llvm_wins} out of {len(speedups)}")
        
        # Print summary to stdout
        print("\nSummary Statistics:")
        print(summary_text)
        
        plt.tight_layout()
    else:
        plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Always show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot LLVM vs MLIR performance comparison')
    parser.add_argument('filename', help='Input data file')
    parser.add_argument('--aot', dest='use_aot', action='store_true', help='Use AOT mode (default: interpreter mode)')
    parser.add_argument('--binaryen-opt-level', type=int, choices=[0, 2, 4], default=0,
                       help='Binaryen optimization level (0, 2, or 4, default: 0)')
    parser.add_argument('--normalize', action='store_true', help='Normalize execution times')
    parser.add_argument('--no-speedup', dest='show_speedup', action='store_false', 
                        help='Disable speedup plot')
    parser.add_argument('--only-speedup', action='store_true',
                        help='Show only the speedup plot')
    parser.add_argument('-o', '--output', help='Output file for the plot (optional)')
    
    # Set defaults for new arguments
    parser.set_defaults(show_speedup=True)
    
    args = parser.parse_args()
    
    # Parse data from the file
    data = parse_data_from_file(args.filename)

    print("data")
    print(data)
    
    # Filter and prepare data based on the specified parameters
    filtered_data = filter_and_prepare_data(data, args.use_aot, args.binaryen_opt_level)

    # Plot the data
    plot_data(filtered_data, args.use_aot, args.binaryen_opt_level, args.output, 
              args.show_speedup, args.normalize, args.only_speedup)

if __name__ == "__main__":
    main()