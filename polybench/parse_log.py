#! /usr/bin/env python3
import json
import re
import sys

def parse_data(filename):
    with open(filename, 'r') as file:
        file_content = file.read()
    
    lines = file_content.strip().split('\n')
    data = []
    
    for line in lines:
        try:
            entry = json.loads(line)
            # Extract the key fields
            tag = entry.get('tag')
            compiler = entry.get('compiler')
            use_aot = entry.get('use_aot', False)
            binaryen_opt_level = entry.get('binaryen_opt_level')
            
            # Extract execution time from stdout using regex
            time_match = re.search(r'\[execution time\] ([\d.]+) miliseconds', entry.get('stdout', ''))
            exec_time = float(time_match.group(1)) if time_match else None
            
            data.append({
                'tag': tag,
                'compiler': compiler,
                'use_aot': use_aot,
                'binaryen_opt_level': binaryen_opt_level,
                'exec_time': exec_time
            })
        except (json.JSONDecodeError, AttributeError):
            continue
    
    return data

def generate_table(data):
    # Get unique values for table dimensions
    tags = sorted(set(item['tag'] for item in data))
    compilers = sorted(set(item['compiler'] for item in data))
    
    # Create header row
    table = [['Benchmark', 'Compiler', 'Interpreter', '', '', 'AoT', '', '']]
    table.append(['', '', 'binaryen O0', 'binaryen O2', 'binaryen O4', 'binaryen O0', 'binaryen O2', 'binaryen O4'])
    
    # Fill the data
    for tag in tags:
        for compiler in compilers:
            row = [tag, compiler]
            
            # Fill interpreter columns (use_aot=False)
            for opt in [0, 2, 4]:
                found = False
                for item in data:
                    if (item['tag'] == tag and 
                        item['compiler'] == compiler and 
                        item['use_aot'] == False and 
                        item['binaryen_opt_level'] == opt):
                        row.append(f"{item['exec_time']:.3f}")
                        found = True
                        break
                if not found:
                    row.append('')
            
            # Fill AoT columns (use_aot=True)
            for opt in [0, 2, 4]:
                found = False
                for item in data:
                    if (item['tag'] == tag and 
                        item['compiler'] == compiler and 
                        item['use_aot'] == True and 
                        item['binaryen_opt_level'] == opt):
                        row.append(f"{item['exec_time']:.3f}")
                        found = True
                        break
                if not found:
                    row.append('')
            
            table.append(row)
    
    return table

def format_for_google_sheets(table):
    return '\n'.join(['\t'.join(str(cell) for cell in row) for row in table])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py filename")
        sys.exit(1)
        
    filename = sys.argv[1]
    try:
        data = parse_data(filename)
        table = generate_table(data)
        google_sheets_format = format_for_google_sheets(table)
        print(google_sheets_format)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)