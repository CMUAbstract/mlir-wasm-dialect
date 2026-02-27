#! /usr/bin/env python3

import os
import re


def remove_attributes(content):
    # Remove module attributes
    content = re.sub(r"module attributes {[^}]*}", "module", content)

    # Remove func attributes while preserving the function signature
    content = re.sub(r"(func\.func.*?) attributes {[^}]*}", r"\1", content)

    return content


def process_directory():
    # Get all .mlir files in the polybench directory
    for filename in os.listdir("."):
        if filename.endswith(".mlir"):
            filepath = os.path.join(".", filename)

            # Read file content
            with open(filepath, "r") as f:
                content = f.read()

            # Remove attributes
            modified_content = remove_attributes(content)

            # Write back to file
            with open(filepath, "w") as f:
                f.write(modified_content)

            print(f"Processed: {filename}")


if __name__ == "__main__":
    process_directory()
