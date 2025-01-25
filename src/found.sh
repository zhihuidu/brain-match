#!/bin/bash

# Step 1: List all output files, sorted by modification time (newest first)
output_files=$(ls -t *.out |head -n 50)

# Step 2: Check if there are any output files
if [[ -z "$output_files" ]]; then
    echo "No output files found."
    exit 1
fi

# Step 3: Search for the keyword "found" in each file
for out_file in $output_files; do
    echo "Checking file: $out_file"
    grep -H "found" "$out_file" 
done

