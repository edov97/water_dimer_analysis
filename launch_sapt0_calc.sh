#!/bin/bash

# Directory containing the input files (for SAPT0)
input_directory="/home/evanich/phymol-dc1/water_dimer_analysis/sapt0_inputs"
# Directory where the output files will be saved
output_directory="/home/evanich/phymol-dc1/water_dimer_analysis/sapt0_outputs"

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# List all input files (for SAPT0)
input_files=$(find "$input_directory" -name "*.in")

# Loop through each input file and run psi4
for input_file in $input_files; do
    base_name=$(basename "$input_file" .in)
    output_file="${output_directory}/${base_name}.out"
    psi4 -n 8 -i "$input_file" -o "$output_file" &
done

# Wait for all background processes to finish
wait
