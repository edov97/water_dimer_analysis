#!/usr/bin/env python3
"""
Generate Psi4 input files for SAPT(0) calculations for water dimers.
Updated to extract δHF and calculate supramolecular HF energies.
"""

import os
from systems_H2O_H2O import SYSTEMS

def fix_geometry(geom_str):
    """Process the geometry string from SYSTEMS."""
    lines = geom_str.strip().splitlines()
    fixed_lines = [line for line in lines if not line.lstrip().startswith("#")]
    return "\n".join(fixed_lines)

def generate_sapt0_input(geom_str, system_key, basis_sapt, output_folder):
    """Generate a Psi4 input file for a SAPT(0) calculation with δHF extraction."""
    os.makedirs(output_folder, exist_ok=True)
    
    molecule_block = fix_geometry(geom_str)
    
    input_text = f"""import psi4

# Set memory and output file
psi4.set_memory('8 GB')
psi4.core.set_output_file('{system_key}_SAPT0_{basis_sapt.replace('-', '_')}_output.dat', False)

# Set options for SAPT0 calculation
psi4.set_options({{
    'basis': '{basis_sapt}',
    'scf_type': 'df',
    'freeze_core': True,
    'e_convergence': 1e-9,
    'd_convergence': 1e-8,
    'maxiter': 100,
    'print': 2,
}})

# Define the dimer molecule
molecule = psi4.geometry(\"\"\"\n{molecule_block}\"\"\")

# Compute SAPT0 energy decomposition
E_sapt = psi4.energy('sapt0')

# Extract individual energy components
E_elst = psi4.variable('SAPT ELST ENERGY')
E_exch = psi4.variable('SAPT EXCH ENERGY') 
E_ind = psi4.variable('SAPT IND ENERGY')
E_disp = psi4.variable('SAPT DISP ENERGY')
E_total_sapt0 = psi4.variable('SAPT0 TOTAL ENERGY')
E_hf_total = psi4.variable('SAPT HF TOTAL ENERGY')

# Extract δHF (HF correction)
try:
    E_hf_total = psi4.variable('SAPT HF TOTAL ENERGY')
    delta_hf = E_hf_total - (E_elst + E_exch + E_ind)
except:
    delta_hf = 0.0
    print("Warning: Could not extract delta HF")

# Calculate supramolecular HF interaction energy
E_supramolecular_hf = E_hf_total

# Convert to common units
hartree_to_kj = 2625.4996394798254

# Print results
print(f"SAPT(0) Results for {system_key} ({basis_sapt}):")
print(f"  E_electrostatic: {{E_elst:.12f}} Hartree")
print(f"  E_exchange: {{E_exch:.12f}} Hartree")
print(f"  E_induction: {{E_ind:.12f}} Hartree")
print(f"  E_dispersion: {{E_disp:.12f}} Hartree")
print(f"  E_total_sapt0: {{E_total_sapt0:.12f}} Hartree")
print(f"  delta_hf: {{delta_hf:.12f}} Hartree")
print(f"  E_supramolecular_hf: {{E_supramolecular_hf:.12f}} Hartree")

# Save results to file
with open('{system_key}_sapt0_{basis_sapt.replace('-', '_')}_results.txt', 'w') as f:
    f.write(f"System = {system_key}\\n")
    f.write(f"Basis = {basis_sapt}\\n")
    f.write(f"E_electrostatic = {{E_elst:.12f}}\\n")
    f.write(f"E_exchange = {{E_exch:.12f}}\\n")
    f.write(f"E_induction = {{E_ind:.12f}}\\n")
    f.write(f"E_dispersion = {{E_disp:.12f}}\\n")
    f.write(f"E_total_sapt0 = {{E_total_sapt0:.12f}}\\n")
    f.write(f"delta_hf = {{delta_hf:.12f}}\\n")
    f.write(f"E_supramolecular_hf = {{E_supramolecular_hf:.12f}}\\n")

print(f"Results saved to {system_key}_sapt0_{basis_sapt.replace('-', '_')}_results.txt")
"""
    
    output_file = os.path.join(output_folder, f"{system_key}_SAPT0_{basis_sapt.replace('-', '_')}.py")
    with open(output_file, "w") as f:
        f.write(input_text)
    
    return output_file

def create_results_processor():
    """Create a script to process SAPT(0) results and create CSV files."""
    processor_content = '''#!/usr/bin/env python3
"""Process SAPT(0) results and create CSV files."""

import pandas as pd
import numpy as np
import os
import re

def extract_distance_from_system(system_name):
    """Extract distance from system name."""
    match = re.search(r'_(\\d+)$', system_name)
    if match:
        system_num = int(match.group(1))
        distance_map = {
            900: 4.157406, 901: 4.346379, 902: 4.535352, 903: 4.724325,
            904: 5.102270, 905: 5.291243, 906: 5.480216, 907: 5.655189,
            908: 5.858162, 909: 6.047135, 910: 6.236108, 911: 6.614054,
            912: 7.370000, 913: 9.225000, 914: 11.080000
        }
        return distance_map.get(system_num, None)
    return None

def parse_results_file(filepath):
    """Parse a SAPT(0) results file."""
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip()
            try:
                results[key] = float(value)
            except ValueError:
                results[key] = value
    return results

def process_basis_set(basis_name):
    """Process results for a specific basis set."""
    print(f"Processing {basis_name} basis set...")
    
    systems_list = [
        'H2O..H2O_900', 'H2O..H2O_901', 'H2O..H2O_902', 'H2O..H2O_903',
        'H2O..H2O_904', 'H2O..H2O_905', 'H2O..H2O_906', 'H2O..H2O_907',
        'H2O..H2O_908', 'H2O..H2O_909', 'H2O..H2O_910', 'H2O..H2O_911',
        'H2O..H2O_912', 'H2O..H2O_913', 'H2O..H2O_914'
    ]
    
    data_rows = []
    
    for system in systems_list:
        results_file = f"{system}_sapt0_{basis_name.replace('-', '_')}_results.txt"
        
        if os.path.exists(results_file):
            try:
                results = parse_results_file(results_file)
                distance = extract_distance_from_system(system)
                
                if distance is not None:
                    row = {
                        'System': system,
                        'Distance': distance,
                        'E_electrostatic': results.get('E_electrostatic', np.nan),
                        'E_exchange': results.get('E_exchange', np.nan),
                        'E_induction': results.get('E_induction', np.nan),
                        'E_dispersion': results.get('E_dispersion', np.nan),
                        'E_total_sapt0': results.get('E_total_sapt0', np.nan),
                        'delta_hf': results.get('delta_hf', np.nan),
                        'E_supramolecular_hf': results.get('E_supramolecular_hf', np.nan)
                    }
                    data_rows.append(row)
                    print(f"  Processed {system} at {distance:.3f} Å")
            except Exception as e:
                print(f"  Error processing {system}: {e}")
        else:
            print(f"  Warning: Results file not found for {system}")
    
    if data_rows:
        df = pd.DataFrame(data_rows)
        df = df.sort_values('Distance').reset_index(drop=True)
        
        csv_filename = f"sapt0_supramolecular_hf_results_{basis_name.replace('-', '_')}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"  Saved {len(df)} results to {csv_filename}")
        return df
    else:
        print(f"  No valid results found for {basis_name}")
        return None

def main():
    """Main processing function."""
    print("Processing SAPT(0) and Supramolecular HF results...")
    print("="*60)
    
    basis_sets = ['6-31g', 'aug-cc-pVTZ']
    results = {}
    
    for basis in basis_sets:
        df = process_basis_set(basis)
        if df is not None:
            results[basis] = df
    
    print("\\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    for basis, df in results.items():
        if df is not None:
            print(f"{basis}: {len(df)} systems processed")
            print(f"  Distance range: {df['Distance'].min():.3f} - {df['Distance'].max():.3f} Å")
            print(f"  E_int(SAPT0) range: {df['E_total_sapt0'].min():.6f} - {df['E_total_sapt0'].max():.6f} Hartree")
            print(f"  E_int(HF) range: {df['E_supramolecular_hf'].min():.6f} - {df['E_supramolecular_hf'].max():.6f} Hartree")
            print()
    
    print("CSV files created:")
    for basis in basis_sets:
        csv_file = f"sapt0_supramolecular_hf_results_{basis.replace('-', '_')}.csv"
        if os.path.exists(csv_file):
            print(f"  {csv_file}")

if __name__ == "__main__":
    main()
'''
    
    with open("sapt0_inputs/process_sapt0_results.py", "w") as f:
        f.write(processor_content)
    os.chmod("sapt0_inputs/process_sapt0_results.py", 0o755)

def main():
    """Main function to generate all SAPT(0) input files."""
    sapt0_output_dir = "sapt0_inputs"
    basis_sets = ["6-31g", "aug-cc-pVTZ"]
    
    print(f"Generating SAPT(0) input files for multiple basis sets...")
    print(f"Basis sets: {', '.join(basis_sets)}")
    print(f"Output directory: {sapt0_output_dir}")
    print(f"Number of systems: {len(SYSTEMS)}")
    print("")
    
    systems_list = []
    
    # Generate input files for each basis set
    for basis_sapt in basis_sets:
        print(f"--- Generating inputs for {basis_sapt} ---")
        
        for system_key, geom_str in SYSTEMS.items():
            try:
                sapt0_file = generate_sapt0_input(
                    geom_str, system_key, basis_sapt, sapt0_output_dir
                )
                
                if system_key not in systems_list:
                    systems_list.append(system_key)
                
                print(f"  Generated: {os.path.basename(sapt0_file)}")
                
            except Exception as e:
                print(f"  Error processing {system_key}: {e}")
                continue
        print()
    
    # Create results processor
    create_results_processor()
    print("Generated results processor: process_sapt0_results.py")
    
    # Create batch script
    batch_content = f"""#!/bin/bash
echo "Running SAPT(0) calculations for all systems and basis sets..."

# Run all calculations
for basis in 6_31g aug_cc_pVTZ; do
    echo "Processing basis set: $basis"
    for system in {' '.join(systems_list)}; do
        echo "  Running $system with $basis"
        python3 ${{system}}_SAPT0_${{basis}}.py
    done
done

echo "Processing results..."
python3 process_sapt0_results.py

echo "All calculations completed!"
"""
    
    with open(f"{sapt0_output_dir}/run_all_sapt0.sh", "w") as f:
        f.write(batch_content)
    os.chmod(f"{sapt0_output_dir}/run_all_sapt0.sh", 0o755)
    
    print("Generated batch script: run_all_sapt0.sh")
    print("")
    
    print("="*70)
    print("USAGE INSTRUCTIONS:")
    print("="*70)
    print(f"1. Navigate to: cd {sapt0_output_dir}")
    print("2. Run all calculations: ./run_all_sapt0.sh")
    print("3. CSV files will be created:")
    for basis in basis_sets:
        print(f"   sapt0_supramolecular_hf_results_{basis.replace('-', '_')}.csv")
    print("4. Use CSV files in Jupyter notebook for comparison with SART")
    print("="*70)

if __name__ == "__main__":
    main()

