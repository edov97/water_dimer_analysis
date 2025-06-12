#!/usr/bin/env python3
"""
Generate Psi4 input files for SAPT(0) and supramolecular HF for water dimers.

This script uses the geometry strings from systems_H2O_H2O.txt (imported as SYSTEMS)
and produces input files without relying on AaronTools geometry parsing.
"""

import os
from systems_H2O_H2O import SYSTEMS

def fix_geometry(geom_str):
    """
    Process the geometry string from SYSTEMS.
    
    Removes the initial comment lines (those starting with '#')
    and returns a string that can be directly inserted in a Psi4 molecule block.
    """
    lines = geom_str.strip().splitlines()
    # Remove any lines that start with '#' (i.e. the metadata comment)
    fixed_lines = [line for line in lines if not line.lstrip().startswith("#")]
    # Return the joined lines. Depending on your needs you might also reformat.
    return "\n".join(fixed_lines)

def generate_sapt0_input(geom_str, system_key, basis_sapt, output_folder):
    """
    Generate a Psi4 input file for a SAPT(0) calculation.
    
    Parameters:
      geom_str   : str
          Raw geometry string from SYSTEMS.
      system_key : str
          Identifier for the system (e.g. "H2O..H2O_900").
      basis_sapt : str
          The basis set for SAPT0 calculations (e.g., "jun-cc-pVDZ").
      output_folder : str
          Directory in which to save the input file.
    
    Returns:
      output_file : str
          Full path to the generated Psi4 input file.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Process the geometry string to remove comment lines.
    molecule_block = fix_geometry(geom_str)
    
    # Create a Psi4 input script as a text string.
    # You can adjust the options below as needed.
    input_text = f"""import psi4

# Set memory and output file.
psi4.set_memory('500 MB')
psi4.core.set_output_file('{system_key}_SAPT0_output.dat', False)

# Set options for SAPT0:
psi4.set_options({{
    'basis': '{basis_sapt}',
    'scf_type': 'df',
    'freeze_core': True,
    'e_convergence': 1e-8,
    'd_convergence': 1e-8,
    'maxiter': 50,
    'PRINT': 1
}})

# Define the molecule
molecule {system_key} {{
{molecule_block}
}}

# Compute SAPT0 energy. 
# (Note: Depending on your Psi4 version and installed SAPT module,
#  SAPT0 might be invoked via the 'energy' command.)
energy('SAPT0')
"""
    # Construct output filename.
    output_file = os.path.join(output_folder, f"{system_key}_SAPT0_{basis_sapt}.in")
    with open(output_file, "w") as f:
        f.write(input_text)
    
    return output_file

def generate_hf_input(geom_str, system_key, basis_hf, output_folder):
    """
    Generate a Psi4 input file for a supramolecular HF calculation.
    
    Parameters:
      geom_str    : str
          Raw geometry string from SYSTEMS.
      system_key  : str
          Identifier for the system.
      basis_hf    : str
          The basis set for the HF calculation (e.g., 'aug-cc-pVTZ').
      output_folder : str
          Directory where the input file will be saved.
    
    Returns:
      output_file : str
          Full path to the generated Psi4 input file.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    molecule_block = fix_geometry(geom_str)
    
    input_text = f"""import psi4

# Set memory and output.
psi4.set_memory('500 MB')
psi4.core.set_output_file('{system_key}_HF_output.dat', False)

# Set options for HF:
psi4.set_options({{
    'basis': '{basis_hf}',
    'scf_type': 'pk',
    'freeze_core': True,
    'e_convergence': 1e-9,
    'd_convergence': 1e-8,
}})

# Define the molecule.
molecule {system_key} {{
{molecule_block}
}}

# Perform a supramolecular HF energy calculation.
energy('SCF')
"""
    output_file = os.path.join(output_folder, f"{system_key}_HF_{basis_hf}.in")
    with open(output_file, "w") as f:
        f.write(input_text)
    
    return output_file

def main():
    # Directories for the output input files.
    sapt0_output_dir = "/home/evanich/phymol-dc1/water_dimer_analysis/sapt0_inputs"
    hf_output_dir = "/home/evanich/phymol-dc1/water_dimer_analysis/hf_inputs"
    
    # Select basis sets.
    sap_basis = "aug-cc-pVTZ"  # Recommended/truncated aug-cc-pVDZ variant for SAPT0.
    hf_basis   = "aug-cc-pVTZ"  # For supramolecular HF.
    
    for system_key, geom_str in SYSTEMS.items():
        # Generate SAPT0 input file.
        sapt0_in = generate_sapt0_input(geom_str, system_key, sap_basis, sapt0_output_dir)
        print(f"Generated SAPT0 input file: {sapt0_in}")
        
        # Generate HF input file.
        hf_in = generate_hf_input(geom_str, system_key, hf_basis, hf_output_dir)
        print(f"Generated HF input file: {hf_in}")

if __name__ == "__main__":
    main()
