#!/usr/bin/env python3
"""
sart_driver_h2o_h2o.py

Driver for single-point SART-FQ calculations on H2O..H2O systems.

Workflow:
  1. Import systems (from systems_H2O_H2O.txt).
  2. Extract intermolecular distance from each system's geometry (from a comment like "# R_00 = 2.2 Angstrom")
     and convert to bohr.
  3. Run a SART-FQ single-point calculation for each system using run_sart_calculation.
  4. Build a standardized DataFrame containing:
      "System", "Distance", "dEA", "dEB", "E1_elst", "E1_exch", 
      "E_elst", "E_exch", "E_LM", "E_LMA", "E_LMB", "E_int",
      "E_def_tot[A]" (dEA + E_LMA) and "E_def_tot[B]" (dEB + E_LMB).
  5. Save the DataFrame to a CSV file.
  
The CSV file is saved under a repository folder called "water_dimer_datas"
and its filename includes the basis set (from the global variable BASIS).

"""

import os
import re
import pandas as pd

# Global settings 
BASIS = 'aug-cc-pVTZ'
MEMORY = '500 MB'
E_CONV = 1.0e-7
MAXITER = 100

# Conversion factor: 1 Angstrom = 1.88973 bohr
ANG_TO_BOHR = 1.88973

# Make sure the output directory for CSV files exists
REPO_DIR = "water_dimer_datas"
os.makedirs(REPO_DIR, exist_ok=True)

# Import your systems and calculation routine.
from systems_H2O_H2O import SYSTEMS
from sart_driver import run_sart_calculation

# --------------------------------------------------------------------
# Extract the distance from the geometry string.
# Expect a line like: "# R_00 = 2.2 Angstrom"
# --------------------------------------------------------------------
def extract_distance_from_geometry(geom_str):
    distance_ang = None
    for line in geom_str.splitlines():
        if "R_00" in line:
            m = re.search(r"R_00\s*=\s*([0-9]*\.?[0-9]+)", line)
            if m:
                distance_ang = float(m.group(1))
                break
    if distance_ang is None:
        raise ValueError("Could not extract R_00 distance (in Angstrom) from geometry string.")
    return distance_ang * ANG_TO_BOHR

# --------------------------------------------------------------------
# Run single-point calculations for all systems.
# --------------------------------------------------------------------
def run_systems_calculations(systems, config=None):
    """
    Loops over the provided SYSTEMS dictionary.
    For each system:
      - Extract the intermolecular distance (convert from Angstrom->bohr)
      - Run the SART-FQ calculation using run_sart_calculation
    Returns:
      A dictionary keyed by the distance (float, bohr) with the SART output.
    """
    results = {}
    for system_key, geom_str in systems.items():
        try:
            distance_bohr = extract_distance_from_geometry(geom_str)
            print(f"Running calculation for {system_key} at {distance_bohr:.2f} bohr")
            result = run_sart_calculation(
                system_name=system_key,
                geometry_str=geom_str,
                separation=distance_bohr,
                config=config,
                config_name="all_potentials"
            )
            if result:
                results[distance_bohr] = result
            else:
                print(f"Calculation for {system_key} returned no result.")
        except Exception as e:
            print(f"Error processing system {system_key}: {str(e)}")
    return results

# --------------------------------------------------------------------
# Build a standardized DataFrame from the SART-FQ results.
# --------------------------------------------------------------------
def create_results_dataframe(results):
    """
    Constructs a standardized DataFrame containing energy data in Hartree.
    Expected keys in each result:
      "system_name", "dEA", "dEB", "E1_elst", "E1_exch", 
      "E_elst", "E_exch", "E_LM", "E_LMA", "E_LMB", "E_int"
    Computes:
      E_def_tot[A] = dEA + E_LMA
      E_def_tot[B] = dEB + E_LMB
      
    Returns a DataFrame with columns:
      "System", "Distance", "dEA", "dEB", "E1_elst", "E1_exch",
      "E_elst", "E_exch", "E_LM", "E_LMA", "E_LMB", "E_int",
      "E_def_tot[A]", "E_def_tot[B]".
    """
    rows = []
    for distance in sorted(results.keys()):
        res = results[distance]
        system_id = res.get("system_name", "Unknown")
        
        dEA     = res.get("dEA", 0.0)
        dEB     = res.get("dEB", 0.0)
        E1_elst = res.get("E1_elst", 0.0)
        E1_exch = res.get("E1_exch", 0.0)
        E_elst  = res.get("E_elst", 0.0)
        E_exch  = res.get("E_exch", 0.0)
        E_LM    = res.get("E_LM", 0.0)
        E_LMA   = res.get("E_LMA", 0.0)
        E_LMB   = res.get("E_LMB", 0.0)
        E_int   = res.get("E_int", 0.0)
        
        E_def_tot_A = dEA + E_LMA
        E_def_tot_B = dEB + E_LMB
        
        row = {
            "System": system_id,
            "Distance": distance,
            "dEA": dEA,
            "dEB": dEB,
            "E1_elst": E1_elst,
            "E1_exch": E1_exch,
            "E_elst": E_elst,
            "E_exch": E_exch,
            "E_LM": E_LM,
            "E_LMA": E_LMA,
            "E_LMB": E_LMB,
            "E_int": E_int,
            "E_def_tot[A]": E_def_tot_A,
            "E_def_tot[B]": E_def_tot_B
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

# --------------------------------------------------------------------
# Main routine.
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: Define configuration (here we use default settings)
    config_default = {
        'calc_elst_pot': True,
        'calc_exch_pot': True,
        'calc_lm': True,
        'FQ_elst_exch': True,
        'FQ_LM': True,
        'SQ_elst_exch': False,
        'SQ_LM': False,
        'Half_symmetrization': False,
        'Full_symmetrization': False,
        'print_results': True,
        'plot_results': True,
        'debug_verbosity_level': 0,
        'debug_functions': False
    }
    
    # Run single-point calculations for all systems in SYSTEMS.
    results_dict = run_systems_calculations(SYSTEMS, config=config_default)
    print('after calcs')
    
    if not results_dict:
        print("No valid results obtained!")
        exit(1)
    
    # Build the standardized DataFrame.
    df = create_results_dataframe(results_dict)
    
    # Construct the filename. Here we use the basis set in the name.
    csv_filename = f"standardized_sart_results_{BASIS}.csv"
    csv_filepath = os.path.join(REPO_DIR, csv_filename)
    
    # Save the DataFrame to CSV.
    df.to_csv(csv_filepath, index=False)
    
    print("Standardized SART results (in Hartree) saved to CSV:")
    print(df.head())
