"""
SART-FQ High-Level Driver Script

This script demonstrates how to use the SART-FQ code with the new selective
potential calculation flags and visualization integration.

Features:
- Dynamic loading of molecular systems from external files
- Support for batch calculations over multiple distances
- Unique output naming to prevent overwriting
- Organized output folders for images and CSV files
- Command-line control of all calculation parameters
- Advanced visualization of energy components vs distance during scans
- Distinction between mathematical (primary) and physical energy components
"""

import psi4
import numpy as np
import sys
import os
import importlib.util
import re
from pathlib import Path
import datetime
import csv
import uuid
import warnings
import io
import pandas as pd
from contextlib import redirect_stdout


# Import modules from the new structure
from utils.sart_hf_fq import do_sart_fq, fqsart_inputs_default
from utils.sart_intermediate_utils import process_sart_results, generate_sart_report
from utils.helper_SAPT import helper_SAPT

# --- Psi4 and Global Settings ---
BASIS = 'aug-cc-pVTZ'
MEMORY = '500 MB'
E_CONV = 1.0e-11
MAXITER = 100

# --- Output Directory Structure ---
OUTPUT_ROOT = Path("SART_Output")
IMAGES_DIR = OUTPUT_ROOT / "images"
CSV_DIR = OUTPUT_ROOT / "csv"
REPORTS_DIR = OUTPUT_ROOT / "reports"

# Create output directories
for directory in [OUTPUT_ROOT, IMAGES_DIR, CSV_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def load_systems_from_file(file_path):
    """
    Load molecular systems from a Python file.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the Python file containing system definitions
        
    Returns:
    --------
    dict
        Dictionary of molecular systems
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: System file '{file_path}' not found.")
        return {}
    
    # Load the module dynamically
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Look for TEST_SYSTEMS or SYSTEMS dictionary
    if hasattr(module, 'TEST_SYSTEMS'):
        return module.TEST_SYSTEMS
    elif hasattr(module, 'SYSTEMS'):
        return module.SYSTEMS
    else:
        print(f"Warning: No systems dictionary found in '{file_path}'.")
        return {}

def get_available_systems(system_files=None):
    """
    Get all available molecular systems from specified files.
    
    Parameters:
    -----------
    system_files : list, optional
        List of file paths to load systems from
        
    Returns:
    --------
    dict
        Combined dictionary of all molecular systems
    """
    if system_files is None:
        # Default system files
        system_files = [
            "systems.py",
            "systems_H3B_NH3.py"
        ]
    
    all_systems = {}
    for file_path in system_files:
        systems = load_systems_from_file(file_path)
        all_systems.update(systems)
    
    return all_systems

def format_geometry_string(geometry_str, separation, additional_params=None):
    """
    Format a geometry string with separation distance and additional parameters.
    
    Parameters:
    -----------
    geometry_str : str
        Geometry string template with placeholders
    
    separation : float
        Separation distance in bohr
    
    additional_params : dict, optional
        Additional parameters to format the geometry string
        
    Returns:
    --------
    str
        Formatted geometry string
    """
    # Default parameters
    params = {'R': separation}
    
    # Add additional parameters
    if additional_params:
        params.update(additional_params)
    
    # Handle special case for H2O..H2O where R_plus = R + 0.504
    if 'R_plus' not in params and '{R_plus}' in geometry_str:
        params['R_plus'] = separation + 0.504
    
    # Format the geometry string
    try:
        return geometry_str.format(**params)
    except KeyError as e:
        print(f"Error: Missing parameter {e} in geometry string.")
        return geometry_str

def generate_unique_filename(base_name, system_name, separation, config_name, extension):
    """
    Generate a unique filename for output files to prevent overwriting.
    
    Parameters:
    -----------
    base_name : str
        Base name for the file
    
    system_name : str
        Name of the molecular system
    
    separation : float
        Separation distance in bohr
    
    config_name : str
        Name of the configuration
    
    extension : str
        File extension (without dot)
        
    Returns:
    --------
    str
        Unique filename
    """
    # Clean system name and config name for use in filenames
    clean_system = re.sub(r'[^\w\-\.]', '_', system_name)
    clean_config = re.sub(r'[^\w\-\.]', '_', config_name)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create unique filename
    filename = f"{base_name}_{clean_system}_{separation:.2f}_{clean_config}_{timestamp}.{extension}"
    
    return filename

def run_sart_calculation(system_name, geometry_str, separation=None, config=None, config_name=None):
    """
    Run a SART-FQ calculation with the specified configuration.
    
    Parameters:
    -----------
    system_name : str
        Name of the molecular system
    
    geometry_str : str
        Geometry string in Psi4 format
    
    separation : float, optional
        Separation distance in bohr
    
    config : dict, optional
        Configuration dictionary (defaults to all potentials active)
    
    config_name : str, optional
        Name of the configuration for output files
        
    Returns:
    --------
    dict
        Dictionary containing the results of the SART-FQ calculation
    """
    # Set up Psi4
    psi4.core.set_output_file("output.dat", False)
    psi4.set_memory(MEMORY)
    psi4.set_options({
        'basis': BASIS,
        'scf_type': 'direct',
        'e_convergence': E_CONV,
        'd_convergence': 1e-8
    })
    
    # Create dimer
    dimer = psi4.geometry(geometry_str)
    
    # Generate a unique system name for this calculation
    if config_name is None:
        config_name = "default"
    
    sys_name = f"{system_name}_{separation:.2f}_{BASIS}".replace(' ', '_')
    unique_name = f"{sys_name}_{config_name}"
    dimer.set_name(unique_name)
    
    # Default configuration (all potentials active)
    if config is None:
        config = {
            'calc_elst_pot': True,
            'calc_exch_pot': True,
            'calc_lm': True,
            'FQ_elst_exch': True,
            'FQ_LM': True,
            'SQ_elst_exch': False,
            'SQ_LM': False,
            'Half_symmetrization': False,
            'Full_symmetrization': False,
            'debug_verbosity_level': 0
        }
    
    # Create input dictionary
    input_dict = {
        **fqsart_inputs_default,
        **config,
        'job_name': unique_name,
        'job_method': 'SART-FQ',
        'job_basis': BASIS,
        'job_identifier': config.get('job_identifier', str(uuid.uuid4())),
        'system_name': system_name,
        'max_iter': MAXITER,
        'tol': E_CONV
    }
    
    # Print calculation information
    print(f"\n{'='*80}")
    print(f"Running SART-FQ calculation for {system_name}")
    print(f"{'='*80}")
    print(f"Distance: {separation:.2f} bohr")
    print(f"Configuration: {config_name}")
    print(f"Basis: {BASIS}")
    print(f"SCF type: direct")
    print(f"Convergence threshold: {E_CONV}")
    print(f"Max iterations: {MAXITER}")
    print(f"Job identifier: {input_dict['job_identifier']}")
    print(f"\nPotential calculation flags:")
    print(f"  calc_elst_pot: {config.get('calc_elst_pot', True)}")
    print(f"  calc_exch_pot: {config.get('calc_exch_pot', True)}")
    print(f"  calc_lm: {config.get('calc_lm', True)}")
    print(f"  FQ_elst_exch: {config.get('FQ_elst_exch', True)}")
    print(f"  FQ_LM: {config.get('FQ_LM', True)}")
    print(f"  SQ_elst_exch: {config.get('SQ_elst_exch', False)}")
    print(f"  SQ_LM: {config.get('SQ_LM', False)}")
    print(f"  Half_symmetrization: {config.get('Half_symmetrization', False)}")
    print(f"  Full_symmetrization: {config.get('Full_symmetrization', False)}")
    print(f"  debug_verbosity_level: {config.get('debug_verbosity_level', 0)}")
    print(f"  debug_functions: {config.get('debug_functions', False)}")
    print(f"{'='*80}\n")
    
    try:
        # Initialize helper_SAPT
        h_sapt = helper_SAPT(dimer=dimer)
        
        # Run SART-FQ calculation
        sart_results = do_sart_fq(
            dimer=dimer,
            sapt=h_sapt,
            input_dict=input_dict
        )
        
        # Generate unique filenames for outputs
        plot_filename = generate_unique_filename(
            "energy_plot", system_name, separation, config_name, "png"
        )
        csv_filename = generate_unique_filename(
            "energy_iteration_data", system_name, separation, config_name, "csv"
        )
        report_filename = generate_unique_filename(
            "report", system_name, separation, config_name, "txt"
        )
        
        # Process results
        output_info = process_sart_results(
            sart_results=sart_results,
            print_results=config.get('print_results', True),
            plot_results=config.get('plot_results', True),
            separation=separation,
            output_dir=str(IMAGES_DIR),  # Save plots in the images directory
            verbose=config.get('debug_verbosity_level', 0) > 0  # Only print verbose output if debug level > 0
        )
        
        # Generate report
        report_files = generate_sart_report(
            sart_results=sart_results,
            plot_results=output_info.get('plot_result'),
            output_dir=str(REPORTS_DIR),
            base_filename=unique_name,
            include_plots=config.get('plot_results', True),
            separation=separation,
            verbose=config.get('debug_verbosity_level', 0) > 0  # Only print verbose output if debug level > 0
        )

       # Save energy data at each iteration to CSV (standardized version)
        if sart_results.get("energy_iterations_data"):
            # Define the desired columns
            columns = ["Iteration", "dEA", "dEB", "E1_elst", "E1_exch", "E_elst", 
                       "E_exch", "E_LM", "E_LMA", "E_LMB", "E_int"]

            iteration_rows = []
            for row in sart_results["energy_iterations_data"]:
                # We assume each 'row' is a dictionary; if it is a list, you should modify the extraction accordingly.
                iteration    = row.get("Iteration", None)
                dEA          = row.get("dEA", 0.0)
                dEB          = row.get("dEB", 0.0)
                E1_elst      = row.get("E1_elst", 0.0)
                E1_exch      = row.get("E1_exch", 0.0)
                E_elst       = row.get("E_elst", 0.0)
                E_exch       = row.get("E_exch", 0.0)
                E_LM         = row.get("E_LM", 0.0)
                E_LMA        = row.get("E_LMA", 0.0)
                E_LMB        = row.get("E_LMB", 0.0)
                E_int        = row.get("E_int", 0.0)

                # Compute total deformation energy for each monomer if E_LMA/E_LMB available.
                E_def_tot_A  = dEA + E_LMA
                E_def_tot_B  = dEB + E_LMB

                iteration_rows.append({
                    "Iteration": iteration,
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
                })

            # Create a DataFrame using the list of dictionaries
            df_iter = pd.DataFrame(iteration_rows, columns=["Iteration", "dEA", "dEB", "E1_elst", 
                                                              "E1_exch", "E_elst", "E_exch", 
                                                              "E_LM", "E_LMA", "E_LMB", "E_int",
                                                              "E_def_tot[A]", "E_def_tot[B]"])

            # Use the same csv_filename (already generated) to store iteration data.
            csv_path = CSV_DIR / csv_filename
            df_iter.to_csv(csv_path, index=False)
            print(f"Iteration energy data saved to: {csv_path}")


        ## Save energy data to CSV
        #if sart_results.get("energy_iterations_data"):
        #    csv_path = CSV_DIR / csv_filename
        #    with open(csv_path, 'w', newline='') as csvfile:
        #        writer = csv.writer(csvfile)
        #        writer.writerow(['Iteration', 'dEA', 'dEB', 'E_elst', 'E_exch', 'E_LM', 'E_int'])
        #        for row in sart_results["energy_iterations_data"]:
        #            writer.writerow(row)
        #    print(f"Energy data saved to: {csv_path}")
        
        print(f"\nSART-FQ calculation completed successfully.")
        print(f"Results saved to: {report_files.get('results_txt', 'Unknown')}")
        if 'plots' in report_files:
            print(f"Plots saved to: {', '.join(report_files['plots'].values())}")
        
        return sart_results
    
    except Exception as e:
        print(f"\nError during SART-FQ calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_distance_scan(system_name, system_template, distances, config=None, config_name=None):
    """
    Run SART-FQ calculations for a system at multiple distances.
    
    Parameters:
    -----------
    system_name : str
        Name of the molecular system
    
    system_template : str
        Geometry string template with placeholders
    
    distances : list
        List of separation distances in bohr
    
    config : dict, optional
        Configuration dictionary
    
    config_name : str, optional
        Name of the configuration for output files
        
    Returns:
    --------
    dict
        Dictionary of results for each distance
    """
    results = {}
    output_files = []
    
    # Determine method (FQ or SQ)
    method = "FQ"
    if config and config.get('FQ_elst_exch', True) == False and config.get('SQ_elst_exch', False) == True:
        method = "SQ"
    elif config and config.get('FQ_LM', True) == False and config.get('SQ_LM', False) == True:
        method = "SQ"
    
    # Loop over distances for scan.
    for distance in distances:
        print(f"\nRunning calculation for {system_name} at distance {distance:.2f} bohr")
        
        # Format geometry string with current distance
        geometry_str = format_geometry_string(system_template, distance)
        
        # Run calculation
        result = run_sart_calculation(
            system_name=system_name,
            geometry_str=geometry_str,
            separation=distance,
            config=config,
            config_name=config_name
        )
        
        # Store result
        results[distance] = result
        
        # Store output file path for scan plotting
        if result:
            # Generate single point calculation report file for scan analysis
            # This report contains the output table of sart_data_print
            try:
                from utils.sart_print_utils import sart_data_print
                
                # Create a dictionary in the format expected by sart_data_print
                print_dict = {
                    'job_index': result.get('job_index', 0),
                    'job_name': result.get('job_name', f'SART-FQ calculation at {distance:.2f} bohr'),
                    'job_method': result.get('job_method', 'SART-FQ'),
                    'job_basis': result.get('job_basis', 'Unknown'),
                    'job_identifier': result.get('job_identifier', f'{system_name}_{distance:.2f}'),
                    'dEA': result.get('dEA', 0.0),
                    'dEB': result.get('dEB', 0.0),
                    'E1_elst': result.get('E1_elst', 0.0),
                    'E1_exch': result.get('E1_exch', 0.0),
                    'E_elst': result.get('E_elst', 0.0),
                    'E_exch': result.get('E_exch', 0.0),
                    'E_LMA': result.get('E_LMA', 0.0),
                    'E_LMB': result.get('E_LMB', 0.0),
                    'E_LM': result.get('E_LM', 0.0),
                    'E_int': result.get('E_int', 0.0)
                }
                
                # Capture the output of sart_data_print
                with io.StringIO() as buf, redirect_stdout(buf):
                    detailed_report = sart_data_print(print_dict)
                    output_text = buf.getvalue()
                
                # Save to file
                report_filename = generate_unique_filename(
                    "single_point_report", system_name, distance, config_name, "txt"
                )
                report_path = REPORTS_DIR / report_filename
                
                with open(report_path, 'w') as f:
                    # Add system and distance information at the top
                    f.write(f"Running SART-{method} calculation for {system_name}\n")
                    f.write(f"Distance: {distance} bohr\n")
                    f.write(f"Method: {method} for Elst+Exch, {method} for LM\n\n")
                    
                    # Add the captured output from sart_data_print
                    f.write(output_text)
                    
                    # Add convergence information at the bottom
                    f.write(f"\nConverged: {result.get('converged', False)}\n")
                    f.write(f"Iterations: {result.get('iterations', 0)}\n")
                
                print(f"Single point calculation report for distance {distance:.2f} saved to: {report_path}")
                output_files.append(str(report_path))
            
            except Exception as e:
                print(f"Error generating single point calculation report for distance {distance:.2f}: {e}")
    
    # Process scan results
    if output_files:
        try:
            from utils.sart_scan_utils import process_scan_results
            
            # Generate timestamp for scan output files
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Process scan results
            scan_results = process_scan_results(
                output_files=output_files,
                output_dir=str(CSV_DIR),
                debug=config.get('debug_verbosity_level', 0) > 0  # Only debug if verbosity > 1
            )
            
            # Print summary of generated files
            if scan_results:
                if 'scan_summary' in scan_results:
                    print(f"Scan summary saved to: {scan_results['scan_summary']}")
                
                if 'energy_data' in scan_results:
                    print(f"Energy data saved to: {scan_results['energy_data']}")
                
                if 'plots' in scan_results:
                    print("\nGenerated scan plots:")
                    
                    if 'primary_energies' in scan_results['plots']:
                        print("  primary_energies:")
                        for unit, path in scan_results['plots']['primary_energies'].items():
                            print(f"    {unit}: {path}")
                    
                    if 'physical_energies' in scan_results['plots']:
                        print("  physical_energies:")
                        for unit, path in scan_results['plots']['physical_energies'].items():
                            print(f"    {unit}: {path}")
        
        except Exception as e:
            print(f"Warning: Error generating scan scan report: {e}")
    
    return results

def generate_scan_report(results, system_name, config_name):
    """
    Generate a scan report for all calculations.
    
    Parameters:
    -----------
    results : dict
        Dictionary of results for each distance
    
    system_name : str
        Name of the molecular system
    
    config_name : str
        Name of the configuration
        
    Returns:
    --------
    str
        Path to the generated report file
    """
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename
    filename = f"scan_report_{system_name}_{config_name}_{timestamp}.txt"
    filepath = REPORTS_DIR / filename
    
    # Generate report content
    with open(filepath, 'w') as f:
        f.write(f"scan Report for {system_name}\n")
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        
        # Write summary table
        f.write("Summary Table:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Distance':^10} | {'E_int':^15} | {'E_elst':^15} | {'E_exch':^15} | {'E_LM':^15}\n")
        f.write(f"{'-'*80}\n")
        
        for distance, result in sorted(results.items()):
            if result:
                e_int = result.get('E_int', 0.0)
                e_elst = result.get('E_elst', 0.0)
                e_exch = result.get('E_exch', 0.0)
                e_lm = result.get('E_LM', 0.0)
                
                f.write(f"{distance:^10.2f} | {e_int:^15.8f} | {e_elst:^15.8f} | {e_exch:^15.8f} | {e_lm:^15.8f}\n")
        
        f.write(f"{'-'*80}\n\n")
        
        # Write detailed results for each distance
        for distance, result in sorted(results.items()):
            if result:
                f.write(f"Detailed Results for Distance {distance:.2f} bohr:\n")
                f.write(f"{'-'*80}\n")
                
                # Write energy components
                f.write("Energy Components:\n")
                for key, value in result.items():
                    if isinstance(value, (int, float)) and not key.startswith('_'):
                        f.write(f"  {key}: {value:.8f}\n")
                
                f.write("\n")
    
    print(f"scan report saved to: {filepath}")
    return str(filepath)

def main():
    """
    Main function to run SART-FQ calculations.
    """
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run SART-FQ calculations with selective potential calculation flags.')
    
    # System selection
    parser.add_argument('--system', type=str, default='He..Li+',
                        help='Name of the molecular system to calculate (default: He..Li+)')
    
    # Distance options
    distance_group = parser.add_mutually_exclusive_group()
    distance_group.add_argument('--distance', type=float, default=3.0,
                               help='Separation distance in bohr (default: 3.0)')
    distance_group.add_argument('--scan', action='store_true',
                               help='Run a distance scan')
    parser.add_argument('--scan-start', type=float, default=0.5,
                       help='Starting distance for scan in bohr (default: 0.5)')
    parser.add_argument('--scan-end', type=float, default=5.0,
                       help='Ending distance for scan in bohr (default: 5.0)')
    parser.add_argument('--scan-steps', type=int, default=10,
                       help='Number of steps for scan (default: 10)')
    
    # Potential calculation flags
    parser.add_argument('--no-elst', action='store_true',
                       help='Disable electrostatic potential calculation')
    parser.add_argument('--no-exch', action='store_true',
                       help='Disable exchange potential calculation')
    parser.add_argument('--no-lm', action='store_true',
                       help='Disable Landshoff-Meister potential calculation')
    
    # Method selection
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument('--fq-method', action='store_true',
                             help='Use first quantization method for all potentials (default)')
    method_group.add_argument('--sq-method', action='store_true',
                             help='Use second quantization method for all potentials')
    
    # Individual method selection
    parser.add_argument('--fq-elst-exch', action='store_true',
                       help='Use first quantization method for electrostatic and exchange potentials')
    parser.add_argument('--sq-elst-exch', action='store_true',
                       help='Use second quantization method for electrostatic and exchange potentials')
    parser.add_argument('--fq-lm', action='store_true',
                       help='Use first quantization method for Landshoff-Meister potential')
    parser.add_argument('--sq-lm', action='store_true',
                       help='Use second quantization method for Landshoff-Meister potential')
    
    # Symmetrization options
    sym_group = parser.add_mutually_exclusive_group()
    sym_group.add_argument('--half-sym', action='store_true',
                          help='Use half symmetrization')
    sym_group.add_argument('--full-sym', action='store_true',
                          help='Use full symmetrization')
    
    # Output options
    parser.add_argument('--no-print', action='store_true',
                       help='Disable printing of results')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting of results')
    parser.add_argument('--verbosity', type=int, default=0,
                       help='Verbosity level (0-3, default: 0)')
    parser.add_argument('--debug-functions', action='store_true',
                       help='Enable debugging of specific functions')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get available systems
    all_systems = get_available_systems()
    
    # Check if system exists
    if args.system not in all_systems:
        print(f"Error: System '{args.system}' not found.")
        print("Available systems:")
        for name in sorted(all_systems.keys()):
            print(f"  {name}")
        return
    
    # Get system template
    system_template = all_systems[args.system]
    
    # Create configuration dictionary
    config = {
        'calc_elst_pot': not args.no_elst,
        'calc_exch_pot': not args.no_exch,
        'calc_lm': not args.no_lm,
        'FQ_elst_exch': args.fq_method or args.fq_elst_exch or (not args.sq_method and not args.sq_elst_exch),
        'FQ_LM': args.fq_method or args.fq_lm or (not args.sq_method and not args.sq_lm),
        'SQ_elst_exch': args.sq_method or args.sq_elst_exch,
        'SQ_LM': args.sq_method or args.sq_lm,
        'Half_symmetrization': args.half_sym,
        'Full_symmetrization': args.full_sym,
        'print_results': not args.no_print,
        'plot_results': not args.no_plot,
        'debug_verbosity_level': args.verbosity,
        'debug_functions': args.debug_functions
    }
    
    # Generate configuration name
    config_name = "all_potentials"
    if args.no_elst or args.no_exch or args.no_lm:
        config_name = ""
        if not args.no_elst:
            config_name += "elst_"
        if not args.no_exch:
            config_name += "exch_"
        if not args.no_lm:
            config_name += "lm_"
        config_name = config_name.rstrip('_')
    
    if args.sq_method:
        config_name += "_SQ"
    elif args.sq_elst_exch or args.sq_lm:
        config_name += "_mixed"
    
    if args.half_sym:
        config_name += "_half_sym"
    elif args.full_sym:
        config_name += "_full_sym"
    
    # Run calculation
    if args.scan:
        # Generate distances for scan
        distances = np.linspace(args.scan_start, args.scan_end, args.scan_steps)
        
        print(f"\nRunning distance scan for {args.system} from {args.scan_start:.2f} to {args.scan_end:.2f} bohr ({args.scan_steps} steps)")
        print(f"Configuration: {config_name}")
        
        # Run scan
        results = run_distance_scan(
            system_name=args.system,
            system_template=system_template,
            distances=distances,
            config=config,
            config_name=config_name
        )
        
        # Generate scan report
        generate_scan_report(
            results=results,
            system_name=args.system,
            config_name=config_name
        )
    else:
        # Format geometry string
        geometry_str = format_geometry_string(system_template, args.distance)
        
        print(f"\nRunning single point calculation for {args.system} at {args.distance:.2f} bohr")
        print(f"Configuration: {config_name}")
        
        # Run single point calculation
        run_sart_calculation(
            system_name=args.system,
            geometry_str=geometry_str,
            separation=args.distance,
            config=config,
            config_name=config_name
        )

if __name__ == "__main__":
    main()
