"""
SART-FQ Utilities - Intermediate-level functions

This module provides intermediate-level functions for SART-FQ calculations,
including printing, plotting, and result processing.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd
from contextlib import redirect_stdout

# Add path to import modules - Corrected path
sys.path.append('/home/evanich/phymol-dc1/sart-fq-test2/utils')

def process_sart_results(sart_results, print_results=True, plot_results=True, separation=None, output_dir=None, verbose=False):
    """
    Process SART-FQ calculation results for visualization and printing.
    (Rest of docstring unchanged)
    """
    output_info = {}
    
    # Prepare data for sart_data_print
    if print_results:
        try:
            from utils.sart_print_utils import sart_data_print
            print_dict = {
                'job_index': sart_results.get('job_index', 0),
                'job_name': sart_results.get('job_name', 'SART-FQ calculation'),
                'job_method': sart_results.get('job_method', 'SART-FQ'),
                'job_basis': sart_results.get('job_basis', 'Unknown'),
                'job_identifier': sart_results.get('job_identifier', 'Unknown'),
                'dEA': sart_results.get('dEA', 0.0),
                'dEB': sart_results.get('dEB', 0.0),
                'E1_elst': sart_results.get('E1_elst', 0.0),
                'E1_exch': sart_results.get('E1_exch', 0.0),
                'E_elst': sart_results.get('E_elst', 0.0),
                'E_exch': sart_results.get('E_exch', 0.0),
                'E_LMA': sart_results.get('E_LMA', 0.0),
                'E_LMB': sart_results.get('E_LMB', 0.0),
                'E_LM': sart_results.get('E_LM', 0.0),
            }
            
            with io.StringIO() as buf, redirect_stdout(buf):
                print_result = sart_data_print(print_dict)
                output_text = buf.getvalue()
            output_info['print_result'] = print_result
            output_info['print_output'] = output_text
        except ImportError:
            if verbose:
                print("Warning: sart_print_utils module not found. Skipping detailed printing.")
    
    # Generate plots if requested and if energy iteration data is present
    if plot_results and sart_results.get("energy_iterations_data"):
        try:
            # Use the updated CSV-writing block.
            # We assume each iteration row is a dictionary.
            # Build a DataFrame from the list of dictionaries.
            df_iter = pd.DataFrame(sart_results["energy_iterations_data"])
            # At this point, if you want to verify, print the DF head if verbose:
            if verbose:
                print("Iteration data DataFrame:")
                print(df_iter.head())
            
            # Optionally, you may re-save this standard CSV now,
            # e.g., to output_dir if desired. This example assumes that
            # the CSV has already been created in sart_driver.py.
            
            # Now, to call your plotting routine for iteration data,
            # you might (for instance) call a function like this:
            try:
                from utils.sart_plot_utils import plot_energy_iterations_direct
                # Here, we convert the DataFrame back to a list of dictionaries for compatibility.
                iter_data_list = df_iter.to_dict(orient="records")
                plot_result = plot_energy_iterations_direct(
                    iter_data_list,
                    separation=separation,
                    system_name=sart_results.get('system_name', None),
                    filename=None,  # or provide a file path if needed
                    verbose=verbose
                )
                output_info['plot_result'] = plot_result
            except ImportError:
                if verbose:
                    print("Warning: sart_plot_utils module not found. Skipping plotting.")
        except Exception as e:
            if verbose:
                print(f"Error processing iteration energy data: {e}")
    
    return output_info

def generate_sart_report_header(sart_results, separation=None):
    """
    Generate a header for SART-FQ report.
    
    Parameters:
    -----------
    sart_results : dict
        Dictionary containing SART-FQ calculation results
    
    separation : float, optional
        Separation distance in bohr
        
    Returns:
    --------
    str
        Formatted header string
    """
    # Determine method (FQ or SQ)
    method = "FQ"
    if sart_results.get('FQ_elst_exch', True) == False and sart_results.get('SQ_elst_exch', False) == True:
        method = "SQ"
    elif sart_results.get('FQ_LM', True) == False and sart_results.get('SQ_LM', False) == True:
        method = "SQ"
    
    # Create header
    header_lines = []
    header_lines.append(f"Running SART-{method} calculation for {sart_results.get('system_name', 'Unknown')}")
    
    if separation is not None:
        header_lines.append(f"Distance: {separation} bohr")
    
    header_lines.append(f"Method: {method} for Elst+Exch, {method} for LM")
    header_lines.append("")  # Empty line after header
    
    return "\n".join(header_lines)

def generate_sart_report(sart_results, output_dir='.', base_filename=None, include_plots=True, separation=None, plot_results=None, verbose=False):
    """
    Generate a comprehensive report of SART-FQ results.
    
    Parameters:
    -----------
    sart_results : dict
        Dictionary containing SART-FQ calculation results
    output_dir : str, default='.'
        Directory to save the report files
    base_filename : str, optional
        Base filename for the report files (default: derived from job_name)
    include_plots : bool, default=True
        Whether to include plots in the report
    separation : float, optional
        Separation distance in bohr for plot titles
    plot_results : dict, optional
        Dictionary with plot results from a previous call to plot_energy_iterations_direct.
        If provided, no new plots will be generated.
    verbose : bool, default=False
        Whether to print verbose output
        
    Returns:
    --------
    dict
        Dictionary with paths to the generated report files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine base filename
    if base_filename is None:
        job_name = sart_results.get('job_name', 'sart_results')
        system_name = sart_results.get('system_name', '')
        
        # Avoid duplicating system_name if it's already in job_name
        if system_name and system_name in job_name:
            base_filename = job_name.replace(' ', '_')
        else:
            base_filename = f"{job_name}_{system_name}".replace(' ', '_')
    
    report_files = {}
    
    # Process results to get sart_data_print output
    output_info = process_sart_results(
        sart_results=sart_results,
        print_results=True,
        plot_results=False,  # We'll handle plotting separately
        separation=separation,
        verbose=verbose
    )
    
    # Generate report content
    report_content = []
    
    # Add header
    report_content.append(generate_sart_report_header(sart_results, separation))
    
    # Add sart_data_print output if available
    if 'print_output' in output_info:
        report_content.append(output_info['print_output'])
    
    # Save results to text file
    results_file = os.path.join(output_dir, f"{base_filename}_results.txt")
    with open(results_file, 'w') as f:
        f.write("\n".join(report_content))
    
    report_files['results_txt'] = results_file
    
    # Use existing plot results or generate new ones if requested
    if plot_results:
        report_files['plots'] = plot_results
    elif include_plots and 'energy_iterations_data' in sart_results:
        try:
            from utils.sart_plot_utils import plot_energy_iterations_direct
            
            # Get system name
            system_name = sart_results.get('system_name', None)
            
            # Generate plot filename in the output directory
            components_filename = os.path.join(output_dir, f"{base_filename}_energy_components.png")
            
            # Convert iteration data to a standardized list-of-dictionaries, using pandas.
            iter_data_list = pd.DataFrame(sart_results["energy_iterations_data"]).to_dict(orient="records")
            
            # Call plot_energy_iterations_direct with the standardized data
            plot_result = plot_energy_iterations_direct(
                iter_data_list,
                separation=separation,
                system_name=system_name,
                filename=components_filename,
                verbose=verbose
            )
            report_files['plots'] = plot_result
        except ImportError:
            if verbose:
                print("Warning: sart_plot_utils module not found. Skipping plotting.")
    
    return report_files


def generate_scan_report(scan_results, output_dir='.', system_name=None, method=None, config_name=None, verbose=False):
    """
    Generate a comprehensive report for a distance scan.
    
    Parameters:
    -----------
    scan_results : dict
        Dictionary containing results for each distance
    
    output_dir : str, default='.'
        Directory to save the report files
    
    system_name : str, optional
        Name of the molecular system
    
    method : str, optional
        Method used (FQ or SQ)
    
    config_name : str, optional
        Name of the configuration
        
    verbose : bool, default=False
        Whether to print verbose output
        
    Returns:
    --------
    dict
        Dictionary with paths to the generated report files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename for the scan report
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean system name and config name for use in filenames
    clean_system = system_name.replace(' ', '_') if system_name else "Unknown"
    clean_config = config_name.replace(' ', '_') if config_name else "all_potentials"
    
    # Get min distance for filename
    distances = sorted(scan_results.keys())
    min_distance = distances[0] if distances else 0.0
    
    base_filename = f"detailed_report_{clean_system}_{min_distance:.2f}_{clean_config}_{timestamp}"
    report_file = os.path.join(output_dir, f"{base_filename}.txt")
    
    # Generate report content
    report_content = []
    
    # Add header
    report_content.append(f"SART-{method} Distance Scan Report")
    report_content.append(f"System: {system_name}")
    report_content.append(f"Method: {method}")
    report_content.append(f"Configuration: {config_name}")
    report_content.append(f"Distances: {', '.join([f'{d:.2f}' for d in distances])} bohr")
    report_content.append(f"Generated: {timestamp}")
    report_content.append("=" * 80)
    report_content.append("")
    
    # Add results for each distance
    for distance in distances:
        result = scan_results[distance]
        
        # Process results to get sart_data_print output
        output_info = process_sart_results(
            sart_results=result,
            print_results=True,
            plot_results=False,
            separation=distance,
            verbose=verbose
        )
        
        # Add header for this distance
        report_content.append(f"{'='*40} Distance: {distance:.2f} bohr {'='*40}")
        report_content.append("")
        
        # Add sart_data_print output if available
        if 'print_output' in output_info:
            report_content.append(output_info['print_output'])
        
        # Add separator between distances
        report_content.append("\n" + "=" * 80 + "\n")
    
    # Save report to file
    with open(report_file, 'w') as f:
        f.write("\n".join(report_content))
    
    return {'scan_report': report_file}
