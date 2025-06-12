"""
Functions for analyzing and visualizing SART-FQ scan results.

This module provides functions for extracting energy components from SART-FQ
calculation results across different distances and plotting them.
"""

import os
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# Import conversion routines from your sart_print_utils module.
from utils.sart_print_utils import conversion_units_from_hartree, convert_energy

def extract_energy_components_from_file(file_path, debug=False):
    """
    Extract energy components from a SART-FQ output file.
    
    Parameters:
    -----------
    file_path : str
        Path to the SART-FQ output file
    
    debug : bool, default=False
        Whether to print debug information during extraction
        
    Returns:
    --------
    dict
        Dictionary containing extracted energy components
    """
    if debug:
        print(f"Extracting energy components from {file_path}")
    
    # Initialize result dictionary
    result = {
        'primary_energies': {},
        'physical_energies': {},
        'metadata': {}
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract system name - try multiple patterns
        system_patterns = [
            r'Running SART-(?:FQ|SQ) calculation for\s+([A-Za-z0-9\.\+\-]+)',
            r'System:\s+([A-Za-z0-9\.\+\-]+)',
            r'([A-Za-z0-9\.\+\-]+)_[\d\.]+_[\w\-]+_all_potentials'
        ]
        
        for pattern in system_patterns:
            system_match = re.search(pattern, content)
            if system_match:
                result['metadata']['system_name'] = system_match.group(1)
                if debug:
                    print(f"Found system name: {result['metadata']['system_name']}")
                break
        
        # If system name not found, try to extract from filename
        if 'system_name' not in result['metadata']:
            filename = os.path.basename(file_path)
            if '_' in filename:
                possible_system = filename.split('_')[0]
                if '..' in possible_system:  # Likely a system name like He..Li+
                    result['metadata']['system_name'] = possible_system
                    if debug:
                        print(f"Extracted system name from filename: {result['metadata']['system_name']}")
        
        # Extract method (FQ or SQ)
        method_patterns = [
            r'Method:\s+([A-Za-z0-9\-]+)\s+for\s+Elst\+Exch',
            r'Method:\s+([A-Za-z0-9\-]+)'
        ]
        
        for pattern in method_patterns:
            method_match = re.search(pattern, content)
            if method_match:
                result['metadata']['method'] = method_match.group(1)
                if debug:
                    print(f"Found method: {result['metadata']['method']}")
                break
        
        # Extract separation
        separation_patterns = [
            r'Distance:\s+([0-9\.]+)\s+bohr',
            r'Separation:\s+([0-9\.]+)\s+bohr',
            r'_([0-9\.]+)_[\w\-]+_all_potentials'
        ]
        
        for pattern in separation_patterns:
            separation_match = re.search(pattern, content)
            if separation_match:
                result['metadata']['separation'] = float(separation_match.group(1))
                if debug:
                    print(f"Found separation: {result['metadata']['separation']} bohr")
                break
        
        # Extract primary energies from sart_data_print table
        primary_table_pattern = r'# SART primary energies:.*?#\s+-+\n#\s+Description.*?::.*?::\s+(.*?)::.*?\n#\s+-+\n(.*?)#\s+=+'
        primary_match = re.search(primary_table_pattern, content, re.DOTALL)
        
        if primary_match:
            # Extract units from header
            units_header = primary_match.group(1).strip()
            units = [unit.strip() for unit in units_header.split('::')]
            
            # Extract table rows
            table_content = primary_match.group(2)
            rows = table_content.strip().split('\n')
            
            if debug:
                print(f"Found primary energies table with units: {units}")
                print(f"Found {len(rows)} rows in primary energies table")
            
            # Process each row
            for row in rows:
                if row.strip().startswith('#') or not row.strip():
                    continue
                
                # Split row by ::
                parts = row.split('::')
                if len(parts) >= 3:
                    # Extract component name and values
                    component_name = parts[0].strip()
                    latex_name = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Store values for each unit
                    values = {}
                    for i, unit in enumerate(units):
                        if i + 2 < len(parts):
                            try:
                                values[unit] = float(parts[i + 2].strip())
                            except ValueError:
                                if debug:
                                    print(f"  Error parsing value for {component_name}, unit {unit}")
                    
                    # Store in result dictionary
                    result['primary_energies'][component_name] = values
                    
                    if debug:
                        print(f"  Extracted primary energy: {component_name} with values: {values}")
        else:
            if debug:
                print("Primary energies table not found using pattern")
        
        # Extract physical energies from sart_data_print table
        # First try the standard format
        physical_table_pattern = r'# SART physical energies:.*?#\s+-+\n#\s+Energy Name\s+(.*?)\n#\s+-+\n(.*?)#\s+=+'
        physical_match = re.search(physical_table_pattern, content, re.DOTALL)
        
        if physical_match:
            # Extract units from header
            units_header = physical_match.group(1).strip()
            units = [unit.strip() for unit in units_header.split('::')]
            
            # Extract table content
            table_content = physical_match.group(2)
            
            if debug:
                print(f"Found physical energies table with units: {units}")
            
            # Extract energy components using a more flexible pattern
            energy_pattern = r'\s+([A-Za-z0-9_\\{}\^,\.]+)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?(?:\s+[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)*)'
            energy_matches = re.finditer(energy_pattern, table_content)
            
            for match in energy_matches:
                component_name = match.group(1).strip()
                values_str = match.group(2).strip()
                
                # Clean up LaTeX notation in component name
                clean_name = component_name.replace('{\\rm', '').replace('\\rm', '').replace('}', '').replace('_', '_').replace('\\', '')
                
                # Extract values for each unit
                values = {}
                value_parts = values_str.split()
                
                for i, unit in enumerate(units):
                    if i < len(value_parts):
                        try:
                            values[unit] = float(value_parts[i])
                        except ValueError:
                            if debug:
                                print(f"  Error parsing value for {clean_name}, unit {unit}")
                
                # Store in result dictionary
                result['physical_energies'][clean_name] = values
                
                if debug:
                    print(f"  Extracted physical energy: {clean_name} with values: {values}")
        else:
            # Try alternative format with different header or end marker
            physical_table_patterns = [
                r'# SART physical energies:.*?#\s+-+\n#\s+Energy Name\s+(.*?)\n#\s+-+\n(.*?)$',
                r'# SART physical energies:.*?#\s+-+\n#\s+Energy Name\s+(.*?)\n#\s+-+\n(.*?)\n\n',
                r'# SART physical energies:.*?#\s+-+\n#\s+(.*?)\n#\s+-+\n(.*?)#\s+=+'
            ]
            
            for pattern in physical_table_patterns:
                physical_match = re.search(pattern, content, re.DOTALL)
                if physical_match:
                    # Extract units from header
                    units_header = physical_match.group(1).strip()
                    # Try different unit separators
                    if '::' in units_header:
                        units = [unit.strip() for unit in units_header.split('::')]
                    else:
                        units = [unit.strip() for unit in units_header.split()]
                    
                    # Extract table content
                    table_content = physical_match.group(2)
                    
                    if debug:
                        print(f"Found physical energies table with alternative pattern, units: {units}")
                    
                    # Extract rows
                    rows = table_content.strip().split('\n')
                    for row in rows:
                        if row.strip().startswith('#') or not row.strip():
                            continue
                        
                        # Try to extract component name and values
                        # First try with :: separator
                        if '::' in row:
                            parts = row.split('::')
                            if len(parts) >= 2:
                                component_name = parts[0].strip()
                                values = {}
                                for i, unit in enumerate(units):
                                    if i + 1 < len(parts):
                                        try:
                                            values[unit] = float(parts[i + 1].strip())
                                        except ValueError:
                                            if debug:
                                                print(f"  Error parsing value for {component_name}, unit {unit}")
                                
                                # Clean up component name
                                clean_name = component_name.replace('{\\rm', '').replace('\\rm', '').replace('}', '').replace('_', '_').replace('\\', '')
                                
                                # Store in result dictionary
                                result['physical_energies'][clean_name] = values
                                
                                if debug:
                                    print(f"  Extracted physical energy: {clean_name} with values: {values}")
                        else:
                            # Try with space separator
                            parts = row.strip().split()
                            if len(parts) >= len(units) + 1:  # At least component name + values for each unit
                                # Component name might contain spaces, so join all parts except the last len(units)
                                component_name = ' '.join(parts[:-len(units)])
                                value_parts = parts[-len(units):]
                                
                                values = {}
                                for i, unit in enumerate(units):
                                    if i < len(value_parts):
                                        try:
                                            values[unit] = float(value_parts[i])
                                        except ValueError:
                                            if debug:
                                                print(f"  Error parsing value for {component_name}, unit {unit}")
                                
                                # Clean up component name
                                clean_name = component_name.replace('{\\rm', '').replace('\\rm', '').replace('}', '').replace('_', '_').replace('\\', '')
                                
                                # Store in result dictionary
                                result['physical_energies'][clean_name] = values
                                
                                if debug:
                                    print(f"  Extracted physical energy: {clean_name} with values: {values}")
                    
                    # If we found and processed the table, break the loop
                    if result['physical_energies']:
                        break
        
        # If still no physical energies, try direct pattern matching for each component
        if not result['physical_energies']:
            if debug:
                print("No physical energies table found, trying direct pattern matching")
            
            # Define patterns for physical energy components
            physical_patterns = {
                'E_elst': r'E_(?:\\{\\rm\s+|{\rm\s+)?elst(?:\\}|})(?:\^(?:\\{\\rm\s+|{\rm\s+)[^}]+(?:\\}|}))?[:\s]+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                'E_exch': r'E_(?:\\{\\rm\s+|{\rm\s+)?exch(?:\\}|})(?:\^(?:\\{\\rm\s+|{\rm\s+)[^}]+(?:\\}|}))?[:\s]+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                'E_def,tot': r'E_(?:\\{\\rm\s+|{\rm\s+)?def,tot(?:\\}|})(?:\^(?:\\{\\rm\s+|{\rm\s+)[^}]+(?:\\}|}))?[:\s]+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                'E_ind,inter': r'E_(?:\\{\\rm\s+|{\rm\s+)?ind,inter(?:\\}|})(?:\^(?:\\{\\rm\s+|{\rm\s+)[^}]+(?:\\}|}))?[:\s]+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                'E_ind,tot': r'E_(?:\\{\\rm\s+|{\rm\s+)?ind,tot(?:\\}|})(?:\^(?:\\{\\rm\s+|{\rm\s+)[^}]+(?:\\}|}))?[:\s]+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                'E_int': r'E_(?:\\{\\rm\s+|{\rm\s+)?int(?:\\}|})(?:\^(?:\\{\\rm\s+|{\rm\s+)[^}]+(?:\\}|}))?[:\s]+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            }
            
            for key, pattern in physical_patterns.items():
                match = re.search(pattern, content)
                if match:
                    # Assume value is in mH (default unit)
                    result['physical_energies'][key] = {'mH': float(match.group(1))}
                    if debug:
                        print(f"  Extracted physical energy using direct pattern: {key}")
        
        # If we have primary energies but no physical energies, try to derive them
        if result['primary_energies'] and not result['physical_energies']:
            if debug:
                print("Attempting to derive physical energies from primary energies")
            
            # Get primary unit (prefer mH if available)
            primary_units = set()
            for values in result['primary_energies'].values():
                primary_units.update(values.keys())
            
            primary_unit = 'mH' if 'mH' in primary_units else next(iter(primary_units)) if primary_units else None
            
            if primary_unit:
                # Derive E_elst from primary energies if available
                if 'E_elst' in result['primary_energies'] and primary_unit in result['primary_energies']['E_elst']:
                    result['physical_energies']['E_elst'] = {primary_unit: result['primary_energies']['E_elst'][primary_unit]}
                    if debug:
                        print(f"  Derived physical energy E_elst from primary energies")
                
                # Derive E_exch from primary energies if available
                if 'E_exch' in result['primary_energies'] and primary_unit in result['primary_energies']['E_exch']:
                    result['physical_energies']['E_exch'] = {primary_unit: result['primary_energies']['E_exch'][primary_unit]}
                    if debug:
                        print(f"  Derived physical energy E_exch from primary energies")
                
                # Derive E_LM from primary energies if available
                if 'E_LM' in result['primary_energies'] and primary_unit in result['primary_energies']['E_LM']:
                    result['physical_energies']['E_LM'] = {primary_unit: result['primary_energies']['E_LM'][primary_unit]}
                    if debug:
                        print(f"  Derived physical energy E_LM from primary energies")
                
                # Derive E_def,tot as sum of dEA and dEB if available
                if 'dEA' in result['primary_energies'] and 'dEB' in result['primary_energies'] and \
                   primary_unit in result['primary_energies']['dEA'] and primary_unit in result['primary_energies']['dEB']:
                    dEA = result['primary_energies']['dEA'][primary_unit]
                    dEB = result['primary_energies']['dEB'][primary_unit]
                    result['physical_energies']['E_def,tot'] = {primary_unit: dEA + dEB}
                    if debug:
                        print(f"  Derived physical energy E_def,tot from primary energies")
                
                # Derive E_int as sum of all components if available
                components_available = True
                component_sum = 0.0
                
                for component in ['E_elst', 'E_exch', 'E_LM']:
                    if component in result['physical_energies'] and primary_unit in result['physical_energies'][component]:
                        component_sum += result['physical_energies'][component][primary_unit]
                    elif component in result['primary_energies'] and primary_unit in result['primary_energies'][component]:
                        component_sum += result['primary_energies'][component][primary_unit]
                    else:
                        components_available = False
                        break
                
                if 'E_def,tot' in result['physical_energies'] and primary_unit in result['physical_energies']['E_def,tot']:
                    component_sum += result['physical_energies']['E_def,tot'][primary_unit]
                elif components_available and 'dEA' in result['primary_energies'] and 'dEB' in result['primary_energies'] and \
                     primary_unit in result['primary_energies']['dEA'] and primary_unit in result['primary_energies']['dEB']:
                    component_sum += result['primary_energies']['dEA'][primary_unit] + result['primary_energies']['dEB'][primary_unit]
                else:
                    components_available = False
                
                if components_available:
                    result['physical_energies']['E_int'] = {primary_unit: component_sum}
                    if debug:
                        print(f"  Derived physical energy E_int from component sum")
        
        # Extract convergence information
        converged_pattern = r'Converged:\s+([A-Za-z]+)'
        iterations_pattern = r'Iterations:\s+(\d+)'
        
        converged_match = re.search(converged_pattern, content)
        iterations_match = re.search(iterations_pattern, content)
        
        if converged_match:
            result['metadata']['converged'] = converged_match.group(1).lower() == 'true'
            if debug:
                print(f"Found convergence status: {result['metadata']['converged']}")
        
        if iterations_match:
            result['metadata']['iterations'] = int(iterations_match.group(1))
            if debug:
                print(f"Found iterations: {result['metadata']['iterations']}")
    
    except Exception as e:
        if debug:
            print(f"Error extracting energy components from {file_path}: {str(e)}")
    
    # Final check for required metadata
    if 'system_name' not in result['metadata'] or result['metadata']['system_name'] is None:
        result['metadata']['system_name'] = "Unknown"
        if debug:
            print(f"Warning: System name not found, using 'Unknown'")
    
    if 'method' not in result['metadata'] or result['metadata']['method'] is None:
        result['metadata']['method'] = "FQ"  # Default to FQ
        if debug:
            print(f"Warning: Method not found, using default 'FQ'")
    
    # Map common key variations to standard keys for primary energies
    key_mapping_primary = {
        'Deformation energy for monomer A': 'dEA',
        'Deformation energy for [A]': 'dEA',
        'Deformation energy for monomer B': 'dEB',
        'Deformation energy for [B]': 'dEB',
        'First-order electrostatic energy': 'E1_elst',
        'First-order exchange energy': 'E1_exch',
        'Converged electrostatic energy': 'E_elst',
        'Converged exchange energy': 'E_exch',
        'Landshoff and Murrell energy': 'E_LM',
        'Landshoff-Meister energy': 'E_LM',
        'Landshoff and Murrell for [A]': 'E_LMA',
        'Landshoff and Murrell for [B]': 'E_LMB',
        'Landshoff energy for [A]': 'E_FA',
        'Landshoff energy for [B]': 'E_FB',
        'Murrell energy for [A]': 'E_WA',
        'Murrell energy for [B]': 'E_WB'
    }
    
    # Standardize primary energy keys
    standardized_primary = {}
    for key, values in result['primary_energies'].items():
        std_key = key_mapping_primary.get(key, key)
        standardized_primary[std_key] = values
    
    result['primary_energies'] = standardized_primary
    
    # Map common key variations to standard keys for physical energies
    key_mapping_physical = {
        'E_elst': 'E_elst',
        'E_{\\rm elst}': 'E_elst',
        'E_{\rm elst}': 'E_elst',
        'Eelst': 'E_elst',
        'E elst': 'E_elst',
        'E_exch': 'E_exch',
        'E_{\\rm exch}': 'E_exch',
        'E_{\rm exch}': 'E_exch',
        'Eexch': 'E_exch',
        'E exch': 'E_exch',
        'E_def,tot': 'E_def,tot',
        'E_{\\rm def,tot}': 'E_def,tot',
        'E_{\rm def,tot}': 'E_def,tot',
        'Edef,tot': 'E_def,tot',
        'E def,tot': 'E_def,tot',
        'E_ind,inter': 'E_ind,inter',
        'E_{\\rm ind,inter}': 'E_ind,inter',
        'E_{\rm ind,inter}': 'E_ind,inter',
        'Eind,inter': 'E_ind,inter',
        'E ind,inter': 'E_ind,inter',
        'E_ind,tot': 'E_ind,tot',
        'E_{\\rm ind,tot}': 'E_ind,tot',
        'E_{\rm ind,tot}': 'E_ind,tot',
        'Eind,tot': 'E_ind,tot',
        'E ind,tot': 'E_ind,tot',
        'E_int': 'E_int',
        'E_{\\rm int}': 'E_int',
        'E_{\rm int}': 'E_int',
        'Eint': 'E_int',
        'E int': 'E_int'
    }
    
    # Standardize physical energy keys
    standardized_physical = {}
    for key, values in result['physical_energies'].items():
        std_key = key_mapping_physical.get(key, key)
        standardized_physical[std_key] = values
    
    result['physical_energies'] = standardized_physical
    
    # Convert primary energies to all standard units if not already present
    for component, values in result['primary_energies'].items():
        if 'mH' in values:
            # Convert to other units if not present
            if 'kJ/mol' not in values:
                values['kJ/mol'] = values['mH'] * 2.625
            if 'kcal/mol' not in values:
                values['kcal/mol'] = values['mH'] * 0.627
            if 'meV' not in values:
                values['meV'] = values['mH'] * 27.211
    
    # Convert physical energies to all standard units if not already present
    for component, values in result['physical_energies'].items():
        if 'mH' in values:
            # Convert to other units if not present
            if 'kJ/mol' not in values:
                values['kJ/mol'] = values['mH'] * 2.625
            if 'kcal/mol' not in values:
                values['kcal/mol'] = values['mH'] * 0.627
            if 'meV' not in values:
                values['meV'] = values['mH'] * 27.211
    
    if debug:
        print(f"Final extracted data:")
        print(f"  Metadata: {result['metadata']}")
        print(f"  Primary energies: {result['primary_energies']}")
        print(f"  Physical energies: {result['physical_energies']}")
    
    return result

def collect_scan_data(output_files, debug=False):
    """
    Collect scan data from multiple output files.
    
    Parameters:
    -----------
    output_files : list
        List of paths to SART-FQ output files
    
    debug : bool, default=False
        Whether to print debug information during collection
        
    Returns:
    --------
    dict
        Dictionary containing collected scan data
    """
    if debug:
        print(f"Collecting scan data from {len(output_files)} files")
    
    # Initialize result dictionary
    scan_data = {
        'distances': [],
        'primary_energies': defaultdict(lambda: defaultdict(list)),
        'physical_energies': defaultdict(lambda: defaultdict(list)),
        'distance_map': {},
        'system_name': None,
        'method': None,
        'units': set(),
        'metadata': defaultdict(list)  # Store metadata like convergence and iterations
    }
    
    # Process each output file
    for file_path in output_files:
        if debug:
            print(f"\nProcessing file: {file_path}")
        
        # Extract energy components
        result = extract_energy_components_from_file(file_path, debug=debug)
        
        # Skip if no separation found
        if 'separation' not in result['metadata']:
            if debug:
                print(f"No separation found in {file_path}, skipping")
            continue
        
        # Get separation
        separation = result['metadata']['separation']
        
        # Store system name and method if not already set
        if scan_data['system_name'] is None and 'system_name' in result['metadata']:
            scan_data['system_name'] = result['metadata']['system_name']
            if debug:
                print(f"Setting system name to: {scan_data['system_name']}")
        
        if scan_data['method'] is None and 'method' in result['metadata']:
            scan_data['method'] = result['metadata']['method']
            if debug:
                print(f"Setting method to: {scan_data['method']}")
        
        # Add separation to list if not already present
        if separation not in scan_data['distances']:
            scan_data['distances'].append(separation)
        
        # Store distance to file mapping
        scan_data['distance_map'][separation] = file_path
        
        # Store metadata
        if 'converged' in result['metadata']:
            scan_data['metadata']['converged'].append((separation, result['metadata']['converged']))
        
        if 'iterations' in result['metadata']:
            scan_data['metadata']['iterations'].append((separation, result['metadata']['iterations']))
        
        # Store primary energy components
        for key, values in result['primary_energies'].items():
            for unit, value in values.items():
                scan_data['primary_energies'][key][unit].append((separation, value))
                scan_data['units'].add(unit)
        
        # Store physical energy components
        for key, values in result['physical_energies'].items():
            for unit, value in values.items():
                scan_data['physical_energies'][key][unit].append((separation, value))
                scan_data['units'].add(unit)
    
    # Sort distances
    scan_data['distances'] = sorted(scan_data['distances'])
    
    # Sort data points by distance
    for key in scan_data['primary_energies']:
        for unit in scan_data['primary_energies'][key]:
            scan_data['primary_energies'][key][unit] = sorted(scan_data['primary_energies'][key][unit], key=lambda x: x[0])
    
    for key in scan_data['physical_energies']:
        for unit in scan_data['physical_energies'][key]:
            scan_data['physical_energies'][key][unit] = sorted(scan_data['physical_energies'][key][unit], key=lambda x: x[0])
    
    # Sort metadata
    for key in scan_data['metadata']:
        scan_data['metadata'][key] = sorted(scan_data['metadata'][key], key=lambda x: x[0])
    
    # Ensure all standard units are available
    standard_units = ['mH', 'kJ/mol', 'kcal/mol', 'meV']
    for unit in standard_units:
        scan_data['units'].add(unit)
    
    if debug:
        print(f"\nCollected scan data:")
        print(f"  System name: {scan_data['system_name']}")
        print(f"  Method: {scan_data['method']}")
        print(f"  Distances: {scan_data['distances']}")
        print(f"  Units: {scan_data['units']}")
        print(f"  Primary energy components: {list(scan_data['primary_energies'].keys())}")
        print(f"  Physical energy components: {list(scan_data['physical_energies'].keys())}")
        print(f"  Metadata: {list(scan_data['metadata'].keys())}")
    
    return scan_data

def plot_scan_results(scan_data, output_dir='.', debug=False):
    """
    Plot scan results.
    
    Parameters:
    -----------
    scan_data : dict
        Dictionary containing scan data from collect_scan_data()
    
    output_dir : str, default='.'
        Directory to save the plots
    
    debug : bool, default=False
        Whether to print debug information during plotting
        
    Returns:
    --------
    dict
        Dictionary containing paths to the generated plots
    """
    import os
    import matplotlib.pyplot as plt

    if debug:
        print(f"Plotting scan results")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get system name and method
    system_name = scan_data.get('system_name', 'Unknown')
    method = scan_data.get('method', 'FQ')
    
    # Get min and max distances for filename
    distances = scan_data.get('distances', [])
    if not distances:
        if debug:
            print("No distances found, cannot plot")
        return {}
    
    min_distance = min(distances)
    max_distance = max(distances)
    distance_range = f"{min_distance:.2f}-{max_distance:.2f}"
    
    # Define standard units and their labels for filename cleanup
    standard_units = {
        'mH': 'mEh',
        'kJ/mol': 'kJ_mol',
        'kcal/mol': 'kcal_mol',
        'meV': 'meV'
    }
    
    # Ensure all standard units are available    
    available_units = list(scan_data.get('units', set()))
    
    # If no units found, use default
    if not available_units:
        available_units = ['mH']
    
    # Plot primary energies
    primary_plots = {}
    for unit in available_units:
        # Clean unit name for filename
        clean_unit = standard_units.get(unit, unit.replace('/', '_').replace(' ', '_'))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each primary energy component
        components_plotted = False
        for key, unit_data in scan_data['primary_energies'].items():
            if unit in unit_data and unit_data[unit]:
                # Extract x and y values
                x_values = [point[0] for point in unit_data[unit]]
                y_values = [point[1] for point in unit_data[unit]]
                
                # Plot each component
                ax.plot(x_values, y_values, marker='o', linestyle='-', label=key)
                components_plotted = True
        
        # Skip if no components were plotted
        if not components_plotted:
            plt.close(fig)
            continue
        
        # Set labels and title
        ax.set_xlabel('Distance (bohr)')
        ax.set_ylabel(f'Energy ({unit})')
        ax.set_title(f'Primary Energy Components vs Distance ({method}) for {system_name}')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Generate filename and save figure
        filename = f"primary_energies_vs_distance_{method}_{distance_range}_{system_name}_{clean_unit}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, format="png", dpi=300)
        plt.close(fig)
        
        primary_plots[unit] = filepath
        
        if debug:
            print(f"Primary energies plot for unit {unit} saved as {filepath}")
    
    # Plot physical energies (all components) overlayed
    physical_plots = {}
    for unit in available_units:
        # Clean unit name for filename
        clean_unit = standard_units.get(unit, unit.replace('/', '_').replace(' ', '_'))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each physical energy component
        components_plotted = False
        for key, unit_data in scan_data['physical_energies'].items():
            if unit in unit_data and unit_data[unit]:
                # Extract x and y values
                x_values = [point[0] for point in unit_data[unit]]
                y_values = [point[1] for point in unit_data[unit]]
                
                # Plot and flag as plotted
                ax.plot(x_values, y_values, marker='o', linestyle='-', label=key)
                components_plotted = True
        
        # Skip if nothing was plotted
        if not components_plotted:
            plt.close(fig)
            continue
        
        # Set labels and title
        ax.set_xlabel('Distance (bohr)')
        ax.set_ylabel(f'Energy ({unit})')
        ax.set_title(f'Physical Energy Components vs Distance ({method}) for {system_name}')
        ax.legend()
        ax.grid(True)
        
        # Generate filename and save the figure
        filename = f"physical_energies_vs_distance_{method}_{distance_range}_{system_name}_{clean_unit}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, format="png", dpi=300)
        plt.close(fig)
        
        physical_plots[unit] = filepath
        
        if debug:
            print(f"Physical energies plot for unit {unit} saved as {filepath}")
    
    # -----------------------------
    # Additional Plot: Interaction Energy in kJ/mol only
    # -----------------------------
    interaction_plot = {}
    # We require kJ/mol to be available; if not, nothing to do.
    if 'kJ/mol' in available_units:
        unit = 'kJ/mol'
        clean_unit = standard_units.get(unit, unit)
        # Check that E_int is present in the physical energies section
        if 'int' in scan_data['physical_energies']:
            unit_data = scan_data['physical_energies']['int']
            if unit in unit_data and unit_data[unit]:
                # Extract the x (distance) and y (E_int) values
                x_values = [point[0] for point in unit_data[unit]]
                y_values = [point[1] for point in unit_data[unit]]
                
                # Create a new figure for E_int
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x_values, y_values, marker='o', linestyle='-', color='red', label='E_int')
                ax.set_xlabel('Distance (bohr)')
                ax.set_ylabel(f'Interaction Energy ({unit})')
                ax.set_title(f'Interaction Energy vs Distance ({method}) for {system_name} [{unit}]')
                ax.legend()
                ax.grid(True)
                
                # Generate filename and save figure
                filename = f"interaction_energy_vs_distance_{method}_{distance_range}_{system_name}_{clean_unit}.png"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, format="png", dpi=300)
                plt.close(fig)
                
                interaction_plot[unit] = filepath
                
                if debug:
                    print(f"Interaction energy plot (kJ/mol) saved as {filepath}")
            else:
                if debug:
                    print("No E_int data available in kJ/mol within physical energies.")
        else:
            if debug:
                print("E_int not found in physical energies; cannot plot interaction energy.")
    
    # Return a dictionary containing all plot file paths
    result = {
        'primary_energies': primary_plots,
        'physical_energies': physical_plots,
        'interaction_energy': interaction_plot  # New key for the separate interaction energy plot
    }
    
    return result


def process_scan_results(output_files, output_dir='.', debug=False):
    """
    Process scan results from multiple output files.
    
    Parameters:
    -----------
    output_files : list
        List of paths to SART-FQ output files
    
    output_dir : str, default='.'
        Directory to save the plots and CSV files
    
    debug : bool, default=False
        Whether to print debug information during processing
        
    Returns:
    --------
    dict
        Dictionary containing paths to the generated files
    """
    if debug:
        print(f"Processing scan results from {len(output_files)} files")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect scan data
    scan_data = collect_scan_data(output_files, debug=debug)
    
    # Get system name and method
    system_name = scan_data.get('system_name', 'Unknown')
    method = scan_data.get('method', 'FQ')
    
    # Get min and max distances for filename
    distances = scan_data.get('distances', [])
    if not distances:
        if debug:
            print("No distances found, cannot process")
        return {}
    
    min_distance = min(distances)
    max_distance = max(distances)
    distance_range = f"{min_distance:.2f}-{max_distance:.2f}"
    
    # Generate timestamp for filenames
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV file with scan summary
    csv_filename = f"scan_summary_{system_name}_{min_distance:.2f}_all_potentials_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Define standard units and their labels
    standard_units = {
        'mH': 'mEh',
        'kJ/mol': 'kJ_mol',
        'kcal/mol': 'kcal_mol',
        'meV': 'meV'
    }
    
    # Ensure all standard units are available
    available_units = list(scan_data.get('units', set()))
    
    # If no units found, use default
    if not available_units:
        available_units = ['mH']
    
    # Select a primary unit for the CSV (prefer mH if available)
    primary_unit = 'mH' if 'mH' in available_units else available_units[0]
    
    # Create CSV file
    with open(csv_path, 'w') as f:
        # Write header
        f.write(f"Distance,E_int,E_elst,E_exch,E_LM,dEA,dEB,Converged,Iterations\n")
        
        # Write data rows
        for distance in distances:
            # Start with distance
            row = f"{distance:.2f}"
            
            # Add E_int (from physical energies)
            e_int_value = None
            if 'E_int' in scan_data['physical_energies']:
                if primary_unit in scan_data['physical_energies']['E_int']:
                    for d, v in scan_data['physical_energies']['E_int'][primary_unit]:
                        if abs(d - distance) < 1e-6:
                            e_int_value = v
                            break
            row += f",{e_int_value if e_int_value is not None else 0.0}"
            
            # Add E_elst (from physical energies)
            e_elst_value = None
            if 'E_elst' in scan_data['physical_energies']:
                if primary_unit in scan_data['physical_energies']['E_elst']:
                    for d, v in scan_data['physical_energies']['E_elst'][primary_unit]:
                        if abs(d - distance) < 1e-6:
                            e_elst_value = v
                            break
            row += f",{e_elst_value if e_elst_value is not None else 0.0}"
            
            # Add E_exch (from physical energies)
            e_exch_value = None
            if 'E_exch' in scan_data['physical_energies']:
                if primary_unit in scan_data['physical_energies']['E_exch']:
                    for d, v in scan_data['physical_energies']['E_exch'][primary_unit]:
                        if abs(d - distance) < 1e-6:
                            e_exch_value = v
                            break
            row += f",{e_exch_value if e_exch_value is not None else 0.0}"
            
            # Add E_LM (from primary energies)
            e_lm_value = None
            if 'E_LM' in scan_data['primary_energies']:
                if primary_unit in scan_data['primary_energies']['E_LM']:
                    for d, v in scan_data['primary_energies']['E_LM'][primary_unit]:
                        if abs(d - distance) < 1e-6:
                            e_lm_value = v
                            break
            row += f",{e_lm_value if e_lm_value is not None else 0.0}"
            
            # Add dEA (from primary energies)
            dea_value = None
            if 'dEA' in scan_data['primary_energies']:
                if primary_unit in scan_data['primary_energies']['dEA']:
                    for d, v in scan_data['primary_energies']['dEA'][primary_unit]:
                        if abs(d - distance) < 1e-6:
                            dea_value = v
                            break
            row += f",{dea_value if dea_value is not None else 0.0}"
            
            # Add dEB (from primary energies)
            deb_value = None
            if 'dEB' in scan_data['primary_energies']:
                if primary_unit in scan_data['primary_energies']['dEB']:
                    for d, v in scan_data['primary_energies']['dEB'][primary_unit]:
                        if abs(d - distance) < 1e-6:
                            deb_value = v
                            break
            row += f",{deb_value if deb_value is not None else 0.0}"
            
            # Add converged status
            converged = True  # Default to True if not found
            for d, v in scan_data['metadata'].get('converged', []):
                if abs(d - distance) < 1e-6:
                    converged = v
                    break
            row += f",{converged}"
            
            # Add iterations
            iterations = 0  # Default to 0 if not found
            for d, v in scan_data['metadata'].get('iterations', []):
                if abs(d - distance) < 1e-6:
                    iterations = v
                    break
            row += f",{iterations}"
            
            f.write(row + "\n")
    
    if debug:
        print(f"Scan summary saved to: {csv_path}")
    
    # Create detailed CSV with all energy components and units
    detailed_csv_filename = f"energy_data_{system_name}_{min_distance:.2f}_all_potentials_{timestamp}.csv"
    detailed_csv_path = os.path.join(output_dir, detailed_csv_filename)
    
    with open(detailed_csv_path, 'w') as f:
        # Write header
        header = "Distance"
        for key in scan_data['primary_energies']:
            for unit in available_units:
                if unit in scan_data['primary_energies'][key]:
                    clean_unit = standard_units.get(unit, unit.replace('/', '_').replace(' ', '_'))
                    header += f",{key}_{clean_unit}"
        
        for key in scan_data['physical_energies']:
            for unit in available_units:
                if unit in scan_data['physical_energies'][key]:
                    clean_unit = standard_units.get(unit, unit.replace('/', '_').replace(' ', '_'))
                    header += f",{key}_{clean_unit}"
        
        f.write(header + "\n")
        
        # Write data rows
        for distance in distances:
            row = f"{distance:.2f}"
            
            # Add primary energies
            for key in scan_data['primary_energies']:
                for unit in available_units:
                    if unit in scan_data['primary_energies'][key]:
                        # Find value for this distance
                        value = None
                        for d, v in scan_data['primary_energies'][key][unit]:
                            if abs(d - distance) < 1e-6:
                                value = v
                                break
                        
                        if value is not None:
                            row += f",{value}"
                        else:
                            row += ",0.0"
            
            # Add physical energies
            for key in scan_data['physical_energies']:
                for unit in available_units:
                    if unit in scan_data['physical_energies'][key]:
                        # Find value for this distance
                        value = None
                        for d, v in scan_data['physical_energies'][key][unit]:
                            if abs(d - distance) < 1e-6:
                                value = v
                                break
                        
                        if value is not None:
                            row += f",{value}"
                        else:
                            row += ",0.0"
            
            f.write(row + "\n")
    
    if debug:
        print(f"Detailed energy data saved to: {detailed_csv_path}")
    
    # Plot scan results
    plots = plot_scan_results(scan_data, output_dir=output_dir, debug=debug)
    
    # Print summary of generated plots
    if debug:
        print("\nGenerated scan plots:")
        
        if 'primary_energies' in plots:
            print("  primary_energies:")
            for unit, path in plots['primary_energies'].items():
                print(f"    {unit}: {path}")
        
        if 'physical_energies' in plots:
            print("  physical_energies:")
            for unit, path in plots['physical_energies'].items():
                print(f"    {unit}: {path}")
    
    # Return paths to generated files
    result = {
        'scan_summary': csv_path,
        'energy_data': detailed_csv_path,
        'plots': plots
    }
    
    return result


# The list of energy keys we want to collect from each single-point result.
# These must match the keys returned by sart_data_print.
ENERGY_KEYS = ["E_int", "E_elst", "E_exch", "E_LM"]

def save_scan_data_to_csv(scan_results, csv_filename):
    """
    Given a dictionary of scan results (key: distance; value: energy dictionary obtained by sart_data_print)
    extract energies of interest and save them in a CSV file.
    
    The CSV file will have the header:
      Distance, E_int, E_elst, E_exch, E_LM

    Parameters:
      scan_results (dict): Keys should be distances (float) and values are dictionaries of energies.
      csv_filename (str): Full path of the CSV file to be created.
    """
    header = ['Distance'] + ENERGY_KEYS
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for dist in sorted(scan_results.keys()):
            result = scan_results[dist]
            # Extract the energies; if a key is missing we simply assign 0.0.
            row = [dist]
            for key in ENERGY_KEYS:
                row.append(result.get(key, 0.0))
            writer.writerow(row)
    print(f"Scan data saved to: {csv_filename}")

def plot_energy_vs_distance(csv_filename, output_dir="plot_images"):
    """
    Reads the CSV file produced during the scan and creates plots of interaction energy and physical energy components
    as a function of distance, converting energies to different units.
    
    For every unit in the conversion dictionary (mH, kJ/mol, kcal/mol, meV), the function saves:
      - a plot of E_int vs. Distance
      - a multi-line plot overlaying E_elst, E_exch and E_LM vs. Distance
    
    Parameters:
      csv_filename (str): Path to the CSV file containing the scan data.
      output_dir (str): Directory where plot images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data from CSV into a DataFrame.
    df = pd.read_csv(csv_filename)
    
    distances = df['Distance']
    
    # Get conversion multipliers (assumes energies are given in Hartree).
    conv_dict = conversion_units_from_hartree(print_dict=False)
    
    # We want to create plots for each energy unit.
    # (The conversion_units_from_hartree returns keys: 'meV', 'kcal/mol', 'kJ/mol', 'mH')
    for unit, factor in conv_dict.items():
        # Convert the energies: note that the energies in the CSV are assumed to be in Hartree.
        # Multiply each column (except Distance) by the conversion factor.
        E_int_converted = df["E_int"] * factor
        E_elst_converted = df["E_elst"] * factor
        E_exch_converted = df["E_exch"] * factor
        E_LM_converted   = df["E_LM"]   * factor
        
        # -----------------------------------------------
        # Plot 1: E_int vs. Distance
        plt.figure(figsize=(6,4))
        plt.plot(distances, E_int_converted, marker='o', linestyle='-', color='blue')
        plt.xlabel("Distance (bohr)")
        plt.ylabel(f"Interaction Energy ({unit})")
        plt.title(f"E_int vs Distance [{unit}]")
        plt.grid(True)
        plt.tight_layout()
        int_filename = os.path.join(output_dir, f"E_int_vs_distance_{unit}.png")
        plt.savefig(int_filename)
        plt.close()
        print(f"Saved: {int_filename}")
        
        # -----------------------------------------------
        # Plot 2: Physical energy components vs. Distance (overlayed)
        plt.figure(figsize=(6,4))
        plt.plot(distances, E_elst_converted, marker='s', linestyle='-', label=f"E_elst ({unit})")
        plt.plot(distances, E_exch_converted, marker='^', linestyle='-', label=f"E_exch ({unit})")
        plt.plot(distances, E_LM_converted, marker='d', linestyle='-', label=f"E_LM ({unit})")
        plt.xlabel("Distance (bohr)")
        plt.ylabel(f"Energy ({unit})")
        plt.title(f"Physical Energy Components vs Distance [{unit}]")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        comp_filename = os.path.join(output_dir, f"physical_components_vs_distance_{unit}.png")
        plt.savefig(comp_filename)
        plt.close()
        print(f"Saved: {comp_filename}")

