"""
Functions for plotting SART-FQ energy components.

This module provides functions for plotting energy components from SART-FQ
calculations. In addition to the existing function that plots a list of energy
data provided in memory, we add new functions that read a CSV file (in a standardized
format) and produce separate graphs for each energy component as a function of iteration.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_energy_iterations_direct(energy_iterations_data, separation=None, system_name=None, filename=None, verbose=False):
    """
    Plot energy components vs iteration directly using energy_iterations_data.
    
    Parameters:
      energy_iterations_data : list
          Either a list of lists (each row: [Iteration, dEA, dEB, E_elst, E_exch, E_LM, E_int])
          OR a list of dictionaries with keys including:
              "Iteration", "dEA", "dEB", "E_elst", "E_exch", "E_LM", "E_int"
      separation : float, optional
          Separation distance in bohr for the plot title.
      system_name : str, optional
          The system name for the plot title.
      filename : str, optional
          Path to save the plot(s).
      verbose : bool, optional
          If True, prints verbose messages.
    
    Returns:
      A dictionary mapping plot names to saved file paths.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    # Check if energy_iterations_data is a list of dictionaries
    if (isinstance(energy_iterations_data, list) and len(energy_iterations_data) > 0 and
            isinstance(energy_iterations_data[0], dict)):
        # Convert list-of-dictionaries to a DataFrame.
        df = pd.DataFrame(energy_iterations_data)
        iterations = df["Iteration"].values
        dEA = df["dEA"].values
        dEB = df["dEB"].values
        E_elst = df["E_elst"].values
        E_exch = df["E_exch"].values
        E_LM = df["E_LM"].values
        E_int = df["E_int"].values
    else:
        # Otherwise, assume it is a list of lists.
        data = np.array(energy_iterations_data)
        # If the array is 1D, attempt to reshape it
        if data.ndim == 1:
            try:
                ncols = len(energy_iterations_data[0])
                data = data.reshape(-1, ncols)
            except Exception as e:
                raise ValueError("Could not reshape energy_iterations_data into a 2D array.") from e
        iterations = data[:, 0]
        dEA = data[:, 1]
        dEB = data[:, 2]
        E_elst = data[:, 3]
        E_exch = data[:, 4]
        E_LM = data[:, 5]
        E_int = data[:, 6]
    
    # ----- Plot 1: Energy Components vs Iteration -----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(iterations, dEA, 'o-', label='dEA')
    ax1.plot(iterations, dEB, 's-', label='dEB')
    ax1.plot(iterations, E_elst, '^-', label='E_elst')
    ax1.plot(iterations, E_exch, 'v-', label='E_exch')
    ax1.plot(iterations, E_LM, 'D-', label='E_LM')
    ax1.plot(iterations, E_int, '*-', label='E_int')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy (a.u.)')
    title = 'Energy Components vs Iteration'
    if system_name:
        title += f' for {system_name}'
    if separation is not None:
        title += f' at {separation:.2f} bohr'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # ----- Plot 2: Total Interaction Energy Change -----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    E_int_change = np.diff(E_int)  # E_int difference between consecutive iterations.
    ax2.plot(iterations[1:], E_int_change, 'o-', label='ΔE_int')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('ΔE_int (a.u.)')
    title = 'Change in Total Interaction Energy vs Iteration'
    if system_name:
        title += f' for {system_name}'
    if separation is not None:
        title += f' at {separation:.2f} bohr'
    ax2.set_title(title)
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('symlog')
    
    result = {}
    if filename:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        components_filename = filename
        fig1.savefig(components_filename, format="png", dpi=300)
        result['components'] = components_filename
        if verbose:
            print(f"Energy components plot saved as {components_filename}")
        changes_filename = filename.replace('.png', '_E_int_change.png')
        fig2.savefig(changes_filename, format="png", dpi=300)
        result['E_int_change'] = changes_filename
        if verbose:
            print(f"Total interaction energy change plot saved as {changes_filename}")
    
    plt.close(fig1)
    plt.close(fig2)
    
    return result

def plot_iterations_from_csv(csv_file, components=None, output_dir=None, verbose=False):
    """
    Read a CSV file (with standardized iteration energy data) and create separate plots for each energy component versus iteration.
    
    The CSV file is expected to have the following columns (all in Hartree):
      "Iteration", "dEA", "dEB", "E1_elst", "E1_exch", "E_elst", "E_exch", "E_LM", 
      "E_LMA", "E_LMB", "E_int", "E_def_tot[A]", "E_def_tot[B]"
    
    Parameters:
    -----------
    csv_file : str
        Path to the standardized CSV file with iteration data.
    
    components : list, optional
        List of column names (energy components) to be plotted.
        If None, all columns except "Iteration" are plotted.
    
    output_dir : str, optional
        Directory to save the individual plots. If None, the plots will be saved in the same directory as the CSV file.
    
    verbose : bool, default=False
        Whether to print verbose output.
    
    Returns:
    --------
    dict
        Dictionary mapping each energy component to its saved plot file path.
    """
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)
    
    if components is None:
        # Plot all columns except "Iteration"
        components = [col for col in df.columns if col != "Iteration"]
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_file))
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = {}
    
    # For each component, generate a plot of Energy vs Iteration
    for comp in components:
        plt.figure(figsize=(8, 5))
        plt.plot(df["Iteration"], df[comp], marker='o', linestyle='-', label=comp)
        plt.xlabel("Iteration")
        plt.ylabel(f"{comp} (a.u.)")
        plt.title(f"{comp} vs Iteration")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Generate a filename for the plot; for example: "Iteration_<component>.png"
        fname = f"Iteration_{comp}.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=300)
        if verbose:
            print(f"Saved plot for {comp} as {fpath}")
        saved_plots[comp] = fpath
        plt.close()
    
    return saved_plots


def plot_differences_from_csv(csv_file, components=None, output_dir=None, verbose=False):
    """
    Read the CSV file (with standardized iteration energy data) and create separate plots of differences (energy change between consecutive iterations) 
    for each specified energy component.
    
    Parameters:
    -----------
    csv_file : str
        Path to the standardized CSV file.
    
    components : list, optional
        List of column names to generate difference plots for.
        If None, all energy components (all columns except "Iteration") are used.
    
    output_dir : str, optional
        Directory to save the difference plots.
    
    verbose : bool, default=False
        Whether to print verbose messages.
    
    Returns:
    --------
    dict
        Dictionary mapping each energy component to its saved difference plot file path.
    """
    # Read data
    df = pd.read_csv(csv_file)
    
    if components is None:
        components = [col for col in df.columns if col != "Iteration"]
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_file))
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    saved_diff_plots = {}
    
    iterations = df["Iteration"].values
    # For each component, compute the difference between consecutive iterations and plot
    for comp in components:
        # Compute differences (np.diff drops the first data point)
        diff_data = np.diff(df[comp].values)
        plt.figure(figsize=(8, 5))
        plt.plot(iterations[1:], diff_data, marker='o', linestyle='-', label=f"{comp} change")
        plt.xlabel("Iteration")
        plt.ylabel(f"Delta {comp} (a.u.)")
        plt.title(f"Change in {comp} vs Iteration")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        fname = f"Iteration_{comp}_diff.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=300)
        if verbose:
            print(f"Saved difference plot for {comp} as {fpath}")
        saved_diff_plots[comp] = fpath
        plt.close()
    
    return saved_diff_plots
