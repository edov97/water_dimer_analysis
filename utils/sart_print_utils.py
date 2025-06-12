"""
Functions to analyse and print the SART energies and other quantities.

"""
from scipy import constants


def conversion_units_from_hartree(print_dict:bool=False):
    """ 
    Define conversion units from Hatree to common units. Return a dictionary.

    Optional input: 
    bool print_dict: if True, will print dictionary
    """
    cal = constants.calorie
    A   = constants.Avogadro
    # these are tuples:
    auJ  = constants.physical_constants['hartree-joule relationship']
    aueV = constants.physical_constants['hartree-electron volt relationship']

    # Compute constants to convert au in physically meaningful quantities.
    au_to_kJ = auJ[0]/1000.0
    au_to_kcal = au_to_kJ/cal
    au_to_meV  = aueV[0]*1000.0
    au_to_mH = 1000.0

    au_to_kcal_per_mol = au_to_kcal * A
    au_to_kJ_per_mol   = au_to_kJ * A

    au_to_units = {}
    au_to_units['meV'] = au_to_meV
    au_to_units['kcal/mol'] = au_to_kcal_per_mol
    au_to_units['kJ/mol'] = au_to_kJ_per_mol
    au_to_units['mH'] = au_to_mH

    if (print_dict):
        print('Conversion from Hartree to specified units:')
        print(au_to_units)

    return au_to_units

def convert_energy(energy:float,units_to_convert_to:list,conv_dict:dict):
    """
    Take an energy and convert it to all specified units
    using conversion multipliers supplied as a dictionary.

    float energy :: input energy
    list units_to_convert_to :: list of units to convert to
    dict conv_dict :: dictionary of conversion multipliers

    to convert: energy*conv_multiplier

    returns a dictionary.

    If conversion multipliers are not present, the energy is set to 0.0

    """
    conv_energy_dict = {}
    for unit in units_to_convert_to:
        if unit in conv_dict:
            mul = conv_dict[unit]
            conv_energy_dict[unit] = energy*mul
        else:
            print(f"\n\n WARNING: Conversion for unit: {unit} is not in conversion dictionary.\n\n")
            conv_energy_dict[unit] = 0.0

    return conv_energy_dict

def sart_data_print(sart_dict:dict):
    """
    Takes the SART energy dictionary and prints an analysis with units conversions.

    Returns dictionary of SART energies.

    Assumes a dimer.

    Input units : Hartree

    The dictionary is expected to contain the following:

    Job identification data:
    'job_index' : index of job/calculation
    'job_name' : name of job/calculation
    'job_method' : method used
    'job_basis' : basis set used
    'job_identifier' : unique identifier (mostly for databases)

    Energy data:
    'dEA' : deformation energy of monomer A
    'dEB' : deformation energy of monomer B
    'E1_elst' : Electrostatic energy in iteration 1.
    'E1_exch' : Exchange energy in iteration 1.
    'E_elst' : Converged electrostatic energy.
    'E_exch' : Converged exchange energy.
    'E_LM' : The Landshoff and Murrell energy.

    Other entries will be ignored or used optionally. 

    Definitions:

    * E_def,tot = dEA + dEB + E_LM : total deformation energy
    * E_elst,rel = E_elst - E1_elst : intermolecular electrostatic relaxation energy
    * E_exch,rel = E_exch - E1_exch : intermolecular exchange relaxation energy
    * E_ind,inter = DE_elst + DE_exch : intermolecular induction energy
    * E_ind = E_ind-inter + E_def,tot : induction energy
    * E_int = E1_elst + E1_exch + E_ind : interaction energy

    Optional, if terms present:
    Note that E_LM = E_LMA + E_LMB == E_FA + E_FB + E_WA + E_WB
    * E_LM[A] = E_FA + E_WA : LM term for monomer A
    * E_LM[B] = E_FB + E_WB : LM term for monomer B
    * E_def,tot[A] = dEA + E_LM[A] : total deformation energy for monomer A
    * E_def,tot[B] = dEB + E_LM[B] : total deformation energy for monomer B


    """
    sart_data_dict = {}

    EPS_SMALL = 1e-14

    au_to_units = conversion_units_from_hartree(print_dict=True) 

    # Job identification data:
    job_data_items = [ 'job_index', 'job_name', 'job_method', 'job_basis', 'job_identifier']
    job_data_item_defaults = [ 0, 'NONE', 'NONE', 'NONE', 'NONE']
    for key, default in zip(job_data_items,job_data_item_defaults):
        if key in sart_dict:
            sart_data_dict[key] = sart_dict[key]
        else:
            sart_data_dict[key] = default

    # Define the output units here.
    # options: mH, kJ/mol, kcal/mol, meV
    units_list = ['mH','kJ/mol','kcal/mol','meV']

    # These refer to the position of the terms in the energy definitions:
    desc_indx  = 0
    latex_indx = 1
    add_indx   = 2
    sub_indx   = 3

    # Definitions of the primary energies
    # ===================================
    # These are the energies that will be passed here from the SART routine.
    #
    # Structure of the dictionary values for primary energies:
    # ('description of the energy','LaTeX expression',[key],[])
    # the two lists are included to make this compatible with the dictionary for composite energies.

    primary_energy_defs = {
            'dEA':     ('Deformation energy for [A]','E_{\\rm def}[A]',
                        ['dEA'],[],),
            'dEB':     ('Deformation energy for [B]','E_{\\rm def}[B]',
                        ['dEB'],[],),
            'E_LM':    ('Landshoff and Murrell energy','E_{\\rm LM}',
                        ['E_LM'],[],),
            'E_LMA':   ('Landshoff and Murrell for [A]','E_{\\rm LM}[A]',
                        ['E_LMA'],[],),
            'E_LMB':   ('Landshoff and Murrell for [B]','E_{\\rm LM}[B]',
                        ['E_LMB'],[],),
            'E_FA':    ('Landshoff energy for [A]','E_{\\rm L}[A]',
                        ['E_FA'],[],),
            'E_FB':    ('Landshoff energy for [B]','E_{\\rm L}[B]',
                        ['E_FB'],[],),
            'E_WA':    ('Murrell energy for [A]','E_{\\rm M}[A]',
                        ['E_FA'],[],),
            'E_WB':    ('Murrell energy for [B]','E_{\\rm M}[B]',
                        ['E_WB'],[],),
            'E1_elst': ('First-order electrostatic energy','E_{\\rm elst}^{(1)}',
                        ['E1_elst'],[],),
            'E1_exch': ('First-order exchange energy','E_{\\rm exch}^{(1)}',
                        ['E1_exch'],[],),
            'E_elst':  ('Converged electrostatic energy','E_{\\rm elst}',
                        ['E_elst'],[],),
            'E_exch':  ('Converged exchange energy','E_{\\rm exch}',
                        ['E_exch'],[],),
            }

    for key in primary_energy_defs:
        if key in sart_dict:
            sart_data_dict[key] = sart_dict[key]
        else:
            sart_data_dict[key] = 0.0

    # If LMA is 0.0, define it from FA and WA, and ditto for B:
    # This allows both FQ and SQ codes to operate as FQ code only computes total: LMA and LMB
    if sart_data_dict['E_LMA']==0.0:
        sart_data_dict['E_LMA'] = sart_data_dict['E_FA'] + sart_data_dict['E_WA']
    if sart_data_dict['E_LMB']==0.0:
        sart_data_dict['E_LMB'] = sart_data_dict['E_FB'] + sart_data_dict['E_WB']

    # Definitions of composite energies:
    # ==================================
    # The values have the structure: ('descriptive string','LaTeX expression',[keys add],[keys subtract])
    # where 
    # [keys add] : contains primary keys of quantities to be added
    # [keys subtract] : contains primary keys of quantities to be subtracted.
    #
    # This should be the only place in which changes are needed if we change the 
    # definitions for the composite (physical) energies.

    composite_energy_defs = {
            'E_def,tot':    ('Total deformation energy','E_{\\rm def,tot}',
                            ['dEA','dEB','E_LM'],[],),
            'E_def,tot[A]': ('Total deformation energy for [A]','E_{\\rm def,tot}[A]',
                            ['dEA','E_LMA'],[],),
            'E_def,tot[B]': ('Total deformation energy for [B]','E_{\\rm def,tot}[B]',
                            ['dEB','E_LMB'],[],),
            'E_elst,rel':  ('Electrostatic relaxation energy','E_{\\rm elst,rel}',
                            ['E_elst'],['E1_elst'],),
            'E_exch,rel':  ('Exchange relaxation energy','E_{\\rm exch,rel}',
                            ['E_exch'],['E1_exch'],),
            'E_ind,inter': ('Intermolecular induction energy','E_{\\rm ind,inter}',
                            ['E_elst','E_exch'],['E1_elst','E1_exch'],),
            'E_ind,tot':   ('Total induction energy','E_{\\rm ind,tot}',
                            ['dEA','dEB','E_LM','E_elst','E_exch'],['E1_elst','E1_exch'],),
            'E_int':       ('Total interaction energy','E_{\\rm int}',
                            ['dEA','dEB','E_LM','E_elst','E_exch'],[],),
            }

    # Now construct the composite energies:

    for key in composite_energy_defs:
        sart_data_dict[key] = 0.0
        add_keys = composite_energy_defs[key][add_indx]
        for add_key in add_keys:
            sart_data_dict[key] += sart_data_dict[add_key]
        sub_keys = composite_energy_defs[key][sub_indx]
        for sub_key in sub_keys:
            sart_data_dict[key] -= sart_data_dict[sub_key]

    # Checks for particular composite energies:
    # -----------------------------------------
    # (1) Monomer deformation energies can only be computed if monomer LM energies
    #     are supplied. In this case, the sum of these will be E_LM:
    #     If this is not the case, zero out the monomer deformation energies.
    LM_terms = ['E_LMA','E_LMB']
    sum_LM_terms = 0.0
    for term in LM_terms:
        sum_LM_terms += sart_data_dict[term]
    if (abs(sart_data_dict['E_LM'] - sum_LM_terms)>EPS_SMALL):
        sart_data_dict['E_def,tot[A]'] = 0.0
        sart_data_dict['E_def,tot[B]'] = 0.0

    # Merge the energy definitions for ease of use later:
    energy_defs = primary_energy_defs | composite_energy_defs

    # Write the composite energy definitions in human-readable form:
    print('#',60*'-')
    print("# SART energy definitions:")
    print('#',60*'-')
    for key in composite_energy_defs:
        energy_exp = key + ' = '
        for term in composite_energy_defs[key][add_indx]:
            energy_exp = energy_exp + ' + ' + term
        for term in composite_energy_defs[key][sub_indx]:
            energy_exp = energy_exp + ' - ' + term
        print(f"# * {composite_energy_defs[key][desc_indx]} =="\
                " ${composite_energy_defs[key][latex_indx]}$ : ")
        print(f"#       DEF:  {energy_exp}")
    print('#',60*'=','\n\n')

    print('#',140*'-')
    print("# ====================")
    print("#  SART SUMMARY TABLE")
    print("# ====================")
    print('#',80*'-')
    for key in job_data_items:
        print(f"{key.upper():20s} = {sart_data_dict[key]}")
    print('#',80*'=','\n')

    print('#',140*'-')
    print('# SART primary energies:')
    print('#',140*'-')
    header = f"# {'Description':35s} ::    LaTeX expression    :: "
    header += " ".join(f"{unit:15s} ::" for unit in units_list)
    print(f"{header}")
    print('#',140*'-')
    for key in primary_energy_defs:
        desc = energy_defs[key][desc_indx]
        latex = energy_defs[key][latex_indx]
        energy = sart_data_dict[key]
        conv_energy = convert_energy(energy,units_list,au_to_units)
        str = f"  {desc:35s} :: ${latex:20s}$ :: "
        str += " ".join(f"{conv_energy[unit]:15.8e} ::" for unit in units_list)
        print(f"{str}")
    print(f"# {140*'='}")

    # The SART physical energies (the useful ones) are printed in a two-level manner:
    # MAIN ENERGY        VALUE
    #     COMP1          VALUE
    #     COMP2          VALUE
    #     etc.
    # To achieve this we use 2D lists:
    #   [MAIN_ENERGY_KEY,[ COMP1_KEY, COMP2_KEY, etc.]]
    # These are then included into a composite 3D list and this list is processed.

    elst_energy_terms        = ['E_elst',['E1_elst','E_elst,rel']]
    exch_energy_terms        = ['E_exch',['E1_exch','E_exch,rel']]
    deftot_energy_terms      = ['E_def,tot',['dEA','dEB','E_LM']]
    indinter_energy_terms    = ['E_ind,inter',['E_elst,rel','E_exch,rel']]
    indtot_energy_terms      = ['E_ind,tot',['E_ind,inter','E_def,tot']]
    interaction_energy_terms = ['E_int',['E1_elst','E1_exch','E_ind,tot']]

    all_energy_terms = [
                        elst_energy_terms,
                        exch_energy_terms,
                        deftot_energy_terms,
                        indinter_energy_terms,
                        indtot_energy_terms,
                        interaction_energy_terms
                        ]

    print('\n#',109*'=',)
    print('# SART physical energies:')
    print('#',109*'-')
    header = f"#  {'Energy Name':30s}   "
    header += " ".join(f"{unit:15s} ::" for unit in units_list)
    print(f"{header}")
    print('#',109*'-')
    for energy_term in all_energy_terms:
        main_key  = energy_term[0]
        comp_keys = energy_term[1]
        energy = sart_data_dict[main_key]
        conv_energy = convert_energy(energy,units_list,au_to_units)
        str = f" {energy_defs[main_key][latex_indx]:30s} "
        str += " ".join(f" {conv_energy[unit]:18.8e}" for unit in units_list)
        print(f"# {109*'-'}\n{str}")
        for comp_key in comp_keys:
            energy = sart_data_dict[comp_key]
            conv_energy = convert_energy(energy,units_list,au_to_units)
            # Additional 10 spaces to offset the component energies from the main energy.
            str = f" {'':10s}{energy_defs[comp_key][latex_indx]:20s} "
            str += " ".join(f" {conv_energy[unit]:18.8e}" for unit in units_list)
            print(f"{str}")
    print(f"# {109*'='}\n\n")

    return sart_data_dict



