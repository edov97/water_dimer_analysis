import numpy as np
#=============================================================
h2o_dimer = """
    O    -0.070857701    {}     1.536397854
    H     0.824662814    {}     1.880885042
    H     0.057133569    {}     0.583212325
    --
    O     0.107681033    -0.000000000    -1.516610065
    H    -0.426206268    -0.755248708    -1.775321168
    H    -0.426206268     0.755248708    -1.775321168
    """

def format_geom(geom_str:str, interval:float, range:tuple):
    """
    Gets the psi4 defined geometry template
    and modifies it according to the arguments 
    Returns as a list of all updated geometries
    Also some specifications for psi4 options eg: unit, orient, com
    """
    geom_list = []
    lower_lim = range[0]
    upper_lim = range[1]
    sep_list = np.arange(lower_lim, upper_lim, interval)
    for i, s in enumerate(sep_list):

        # print(geom_str.format(sep_list[i], sep_list[i], sep_list[i]))
        new_geom_str = geom_str.format(sep_list[i], sep_list[i], sep_list[i])
        geom_list.append(new_geom_str)
    return geom_list

def translate_geom(monA_coords: list[list],translation_vector:list):
    """
    Translates geometry wrt another along a vector
    """
    for i,atom in enumerate(monA_coords):
        monA_coords[i][0]+= translation_vector[0]
        monA_coords[i][1]+= translation_vector[1]
        monA_coords[i][2]+= translation_vector[2]
    return monA_coords

#=========================================================================
### ASE helper utils
from pathlib import Path
from ase.atoms import Atoms
from ase.io.formats import write, read

def translate_B_from_A(geomA_file, geomB_file, 
                  disp_factors: list = np.arange(-0.5,2,0.25),
                  displacement=None,
                  to_wrie:bool = True):
    """
    Takes the monomer geometry file(xyz) as input and translates 2nd monomer 
    with displacement = disp_factor* displacement vector,
    If displacement_vector is None, the separation vector between the COM's are considered.
    --
    Writes the dimer geometries to file
    Returns list of psi4 formatted str for psi4.geometry() method
    """
    
    molA_name = str(Path(geomA_file).name).removesuffix('.xyz')
    molB_name = str(Path(geomB_file).name).removesuffix('.xyz')
    geom_folder = Path(geomB_file).parent
    molA = read(geomA_file)
    molB = read(geomB_file)

    comA = molA.get_center_of_mass()
    comB = molB.get_center_of_mass()

    if displacement is None:
        displacement_vector = comB - comA
        initial_com_diff = np.linalg.norm(displacement_vector)
        print('Initial separation:',initial_com_diff)

    elif isinstance(displacement, np.ndarray):
        displacement_vector = displacement

    elif isinstance(displacement, tuple):
        A_indx, B_indx = displacement
        displacement_vector = molB.get_positions()[B_indx]- molA.get_positions()[A_indx]
        initial_diff = np.linalg.norm(displacement_vector)
        print('Initial separation:',initial_diff)

    list_of_psi4_str_to_generate = []
    for s in disp_factors:
        s = round(s, 2)
        
        molB_copy = molB.copy()
        molB_copy.translate(s*displacement_vector)
        
        s_factor = round(1+s ,3)   # Provided the geometry inputs are the equilibrium geometry
        if isinstance(displacement, tuple):
            A_ind, B_ind = displacement
            new_diff_vector = molB_copy.get_positions()[B_ind]- molA.get_positions()[A_ind]
            new_diff = np.linalg.norm(new_diff_vector)
            print('Separated by:',s_factor, new_diff)
            # print('Separated by:',s, new_diff)
        else:
            new_comB = molB_copy.get_center_of_mass()
            new_com_BA = new_comB - comA
            new_diff = np.linalg.norm(new_com_BA)
            print('Separated by:',s_factor, new_diff)
            # print('Separated by:',s, new_diff)
        dimer = Atoms(molA)
        dimer.extend(molB_copy)

        if to_wrie:
            dimer_fname = f'{molA_name}_{molB_name}_{s_factor}.xyz'
            # dimer_fname = f'{molA_name}_{molB_name}_{new_diff:.3f}.xyz'
            dimer_fpath = geom_folder / dimer_fname
            write(filename=dimer_fpath,
                    images = dimer)
        
        # dimer_to_str = []
        molA_str = ase_to_str(mol=molA)
        molB_str = ase_to_str(mol= molB_copy)
        _dimer = [molA_str, molB_str, s_factor, new_diff]
        # dimer_str = [*molA_str,'\n', '--', *molB_str]
        # dimer_to_str.extend(_dimer)
        # new_dimer_str = ''.join(dimer_str)        
        list_of_psi4_str_to_generate.append(_dimer)
    return list_of_psi4_str_to_generate

def ase_to_str(mol:Atoms):
    """
    Returns formatted geometry coordinates string from ase.Atoms object
    """
    f_str = []
    # atoms_list = []
    for i,symbol in enumerate(mol.get_chemical_symbols()):
        atoms_list = []
        atoms_list.append(str(symbol))

        atoms_list.extend([str(item) for item in mol.get_positions()[i]])  
        f_str.extend('\n')  
        for item in atoms_list:
            f_str.append(item)    
            f_str.append('\t')
    return f_str

def format_psi4_geom(coord_list:list, species_info:list, psi4_options_str:str=None):
    """
    Takes the list of monomer geometry, species info and extra psi4 options as inputs
    Formats the Psi4 readable geometry string and returns in string format

        coord_list: List[list]: Monomers with X, y,z coordinates
                                For dimers, len(coord_list) = 2
        species_info:list[list]: Species info in psi4 format: like charge, multiplicity
        psi4_options_str: Additional psi4 strings
    """
    geom_str_list = []
    for i,monomer in enumerate(coord_list):
        if species_info[i] is not None:
            geom_str_list.append('\n')
            geom_str_list.extend(species_info[i])
            geom_str_list.append('\n')
        geom_str_list.extend(monomer)
        
        if i+1< len(coord_list):
            # geom_str_list.extend(['--'])
            geom_str_list.extend(['\n', '--', '\n'])
    
    geom_str_list.extend(psi4_options_str)
    geom_str = ''.join(geom_str_list) 

    return geom_str  


def read_xyz_to_str(fname, natomA:int=1, distance:tuple=None):
    """
    ---Reads dimer geometry file and formats the psi4 geometry string
        fname: Dimer geometry xyz file
        natomA: Number of atoms of A monomer
        distance: additional argument to calculate the distance between two atoms 
            described by their indices in the tuple format
    ---Returns the list of strings [Monomer A, monomer B, distance(if applicable)]
    """

    with open(fname, 'r') as f:
        data = f.readlines()
    ntotal = len(data)

    molA_list = []
    for i in range(2, natomA+2):
        molA_list.append(data[i])

    molB_list = []
    for i in range(natomA+2, ntotal):
        molB_list.append(data[i])

    if distance is not None:
        dimer = read(filename=fname)
        dist = dimer.get_distance(
                a0=distance[0],
                a1=distance[1])
    return [molA_list, molB_list, dist]


#####################################################################################
if __name__ == '__main__':
    import pathlib
    import os

    # geom_folder = pathlib.Path('/home/huma/phymol/git_repo/humahuti/phymol-dc1/no_pb_test/test/geom')
    geom_folder = pathlib.Path('/home/huma/phymol/git_repo/humahuti/phymol-dc1/no_pb_test/hf2_new')
    monA_fname = geom_folder /'hf.xyz'
    monB_fname = geom_folder /'HF.xyz'
    # monA_fname = geom_folder /'li+.xyz'
    # monB_fname = geom_folder /'h2o.xyz'

    big_list = translate_B_from_A(geomA_file=monA_fname,
                                  geomB_file=monB_fname,
                                  disp_factors= np.arange(-1,2,0.10),
                                #   displacement= np.asarray([0,0,1]),
                                  displacement= (1,0),
                                  to_wrie= True
                                  )
    # monA = read_xyz_to_str(fname='h2o.xyz')
    