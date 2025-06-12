"""
Utilities for calculating densities in SART.
"""

import psi4

import numpy as np

from interaction_induced.cubes import make_cube
from interaction_induced.utils import prepare_path

from utils.helper_SAPT import helper_SAPT


def natom_without_ghosts(molecule: psi4.core.Molecule) -> int:
    """Count number of atoms in a molecule excluding ghost atoms.

    Args:
        molecule (psi4.core.Molecule): Molecule object.

    Returns:
        int: Number of atoms in the molecule excluding ghost atoms.
    """
    return sum(1 for atom in range(molecule.natom()) if molecule.Z(atom) != 0)


def calc_sart_densities(
    wfnA: psi4.core.Wavefunction,
    wfnB: psi4.core.Wavefunction,
    dimer: psi4.core.Molecule,
    h_sapt: helper_SAPT,
    CA: np.ndarray,
    CB: np.ndarray,
    **kwargs,
) -> dict[np.ndarray]:
    """Calculate densities for Slater determinants and density operator expectation values.

    Args:
        wfnA (psi4.core.Wavefunction): Wavefunction for monomer A.
        wfnB (psi4.core.Wavefunction): Wavefunction for monomer B.
        dimer (psi4.core.Molecule): Dimer molecule object.
        h_sapt (helper_SAPT): helper_SAPT object.
        CA (np.ndarray): Orbital coefficient matrix for monomer A.
        CB (np.ndarray): Orbital coefficient matrix for monomer B.

    Returns:
        dict[np.ndarray]: Densities for Slater determinants and density operator expectation values.
    """
    CUBES_DIR = prepare_path(f"visualisation/cubes/{dimer.name()}/")

    molA = wfnA.molecule()
    molB = wfnB.molecule()

    natom_A = natom_without_ghosts(molA)
    natom_B = natom_without_ghosts(molB)

    ndocc_A = h_sapt.ndocc_A
    ndocc_B = h_sapt.ndocc_B

    # Set cube filenames
    fname_cube_A = CUBES_DIR + 'sart_density_A'
    fname_cube_B = CUBES_DIR + 'sart_density_B'
    fname_cube_AB = CUBES_DIR + 'sart_density'

    if kwargs.get("geom_index") is not None:
        fname_cube_A += f'_{kwargs["geom_index"]}'
        fname_cube_B += f'_{kwargs["geom_index"]}'
        fname_cube_AB += f'_{kwargs["geom_index"]}'

    if kwargs.get("itr") is not None:
        fname_cube_A += f'_{kwargs["itr"]}'
        fname_cube_B += f'_{kwargs["itr"]}'
        fname_cube_AB += f'_{kwargs["itr"]}'

    fname_cube_A += '.cube'
    fname_cube_B += '.cube'
    fname_cube_AB += '.cube'

    CA_occ = CA[:, :ndocc_A]
    CB_occ = CB[:, :ndocc_B]

    s_ab = CA_occ.T.dot(h_sapt.S).dot(CB_occ)

    S_AB = np.block(
        [
            [np.diag(np.ones(ndocc_A)), s_ab],
            [s_ab.T, np.diag(np.ones(ndocc_B))],
        ]
    )

    D_AB = np.linalg.inv(S_AB)

    D_aa = D_AB[:ndocc_A, :ndocc_A]
    D_bb = D_AB[ndocc_A:, ndocc_A:]
    D_ab = D_AB[:ndocc_A, ndocc_A:]
    D_ba = D_ab.T

    ### Densities of Slater determinants
    orb_density_A = 2 * (CA_occ.dot(CA_occ.T))
    if kwargs.get("DA") is not None:
        orb_density_A -= 2 * kwargs["DA"]  # initial density
    else:
        print("Initial density for A not provided, calculating total density.")

    orb_density_B = 2 * (CB_occ.dot(CB_occ.T))
    if kwargs.get("DB") is not None:
        orb_density_B -= 2 * kwargs["DB"]  # initial density
    else:
        print("Initial density for B not provided, calculating total density.")

    cube_A = make_cube(
        molA,
        orb_density_A,
        obj_type='density',
        basisset=wfnA.basisset(),
    )

    cube_B = make_cube(
        molB,
        orb_density_B,
        obj_type='density',
        basisset=wfnB.basisset(),
    )

    cube_AB = make_cube(
        dimer,
        orb_density_A + orb_density_B,
        obj_type='density',
    )

    # correct ghosts having 0 atomic number
    cube_A.atoms = cube_A.atoms[:natom_A] + cube_B.atoms[natom_A:]
    cube_B.atoms = cube_A.atoms[:natom_A] + cube_B.atoms[natom_A:]

    cube_A.save(fname_cube_A.replace('density', 'orbital_density'))
    cube_B.save(fname_cube_B.replace('density', 'orbital_density'))
    cube_AB.save(fname_cube_AB.replace('density', 'orbital_density'))

    ### Density operator expectation value densities
    density_A = 2 * (CA_occ.dot(D_aa).dot(CA_occ.T) + CB_occ.dot(D_ba).dot(CA_occ.T))
    if kwargs.get("DA") is not None:
        density_A -= 2 * kwargs["DA"]  # initial density
    else:
        print("Initial density for A not provided, calculating total density.")

    density_B = 2 * (CB_occ.dot(D_bb).dot(CB_occ.T) + CA_occ.dot(D_ab).dot(CB_occ.T))
    if kwargs.get("DB") is not None:
        density_B -= 2 * kwargs["DB"]  # initial density

    cube_op_A = make_cube(
        molA,
        density_A,
        obj_type='density',
        basisset=wfnA.basisset(),
    )

    cube_op_B = make_cube(
        molB,
        density_B,
        obj_type='density',
        basisset=wfnB.basisset(),
    )

    cube_op_AB = make_cube(
        dimer,
        density_A + density_B,
        obj_type='density',
    )

    # correct ghosts atomic numbers
    cube_op_A.atoms = cube_op_A.atoms[:natom_A] + cube_op_B.atoms[natom_A:]
    cube_op_B.atoms = cube_op_A.atoms[:natom_A] + cube_op_B.atoms[natom_A:]

    cube_op_A.save(fname_cube_A)
    cube_op_B.save(fname_cube_B)
    cube_op_AB.save(fname_cube_AB)

    return {
        'density_B': density_B,
        'density_A': density_A,
        'density_dimer': density_A + density_B,
        'orbital_density_A': orb_density_A,
        'orbital_density_B': orb_density_B,
        'orbital_density_dimer': orb_density_A + orb_density_B,
    }
