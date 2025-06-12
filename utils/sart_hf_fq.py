"""
SART-HF-FQ - Low-level implementation

This module provides the core functionality for SART-HF-FQ calculations,
with complete decoupling of FQ and SQ methods.

The module includes functions for:
- Computing first-order energies (E1)
- Building electrostatic, exchange, and LM potentials
- Performing SART-FQ iterations
"""

import numpy as np
import opt_einsum as oe
import copy
import pprint
import psi4
import re
import matplotlib.pyplot as plt
from psi4.driver.p4util.exceptions import ConvergenceError
from psi4.core import MintsHelper

from scipy.linalg import eigh  # Use eigh for Hermitian matrices

# Import utility modules
import sys
sys.path.append('/home/evanich/phymol-dc1/no_pb_dev_fq/ai_routine')
from utils.helper_SAPT import helper_SAPT, sapt_timer

# Default input dictionary with all configuration options
fqsart_inputs_default = {
    # Calculation control flags
    'calc_elst_pot': True,  # Controls calculation of electrostatic potential
    'calc_exch_pot': True,  # Controls calculation of exchange potential
    'calc_lm': True,        # Controls calculation of LM potential
    
    # Method selection flags
    'FQ_elst_exch': True,   # Use FQ method for electrostatic+exchange potential
    'FQ_LM': True,          # Use FQ method for LM potential
    'SQ_elst_exch': False,  # Calculate SQ electrostatic+exchange potential for comparison
    'SQ_LM': False,         # Calculate SQ LM potential for comparison
    
    # Symmetrization options
    'Full symmetrization': False,  # Symmetrize both operators and mapping operators
    'Half symmetrization': False,  # Symmetrize only operators
    
    # Debug and verbosity options
    'Print matrix elements': False, # Kept for backward compatibility, use debug_verbosity_level=3 instead
    'debug_verbosity_level': 1,     # 0: Minimal (input info & final results only)
                                    # 1: Level 0 + scalars at each iteration
                                    # 2: Level 1 + total potentials
                                    # 3: Level 2 + potential components
                                    # 4: Level 3 + debug function details
    'debug_functions': False,       # Controls debug functions (lr, build_enuc_pot)
    
    # Other calculation options
    'long range': False,    # Calculate long-range form of interaction operator
    'max_iter': 100,        # Maximum number of iterations
    'tol': 1.0e-8,          # Convergence tolerance
    
    # Job metadata
    'job_name': 'SART-FQ',
    'job_method': 'SART-FQ',
    'job_basis': 'Unknown',
    'job_identifier': 'Unknown',
    'system_name': 'Unknown'
}

def compute_Eelex(CA, CB, dimer, sapt, D, input_dict, maxiter=10, geom_index=None):
    """
    Computes E_elst + E_exch for the given dimer geometry.
    Modified so that coefficients are passed, in order to test this after
    SART iterations. Also modified to accept input_dict for debug prints
    and to return explicit Eelst and Eexch components.

    Parameters:
    - CA (numpy.ndarray): Coefficient matrix of MOs for monomer A (shape: n_basis x nmo_A).
    - CB (numpy.ndarray): Coefficient matrix of MOs for monomer B (shape: n_basis x nmo_B).
    - dimer (psi4.core.Molecule): The dimer geometry.
    - sapt (helper_SAPT): An instance of the helper_SAPT class.
    - D (numpy.ndarray): Inverse MO overlap matrix (nocc_total x nocc_total).
    - input_dict (dict): Dictionary controlling debug print options.
    - maxiter (int): Maximum number of iterations (optional).
    - geom_index: Geometry index (optional).

    Returns:
    - Eelex_fq (float): The computed value of E_elst + E_exch (FQ).
    - Eelst_fq (float): The first-order electrostatic energy (FQ).
    - Eexch_fq (float): The first-order exchange energy (FQ).
    """
    h_sapt = sapt
    verbosity = input_dict.get("debug_verbosity_level", 0)

    # Extract monomer nuclear repulsion energies
    EnucA = dimer.extract_subsets(1,2).nuclear_repulsion_energy()
    EnucB = dimer.extract_subsets(2,1).nuclear_repulsion_energy()
    EnucAB = dimer.nuclear_repulsion_energy()
    W_AB = EnucAB - EnucA - EnucB  # This is W_{AB}

    # Get number of MOs and occupied orbitals from h_sapt
    nocc_A = h_sapt.ndocc_A  # Number of occupied MOs in monomer A
    nocc_B = h_sapt.ndocc_B  # Number of occupied MOs in monomer B

    # Extract occupied MO coefficients
    CA_occ = CA[:, :nocc_A]  # Shape: n_basis x nocc_A
    CB_occ = CB[:, :nocc_B]  # Shape: n_basis x nocc_B

    # Get the potentials and integrals:
    mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())
    mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())

    # Get VA and VB as in Huma's code:
    VA  = np.asarray(mintsA.ao_potential())
    VB  = np.asarray(mintsB.ao_potential())

    # Combine occupied MO coefficients from both monomers
    C_occ = np.hstack((CA_occ, CB_occ))  # Shape: n_basis x nocc_total

    # Compute UB_MO = <a|U_B|r>
    UB_MO = np.dot(np.dot(CA_occ.T, VB), C_occ)  # Shape: nocc_A x nocc_total

    # Compute UA_MO = <b|U_A|r>
    UA_MO = np.dot(np.dot(CB_occ.T, VA), C_occ)  # Shape: nocc_B x nocc_total
    D_ra = D[:, :nocc_A]  # Shape: nocc_total x nocc_A
    D_sb = D[:, nocc_A:]  # Shape: nocc_total x nocc_B

    # Compute Term1 = <a|U_B|r>*D_ra and Term2 = <b|U_A|s>*D_sb
    term1 = 2*np.einsum('rp, rp ->', D_ra, UB_MO.T)
    term2 = 2*np.einsum('sq, sq ->', D_sb, UA_MO.T)

    # Compute two-electron integrals in AO basis from h_sapt
    I_orig = h_sapt.I 
    I = I_orig.swapaxes(1, 2) # Adjust axes to match (n_basis x n_basis x n_basis x n_basis)

    # Prepare MO coefficient matrices for occupied MOs
    C_a = CA_occ  # n_basis x nocc_A (p in monomer A)
    C_b = CB_occ  # n_basis x nocc_B (q in monomer B)
    C_r = C_occ   # n_basis x nocc_total (r in all occupied MOs)
    C_s = C_occ   # n_basis x nocc_total (s in all occupied MOs)

    # Compute (ar|bs)
    eri_arbs = oe.contract('prqs,pi,rj,qk,sl->ijkl',
                           I, C_a, C_r, C_b, C_s)
    # Shape: nocc_A x nocc_total x nocc_B x nocc_total

    # Compute (as|br)
    eri_asbr = oe.contract('prqs,pi,rj,qk,sl->ilkj',
                           I, C_a, C_r, C_b, C_s)
    # Shape: nocc_A x nocc_total x nocc_B x nocc_total

    term3 = 4 * (np.einsum('ra, sb, arbs ->', D_ra, D_sb, eri_arbs) )
    term4 = 2 * np.einsum('ra, sb, arbs ->', D_ra, D_sb, eri_asbr) 

    Eelex_fq = W_AB + term1 + term2 + term3 - term4

    # This code is correct, but it re-does the transformation step
    ## Compute the electrostatic energy

    #tr_UB_aa = 2*oe.contract('qp,qr,rs->', CA_occ, VB, CA_occ)
    #tr_UA_bb = 2*oe.contract('qp,qr,rs->', CB_occ, VA, CB_occ)
    #tr_eri_aabb = 4*oe.contract('arbs,ai,ri,bj,sj->', I, C_a, C_a, C_b, C_b)
    #Eelst_fq = W_AB + tr_UA_bb + tr_UB_aa + tr_eri_aabb

    # This code re-uses objects already computed:

    # elst = W_AB + 2 (UB)aa + 2 (UA)bb + 4 (aa|bb)
    el_term1 = 0.0
    for a in range(nocc_A):
        el_term1 += UB_MO[a,a]
    el_term1 = 2.0*el_term1

    el_term2 = 0.0
    for b in range(nocc_B):
        el_term2 += UA_MO[b,b+nocc_A]
    el_term2 = 2.0*el_term2

    el_term3 = 0.0
    for a in range(nocc_A):
        for b in range(nocc_B):
            el_term3 += eri_arbs[a,a,b,b+nocc_A]
    el_term3 = 4.0*el_term3

    Eelst_fq = W_AB + el_term1 + el_term2 + el_term3

    Eexch_fq = Eelex_fq - Eelst_fq

    # Print scalar components at verbosity level 2 or higher
    if verbosity >= 2:
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): W_AB = {W_AB}")
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): Eelst_fq = {Eelst_fq}")
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): Eexch_fq = {Eexch_fq}")

    return Eelex_fq, Eelst_fq, Eexch_fq

def compute_delta_LMX(monomer, CA, CB, h_sapt, D, H, I, F, input_dict=None):
    """
    Computes the sum of Murell and Landshoff delta terms for a selected monomer using the supplied one-electron
    operator H (which stands for HA or HB depending on the monomer) and the two-electron integrals I.
    
    The evaluation is as follows:
    
    For monomer A:
      X = np.einsum('qr,ra,pa->qp', C_occ, D_block, CA_occ) 
      term11 = 2 * np.einsum('pq,qp ->', H, X)
      term12 = 2 * np.einsum('pqrs,qp,sr ->', I, X, X)
      term13 = - np.einsum('psrq,qp,sr ->', I, X, X) # written with the convential chemist index ordering for exchange.
      term1  = term11 + term12 + term13
      term2  = np.sum(np.einsum('pi,pq,qi ->', CA_occ, F, CA_occ))
      delta_LA = term1 - term2
      
    For monomer B:
      X = np.einsum('qr,rb,pb->qp', C_occ, D_block, CB_occ) 
      term11 = 2 * np.einsum('pq,qp ->', H, X)
      term12 = 2 * np.einsum('pqrs,qp,sr ->', I, X, X)
      term13 = - np.einsum('psrq,qp,sr ->', I, X, X) # written with the convential chemist index ordering for exchange.
      term1  = term11 + term12 + term13
      term2  = np.sum(np.einsum('pi,pq,qi ->', CB_occ, F, CB_occ))
      delta_LB = term1 - term2

    A common intermediate is computed:
        X = np.einsum('qr,ra,pa->qp', C_occ, D_block, CA_occ or CB_occ) 
    where C_occ is the full occupied space formed by concatenating the 
    occupied MOs from monomer A and monomer B, and D_block is the relevant slice
    of the inverse MO overlap matrix D.
    
    Parameters:
      monomer (str): 'A' or 'B', indicating which monomer is selected.
      CA (np.ndarray): MO coefficient matrix for monomer A (n_basis x nmo_A).
      CB (np.ndarray): MO coefficient matrix for monomer B (n_basis x nmo_B).
      h_sapt: Helper object containing relevant data (e.g., attributes ndocc_A and ndocc_B).
      D (np.ndarray): Inverse MO overlap matrix (n_occ_total x n_occ_total).
      H (np.ndarray): One-electron operator (AO basis) corresponding to the selected monomer.
      I (np.ndarray): Two-electron integrals in physicists' notation
                      (shape: n_basis x n_basis x n_basis x n_basis).
      input_dict (dict, optional): Dictionary controlling debug print options.
    
    Returns:
      delta_LMX (float): The computed Landshoff delta value for the given monomer.
    """
    # Get verbosity level from input_dict
    if input_dict is None:
        input_dict = fqsart_inputs_default
    verbosity = input_dict.get("debug_verbosity_level", 0)
    print_matrix_elements = input_dict.get("Print matrix elements", False)
    
    # Print scalar values at verbosity level 1 or higher
    if verbosity >= 1:
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): Computing delta LM energy")
    
    # Retrieve number of occupied orbitals for each monomer.
    nocc_A = h_sapt.ndocc_A   # Number of occupied MOs in monomer A
    nocc_B = h_sapt.ndocc_B   # Number of occupied MOs in monomer B
    # Extract the occupied parts of the MO coefficient matrices.
    CA_occ = CA[:, :nocc_A]               # Shape: (n_basis x nocc_A)
    CB_occ = CB[:, :nocc_B]               # Shape: (n_basis x nocc_B)

    # Form the full occupied block from both monomers.
    C_occ = np.hstack((CA_occ, CB_occ))    # Shape: (n_basis x n_occ_total)
    
    if monomer.upper() == 'A':
        # Select the slice of D corresponding to monomer A's occupied orbitals.
        D_block = D[:, :nocc_A]           # Shape: (n_occ_total x nocc_A)
        
        # Compute the common intermediate object for monomer A.
        X = np.einsum('qr,ra,pa->qp', C_occ, D_block, CA_occ)  # Shape: (n_basis x nbasis)

        # Diagonal (trace) contribution using the Fock operator.
        term2  = np.einsum('mp,mn,np->', CA_occ, F, CA_occ) 
        term2  += np.einsum('mp,mn,np->', CA_occ, H, CA_occ)

    elif monomer.upper() == 'B':
        # Select the D block corresponding to monomer B's occupied orbitals.
        D_block = D[:, nocc_A:]           # Shape: (n_occ_total x nocc_B)
        
        # Compute the common intermediate object for monomer B.
        X = np.einsum('qr,rb,pb->qp', C_occ, D_block, CB_occ)  # Shape: (n_basis x nbasis)

        # Diagonal (trace) contribution using the Fock operator.
        term2  = np.einsum('mp,mn,np->', CB_occ, F, CB_occ) 
        term2  += np.einsum('mp,mn,np->', CB_occ, H, CB_occ)
    else:
        raise ValueError("monomer must be 'A' or 'B'")
    
    # --- Standard computation ---
    # Compute term11, term12 and term13 using the combined intermediate. 
    # This should be the most computetional efficient way of computing it.
    term11 = 2 * np.einsum('pq,qp->', H, X)
    term12 = 2 * np.einsum('pqrs,qp,sr->', I, X, X)
    term13 = - np.einsum('psrq,qp,sr ->', I, X, X) # written with the convential chemist index ordering for exchange.
    term1  = term11 + term12 + term13
    
    # Print components at verbosity level 2 or higher
    if verbosity >= 3:
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term11 = {term11}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term12 = {term12}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term13 = {term13}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term1 = {term1}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term2 = {term2}")
    
    # Print matrix elements at verbosity level 3 or higher
    if verbosity >= 4 or print_matrix_elements:
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): D_block =\n{D_block}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): X =\n{X}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): H =\n{H}")
    
    # Compute the final Landshoff Delta.
    delta_LMX = term1 - term2

    ## Print scalar result at verbosity level 1 or higher
    #if verbosity >= 1:
    #    print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): delta_LM{monomer} = {delta_LMX}")

    return delta_LMX

def compute_D_matrix(CA_occ, CB_occ, S):
    """
    Computes the inverse MO overlap matrix D.
    
    Parameters:
      CA_occ (numpy.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (numpy.ndarray): Occupied MO coefficients for monomer B.
      S (numpy.ndarray): AO overlap matrix.
      
    Returns:
      D (numpy.ndarray): Inverse MO overlap matrix.
    """
    # Compute the overlap matrices within monomers (should be close to identity)
    SAA = np.dot(np.dot(CA_occ.T, S), CA_occ)
    SBB = np.dot(np.dot(CB_occ.T, S), CB_occ)

    # Compute the off-diagonal overlap matrix between monomers
    SAB = np.dot(np.dot(CA_occ.T, S), CB_occ)
    SBA = SAB.T
    
    # Combine into block matrix
    S_MO = np.block([[SAA, SAB], [SBA, SBB]])
    
    # Compute inverse
    D = np.linalg.inv(S_MO)
    
    return D

def build_F_int(CA, CB, h_sapt, D, monomer='B', input_dict=None):
    """
    Constructs the interaction operator F_X^{int} in the AO basis for a given monomer,
    using the D matrix from compute_D_matrix() function, and following the specific index
    transformations as per the definitions provided.

    Parameters:
    - CA (numpy.ndarray): Coefficient matrix for monomer A (shape: n_basis x nmo_A).
    - CB (numpy.ndarray): Coefficient matrix for monomer B (shape: n_basis x nmo_B).
    - h_sapt (helper_SAPT): Helper object containing necessary wavefunction and integral data.
    - D (numpy.ndarray): D matrix from compute_D_matrix(), shape (n_occ_total x n_occ_total).
    - I_val (numpy.ndarray): Two-electron integrals, already correctly oriented (e.g. (mu nu|lambda sigma) or (mu lambda|nu sigma) depending on convention needed).
    - monomer (str): \'A\' or \'B\' to specify which monomer\'s F_int to compute.
    - input_dict (dict): Dictionary controlling debug print options.

    Returns:
    - FX_int_AO (numpy.ndarray): Interaction operator in the AO basis, shape (n_basis x n_basis).
    """
    verbosity = 0
    print_matrix_elements = False
    if input_dict is None:
        input_dict = fqsart_inputs_default
    if input_dict:
        verbosity = input_dict.get("debug_verbosity_level", 0)
        print_matrix_elements = input_dict.get("Print matrix elements", False)

    # Extract necessary variables from h_sapt
    I = h_sapt.I.swapaxes(1, 2) # Using I_val passed as argument now
    nocc_A = h_sapt.ndocc_A
    nocc_B = h_sapt.ndocc_B

    # Occupied MO coefficients
    CA_occ = CA[:, :nocc_A]  # Shape: nbf x nocc_A
    CB_occ = CB[:, :nocc_B]  # Shape: nbf x nocc_B

    # Split D matrix into blocks
    D_AA = D[:nocc_A, :nocc_A]
    D_AB = D[:nocc_A, nocc_A:]
    D_BA = D[nocc_A:, :nocc_A]
    D_BB = D[nocc_A:, nocc_A:]

    # Choose monomer
    if monomer == 'A':
        if verbosity >= 1:
            print(f"--- Building F_A^{{int}} ---")    
        mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())
        VA  = np.asarray(mintsA.ao_potential())
        UX = VA
        CX_occ = CA_occ # Monomer X is A
        # For F_A_int, we sum contributions from D_AA (A interacting with A) and D_BA (A interacting with B)
        # D_XY_list: (D-block, Coeffs_Y)
        D_XY_list = [(D_AA, CA_occ), (D_BA, CB_occ)] 
    elif monomer == 'B':
        if verbosity >= 1:
            print(f"--- Building F_B^{{int}} ---")
        mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())
        VB  = np.asarray(mintsB.ao_potential())
        UX = VB
        CX_occ = CB_occ # Monomer X is B
        # For F_B_int, we sum contributions from D_AB (B interacting with A) and D_BB (B interacting with B)
        # D_XY_list: (D-block, Coeffs_Y)
        D_XY_list = [(D_AB, CA_occ), (D_BB, CB_occ)]
    else:
        raise ValueError("monomer must be \'A\' or \'B\'")

    # Initialize F_X_int_AO with the one-electron potential operator of the OTHER monomer
    FX_int_AO = UX.copy()  # Shape: nbf x nbf. UX is V_other

    if verbosity >= 3 or print_matrix_elements:
        print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): Initial FX_int_AO (UX - potential from other monomer):\n{UX}")

    # --- Compute Coulomb term ---
    # (J_X)_{\mu\nu} = 2 * sum_{y in {A,B}} sum_{b in Y, s in Y} (D_X,Y)_{bs} * (\mu \nu | b s)
    # where b are MOs of X, s are MOs of Y
    JX = np.zeros_like(FX_int_AO)
    for (D_XY, CY_occ) in D_XY_list:
        ## eri_munu_bs = (mu nu | b s) where b from CX_occ, s from CY_occ_coeffs
        ## I_val is (mu nu | p q) or similar, needs to be (mu nu | lambda sigma)
        ## So, CX_occ maps to lambda, CY_occ_coeffs maps to sigma
        #eri_munu_bs = oe.contract('mnpq,pl,qs->mnls', I_val, CX_occ, CY_occ_coeffs) # mu,nu,lambda,sigma -> mu,nu,b_X,s_Y)
        #JX_contrib = 2 * oe.contract('ls,mnls->mn', D_block_XY, eri_munu_bs) # D_XY is (lambda_X, sigma_Y) or (b_X, s_Y)
        #JX += JX_contrib

        # Compute (mu nu | b s)
        eri_munu_bs = oe.contract('mnpq,pb,qs->mnbs', I, CX_occ, CY_occ)
        # Contract with D_XY
        JX_contrib = 2*oe.contract('sb,mnbs->mn', D_XY, eri_munu_bs) # type: ignore
        JX += JX_contrib
        if verbosity >= 4 or print_matrix_elements:
            print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): JX contribution from D_block {D_XY.shape} and CY_occ {CY_occ.shape}:\n{JX_contrib}")

    FX_int_AO += JX
    if verbosity >= 3 or print_matrix_elements:
        print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): Total JX:\n{JX}")
    if verbosity >=4 or print_matrix_elements:
        print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): FX_int_AO after JX:\n{FX_int_AO}")

    # --- Compute Exchange term ---
    # (K_X)_{\mu\nu} = -1 * sum_{y in {A,B}} sum_{b in X, s in Y} (D_X,Y)_{bs} * (\mu b | s \nu)
    KX = np.zeros_like(FX_int_AO)
    for (D_XY, CY_occ) in D_XY_list:
        ## eri_mub_snu = (mu b | s nu) where b from CX_occ, s from CY_occ_coeffs
        ## I_val is (mu nu | p q) or (mu p | nu q) for chemist (mu nu || p q) = (mu p | nu q) - (mu q | nu p)
        ## Assuming I_val is (mu lambda | nu sigma) for (mu nu || lambda sigma)
        ## (mu b | s nu) -> I_val(mu,lambda_b, nu,sigma_s) -> contract I_val with CX_occ on 2nd index, CY_occ_coeffs on 4th index
        #eri_mub_snu = oe.contract('mlns,lb,sq->mbnq', I_val, CX_occ, CY_occ_coeffs) # mu,lambda,nu,sigma -> mu,b_X,nu,s_Y
        #KX_contrib = oe.contract('bs,mbnq->mq', D_block_XY, eri_mub_snu) # D_XY is (b_X,s_Y)
        # Compute (mu s | b nu)
        eri_mus_bnu = oe.contract('mqpn,pb,qs->msbn', I, CX_occ, CY_occ)
        # Contract with D_XY
        KX_contrib = oe.contract('sb,msbn->mn', D_XY, eri_mus_bnu)
        KX += KX_contrib # type: ignore
        if verbosity >= 4 or print_matrix_elements:
            print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): KX contribution from D_block {D_XY.shape} and CY_occ {CY_occ.shape}:\n{KX_contrib}")

    FX_int_AO -= KX # Subtract exchange
    if verbosity >= 3 or print_matrix_elements:
        print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): Total KX:\n{KX}")
    if verbosity >=2:
        print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): Final FX_int_AO:\n{FX_int_AO}")

    return FX_int_AO

def build_F_int_lr(h_sapt, monomer='A', input_dict=None):
    """
    Computes the long-range perturbed Fock operator for monomer X.
    
    For monomer A:
      F_int_lr = ⟨μ|U_A|ν⟩ + 2 * ∑ₐ (μν|aa) - ∑ₐ (μa|νa)
    
    For monomer B:
      F_int_lr = ⟨μ|U_B|ν⟩ + 2 * ∑_b (μν|bb) - ∑_b (μb|νb)
      
    Here the sums over a (or b) are performed by contracting the two-electron integral
    with the occupied orbital coefficients (h_sapt.C_A or h_sapt.C_B).
    
    Parameters:
      h_sapt: helper_SAPT instance containing wavefunctions and integrals.
      monomer (str): Either 'A' or 'B'.
      input_dict (dict, optional): Dictionary controlling debug print options.
    
    Returns:
      F_lr (np.ndarray): The long-range interaction Fock matrix.
    """
    # Set default input_dict if not provided
    if input_dict is None:
        input_dict = fqsart_inputs_default
    
    # Get verbosity level and debug flags
    verbosity = input_dict.get("debug_verbosity_level", 0)
    debug_functions = input_dict.get("debug_functions", False)
    print_matrix_elements = input_dict.get("Print matrix elements", False)
    
    # Only execute this debug function if debug_functions is True
    if not debug_functions:
        return None
    
    # Print function entry at verbosity level 1 or higher
    if verbosity >= 1:
        print(f"DEBUG build_F_int_lr (Verbosity {verbosity}, Monomer {monomer}): Building long-range interaction Fock matrix")
    
    # Retrieve the two-electron integrals in AO basis; adjust axes if needed.
    I = h_sapt.I.swapaxes(1,2)

    if monomer.upper() == 'A':
        # For monomer A use the nuclear–electron potential from wfnA.
        mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())
        U = np.asarray(mintsA.ao_potential()) # N-e potential for A
        nocc = h_sapt.ndocc_A
        CA = h_sapt.C_A
        CA_occ = CA[:, :nocc]
        # Contract: 2∑ₐ (μν|aa) = 2 * einsum('pqrs, ra, sa -> pq', I, CA_occ, CA_occ)
        term1 = 2 * np.einsum('pqrs,ra,sa->pq', I, CA_occ, CA_occ)
        # Contract: ∑ₐ (μa|νa) = einsum('prqs,ra,sa->pq', I, CA_occ, CA_occ)
        term2 = np.einsum('prqs,ra,sa->pq', I, CA_occ, CA_occ)
    elif monomer.upper() == 'B':
        # For monomer B use the potential from wfnB.
        mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())
        U = np.asarray(mintsB.ao_potential()) # N-e potential for B
        nocc = h_sapt.ndocc_B
        CB = h_sapt.C_B
        CB_occ = CB[:, :nocc]
        # Contract over the occupied orbitals of B:
        term1 = 2 * np.einsum('pqrs,rb,sb->pq', I, CB_occ, CB_occ)
        term2 = np.einsum('prqs,rb,sb->pq', I, CB_occ, CB_occ)
    else:
        raise ValueError("monomer must be 'A' or 'B'")
    
    # Print matrix components at verbosity level 4 or higher (debug functions)
    if verbosity >= 4 or print_matrix_elements:
        print(f"DEBUG build_F_int_lr (Verbosity {verbosity}, Monomer {monomer}): U =\n{U}")
        print(f"DEBUG build_F_int_lr (Verbosity {verbosity}, Monomer {monomer}): J =\n{term1}")
        print(f"DEBUG build_F_int_lr (Verbosity {verbosity}, Monomer {monomer}): K =\n{term2}")
    
    F_lr = U + term1 - term2
    F_lr_noK = U + term1
    
    # Print additional matrix components at verbosity level 4 or higher
    if verbosity >= 4 or print_matrix_elements:
        print(f"DEBUG build_F_int_lr (Verbosity {verbosity}, Monomer {monomer}): F_lr_noK =\n{F_lr_noK}")
    if verbosity >=2:
        print(f"DEBUG build_F_int_lr (Verbosity {verbosity}, Monomer {monomer}): F_lr =\n{F_lr}")
    
    return F_lr


def build_elst_pot(monomer, h_sapt, CA_occ, CB_occ, input_dict):
    """
    Constructs the electrostatic potential only for a specified monomer.
    
    Parameters:
      monomer (str): "A" or "B"
      h_sapt (helper_SAPT): An instance of the helper_SAPT class.
      CA_occ (numpy.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (numpy.ndarray): Occupied MO coefficients for monomer B.
      input_dict (dict): Dictionary controlling debug print options.
      
    Returns:
      W_tot (numpy.ndarray): The total electrostatic interaction potential.
    """
    if input_dict is None:
        input_dict = fqsart_inputs_default
    verbosity = input_dict.get("debug_verbosity_level", 0)
    print_matrix_elements = input_dict.get("Print matrix elements", False)
    
    if verbosity >= 1:
        print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer {monomer}): Building electrostatic potential")

    # Get the 2 electron integrals.    
    I = h_sapt.I.swapaxes(1,2)

    # Build the interaction Fock matrix (electrostatic potential)
    if monomer.upper() == "A":
        if verbosity >= 1:
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer A): Building Elst interaction potential for A = VB + 2*JB")
        mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())
        VB  = np.asarray(mintsB.ao_potential())
        JB = np.einsum('pqrs,rb,sb->pq', I, CB_occ, CB_occ)
        W_tot = VB + 2*JB

        if verbosity >= 4: # Components VB and JB at verbosity 3
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer A): VB component:\n{VB}")
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer A): JB component:\n{JB}")

    elif monomer.upper() == "B":
        if verbosity >= 1:
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer B): Building Elst interaction potential for B = VA + 2*JA")
        mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())
        VA  = np.asarray(mintsA.ao_potential())
        JA = np.einsum('pqrs,rb,sb->pq', I, CA_occ, CA_occ)
        W_tot = VA + 2*JA

        if verbosity >= 4: # Components VA and JA at verbosity 2
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer B): VA component:\n{VA}")
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer B): JA component:\n{JA}")
    else:
        raise ValueError("monomer must be either \"A\" or \"B\"")
    
    
    return W_tot

def build_enuc_int_pot(monomer, h_sapt, CA_occ, CB_occ, D, S, VA, VB, input_dict):
    """
    Constructs the nuclear-electron interaction potential for a specified monomer.
    This is a debug function for separating nuclear-electron and electron-electron contributions.
    
    For monomer A, the routine follows these steps:
      -- Extract D matrix blocks:
            D_AA = D[:nocc_A, :nocc_A]
            D_AB = D[:nocc_A, nocc_A:]
            D_BB = D[nocc_A:, nocc_A:]
      -- Compute mapping operators:
            ADBS = np.einsum('pq,qi,ji,jk -> pk', CA_occ, D_AB, CB_occ, S, optimize=True)
            BDBS = np.einsum('pq,qi,ji,jk -> pk', CB_occ, D_BB, CB_occ, S, optimize=True)
      -- Form the electrostatic + exchange potential with only electron-nuclear contributions:
            WA_elst_exch_enuc = VB
              - np.einsum('pq, qi -> pi', VB, ADBS)
              - np.einsum('pq, qi -> pi', VB, BDBS)
              - np.einsum('qp, qi -> pi', ADBS, VB)
              + np.einsum('qp, qi, ij -> pj', ADBS, VB, ADBS)
              + np.einsum('qp, qi, ij -> pj', ADBS, VB, BDBS)
              - np.einsum('qp, qi -> pi', BDBS, VA)
              + np.einsum('qp, qi, ij -> pj', BDBS, VA, ADBS)
              + np.einsum('qp, qi, ij -> pj', BDBS, VA, BDBS)
              for Bra:
              term1 = VB
              term2 = -VB*ADBS
              term3 = -VB*BDBS
              term4 = -SBDA*VB
              term5 = SBDA*VB*ADBS
              term6 = SBDA*VB*BDBS
              term7 = -SBDB*VA
              term8 = SBDB*VA*ADBS
              term9 = SBDB*VA*BDBS

              term4 = term2^T
              term5 = term5^T

              for ket:
              term1 = VB
              term2 = -SBDA*VB = bra term4
              term3 = -SBDB*VB = bra term3^T
              term4 = -VB*ADBS = bra term2
              term5 = SBDA*VB*ADBS = bra term5
              term6 = SBDA*VB*BDBS = bra term6
              term7 = -SBDB*VA = bra term7
              term8 = SBDB*VA*ADBS = bra term8
              term9 = SBDB*VA*BDBS = bra term9 
    
    Parameters:
      monomer (str): "A" or "B"
      h_sapt (helper_SAPT): An instance of the helper_SAPT class.
      CA_occ (numpy.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (numpy.ndarray): Occupied MO coefficients for monomer B.
      input_dict (dict): Dictionary controlling debug print options.
      
    Returns:
      tuple: (W_nuc, W_elec) The nuclear-electron and electron-electron interaction potentials.
    """
    # Set default input_dict if not provided
    if input_dict is None:
        input_dict = fqsart_inputs_default
    verbosity = input_dict.get("debug_verbosity_level", 0)
    debug_functions = input_dict.get("debug_functions", False)
    print_matrix_elements = input_dict.get("Print matrix elements", False) # Retained for backward compatibility
    
    nocc_A = h_sapt.ndocc_A
    # Only execute this debug function if debug_functions is True
    if not debug_functions:
        return None, None
    
    # Print function entry at verbosity level 4 or higher (debug functions)
    if verbosity >= 1:
        print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): Building nuclear-electron interaction potential")
    
    if monomer.upper() == "A":
        D_AB = D[:nocc_A, nocc_A:]
        D_BB = D[nocc_A:, nocc_A:]
        ADBS = np.einsum('pq,qi,ji,jk -> pk', CA_occ, D_AB, CB_occ, S, optimize=True)
        BDBS = np.einsum('pq,qi,ji,jk -> pk', CB_occ, D_BB, CB_occ, S, optimize=True)
        
        term1 = VB.copy()
        term2 = -np.einsum('pq,qi -> pi', VB, ADBS)
        term3 = -np.einsum('pq,qi -> pi', VB, BDBS)
        term4 = -np.einsum('qp,qi -> pi', ADBS, VB)
        term5 = np.einsum('qp,qi,ij -> pj', ADBS, VB, ADBS)
        term6 = np.einsum('qp,qi,ij -> pj', ADBS, VB, BDBS)
        term7 = -np.einsum('qp,qi -> pi', BDBS, VA)
        term8 = np.einsum('qp,qi,ij -> pj', BDBS, VA, ADBS)
        term9 = np.einsum('qp,qi,ij -> pj', BDBS, VA, BDBS)
        
        # Print components at verbosity level 4 or higher (debug functions)
        if verbosity >= 4:
            print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): VB =\n{VB}")
        
    
    elif monomer.upper() == "B":
        D_BA = D[nocc_A:, :nocc_A]
        D_AA = D[:nocc_A, :nocc_A]
        BDAS = np.einsum('pq,qi,ji,jk -> pk', CB_occ, D_BA, CA_occ, S, optimize=True)
        ADAS = np.einsum('pq,qi,ji,jk -> pk', CA_occ, D_AA, CA_occ, S, optimize=True)

        term1 = VA.copy()
        term2 = -np.einsum('pq,qi -> pi', VA, BDAS) 
        term3 = -np.einsum('pq,qi -> pi', VA, ADAS)
        term4 = -np.einsum('qp,qi -> pi', BDAS, VA)
        term5 = np.einsum('qp,qi,ij -> pj', BDAS, VA, BDAS)
        term6 = np.einsum('qp,qi,ij -> pj', BDAS, VA, ADAS) 
        term7 = -np.einsum('qp,qi -> pi', ADAS, VB)
        term8 = np.einsum('qp,qi,ij -> pj', ADAS, VB, BDAS)
        term9 = np.einsum('qp,qi,ij -> pj', ADAS, VB, ADAS)
       
        if verbosity >= 4:
            print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): VA =\n{VA}")
        
    else:
        raise ValueError("monomer must be either \"A\" or \"B\"")
    
    WX_elst_exch = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    # --- compute the exchange potential without LM ---
    WX_exch = WX_elst_exch - term1 + term3 + term7 - term9 

    if verbosity >= 2:
        print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): Total E_Nuc Elst+Exch potential (W{monomer}_elst_exch):\n{WX_elst_exch}")
    
    if verbosity >= 4 or print_matrix_elements:
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): XDYS :\n', ADBS)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): YDYS :\n', BDBS)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term1 (VY) :\n', term1)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term2 (-VY*XDYS, expected 0):\n', term2)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term3 (-VY*YDYS) :\n', term3)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term4 (-XDYS*VY, expected 0) :\n', term4)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term5 (XDYS*VY*XDYS, expected 0) :\n', term5)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term6 (XDYS*VY*YDYS, expected 0) :\n', term6)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term7 (-YDYS*VX) :\n', term7)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term8 (YDYS*VX*XDYS, expected 0) :\n', term8)
        print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): term9 (YDYS*VX*YDYS) :\n', term9)
        print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer {monomer}): Exchange part (WX_exch):\n{WX_exch}")

    
    return WX_elst_exch

def build_elst_exch_pot(monomer, h_sapt, CA_occ, CB_occ, D, S, FA_int, FB_int, input_dict):
    """
    Constructs the total electrostatic + exchange potential for a specified monomer.
    
    For monomer A, the routine follows these steps:
      -- Extract D matrix blocks:
            D_AA = D[:nocc_A, :nocc_A]
            D_AB = D[:nocc_A, nocc_A:]
            D_BB = D[nocc_A:, nocc_A:]
      -- Compute mapping operators:
            ADBS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AB, CB_occ, S, optimize=True)
            BDBS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BB, CB_occ, S, optimize=True)
      -- Form the electrostatic + exchange potential:
            WA_elst_exch = FB_int 
              - np.einsum("pq, qi -> pi", FB_int, ADBS)
              - np.einsum("pq, qi -> pi", FB_int, BDBS)
              - np.einsum("qp, qi -> pi", ADBS, FB_int)
              + np.einsum("qp, qi, ij -> pj", ADBS, FB_int, ADBS)
              + np.einsum("qp, qi, ij -> pj", ADBS, FB_int, BDBS)
              - np.einsum("qp, qi -> pi", BDBS, FA_int)
              + np.einsum("qp, qi, ij -> pj", BDBS, FA_int, ADBS)
              + np.einsum("qp, qi, ij -> pj", BDBS, FA_int, BDBS)
    
    For monomer B, similar expressions are used with:
            BDAS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BA, CA_occ, S, optimize=True)
            ADAS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AA, CA_occ, S, optimize=True)
    
    Parameters:
      monomer (str): "A" or "B"
      h_sapt (helper_SAPT): An instance of the helper_SAPT class.
      CA_occ (numpy.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (numpy.ndarray): Occupied MO coefficients for monomer B.
      D (numpy.ndarray): Inverse MO overlap matrix.
      S (numpy.ndarray): AO overlap matrix.
      FA_int (numpy.ndarray): Interaction Fock matrix for monomer A.
      FB_int (numpy.ndarray): Interaction Fock matrix for monomer B.
      input_dict (dict): Dictionary controlling debug print options.
      
    Returns:
      WX_elst_exch (numpy.ndarray): The total electrostatic + exchange potential.
    """
    # Set default input_dict if not provided
    if input_dict is None:
        input_dict = fqsart_inputs_default
    verbosity = input_dict.get("debug_verbosity_level", 0)
    print_matrix_elements = input_dict.get("Print matrix elements", False)
    full_sym = input_dict.get("Full symmetrization", False)
    half_sym = input_dict.get("Half symmetrization", False)

    if verbosity >= 1:
        print(f"--Building {monomer} Elst+Exch potential--")
        if full_sym:
            print(f"- Full symmetrization scheme -")
        elif half_sym:
            print(f"- Half symmetrization scheme -")
        else:
            print(f"- No symmetrization scheme -")
    
    nocc_A = h_sapt.ndocc_A
    
    if monomer.upper() == "A":
        # Compute the electrostatic + exchange potential for monomer A
        D_AB = D[:nocc_A, nocc_A:]
        D_BB = D[nocc_A:, nocc_A:]
        
        # Compute mapping operators
        ADBS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AB, CB_occ, S, optimize=True)
        BDBS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BB, CB_occ, S, optimize=True)
        
        # Symmetrize operators if requested
        current_FA_int = FA_int.copy()
        current_FB_int = FB_int.copy()
        
        if full_sym:
            current_FA_int = (current_FA_int + current_FA_int.T)/2
            current_FB_int = (current_FB_int + current_FB_int.T)/2
            ADBS = (ADBS + ADBS.T)/2
            BDBS = (BDBS + BDBS.T)/2
            if verbosity >= 3 or print_matrix_elements:
                print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): FA_int, FB_int, ADBS, BDBS symmetrized for full_sym.")
        elif half_sym:
            current_FA_int = (current_FA_int + current_FA_int.T)/2
            current_FB_int = (current_FB_int + current_FB_int.T)/2
            if verbosity >= 3 or print_matrix_elements:
                print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): FA_int, FB_int symmetrized for half_sym.")
        
        # Compute the potential terms
        term1 = current_FB_int.copy()
        term2 = -np.einsum("pq,qi -> pi", current_FB_int, ADBS)
        term3 = -np.einsum("pq,qi -> pi", current_FB_int, BDBS)
        term4 = -np.einsum("qp,qi -> pi", ADBS, current_FB_int)
        term5 = np.einsum("qp,qi,ij -> pj", ADBS, current_FB_int, ADBS)
        term6 = np.einsum("qp,qi,ij -> pj", ADBS, current_FB_int, BDBS)
        term7 = -np.einsum("qp,qi -> pi", BDBS, current_FA_int)
        term8 = np.einsum("qp,qi,ij -> pj", BDBS, current_FA_int, ADBS)
        term9 = np.einsum("qp,qi,ij -> pj", BDBS, current_FA_int, BDBS)
        
        # Combine all terms
        WA_elst_exch = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
        
        # Print components at verbosity level 3 or higher
        if verbosity >= 3 or print_matrix_elements:
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): ADBS (mapping op):\n{ADBS}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): BDBS (mapping op):\n{BDBS}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): current_FA_int:\n{current_FA_int}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): current_FB_int:\n{current_FB_int}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term1 (FB_int):\n{term1}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term2 (-FB_int*ADBS):\n{term2}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term3 (-FB_int*BDBS):\n{term3}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term4 (-ADBS*FB_int):\n{term4}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term5 (ADBS*FB_int*ADBS):\n{term5}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term6 (ADBS*FB_int*BDBS):\n{term6}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term7 (-BDBS*FA_int):\n{term7}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term8 (BDBS*FA_int*ADBS):\n{term8}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term9 (BDBS*FA_int*BDBS):\n{term9}")
        
        
        return WA_elst_exch
    
    elif monomer.upper() == "B":
        # Compute the electrostatic + exchange potential for monomer B
        D_AA = D[:nocc_A, :nocc_A]
        D_BA = D[nocc_A:, :nocc_A]
        
        # Compute mapping operators
        BDAS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BA, CA_occ, S, optimize=True)
        ADAS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AA, CA_occ, S, optimize=True)
        
        # Symmetrize operators if requested
        current_FA_int = FA_int.copy()
        current_FB_int = FB_int.copy()
        
        if full_sym:
            current_FA_int = (current_FA_int + current_FA_int.T)/2
            current_FB_int = (current_FB_int + current_FB_int.T)/2
            BDAS = (BDAS + BDAS.T)/2
            ADAS = (ADAS + ADAS.T)/2
            if verbosity >= 1:
                print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): FA_int, FB_int, BDAS, ADAS symmetrized for full_sym.")
        elif half_sym:
            current_FA_int = (current_FA_int + current_FA_int.T)/2
            current_FB_int = (current_FB_int + current_FB_int.T)/2
            if verbosity >= 1:
                print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): FA_int, FB_int symmetrized for half_sym.")
        
        # Compute the potential terms
        term1 = current_FA_int.copy()
        term2 = -np.einsum("pq,qi -> pi", current_FA_int, BDAS)
        term3 = -np.einsum("pq,qi -> pi", current_FA_int, ADAS)
        term4 = -np.einsum("qp,qi -> pi", BDAS, current_FA_int)
        term5 = np.einsum("qp,qi,ij -> pj", BDAS, current_FA_int, BDAS)
        term6 = np.einsum("qp,qi,ij -> pj", BDAS, current_FA_int, ADAS)
        term7 = -np.einsum("qp,qi -> pi", ADAS, current_FB_int)
        term8 = np.einsum("qp,qi,ij -> pj", ADAS, current_FB_int, BDAS)
        term9 = np.einsum("qp,qi,ij -> pj", ADAS, current_FB_int, ADAS)
        
        # Combine all terms
        WB_elst_exch = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
        
        # Print components at verbosity level 3 or higher
        if verbosity >= 4 or print_matrix_elements:
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): BDAS (mapping op):\n{BDAS}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): ADAS (mapping op):\n{ADAS}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): current_FA_int:\n{current_FA_int}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): current_FB_int:\n{current_FB_int}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term1 (FA_int):\n{term1}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term2 (-FA_int*BDAS):\n{term2}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term3 (-FA_int*ADAS):\n{term3}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term4 (-BDAS*FA_int):\n{term4}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term5 (BDAS*FA_int*BDAS):\n{term5}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term6 (BDAS*FA_int*ADAS):\n{term6}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term7 (-ADAS*FB_int):\n{term7}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term8 (ADAS*FB_int*BDAS):\n{term8}")
            print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term9 (ADAS*FB_int*ADAS):\n{term9}")
        
        # Print total potential at verbosity level 2 or higher
        if verbosity >= 2:
            print(f'DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): Total Elst+Exch potential (WB_elst_exch):\n{WB_elst_exch}')
        
        return WB_elst_exch
    
    else:
        raise ValueError("monomer must be either \"A\" or \"B\"")

def build_elst_exch_pot_lr(monomer, h_sapt, CA_occ, CB_occ, FA_int, FB_int, S, input_dict):
    """
    Constructs the long-range form of the electrostatic + exchange potential for a specified monomer.
    This is a debug function for checking asymptotic behavior.
    
    At long range the off-diagonal D blocks vanish and the intra-monomer D blocks become the identity.
    Thus, we define:
      For monomer A:
         BBS = np.einsum('pa,qa,qk->pk', CB_occ, CB_occ, S, optimize=True)
      For monomer B:
         AAS = np.einsum('pa,qa,qk->pk', CA_occ, CA_occ, S, optimize=True)
    
    For monomer A the surviving elst+exch terms are:
      WA_elst_exch = FB_int
                      - np.einsum('pq,qi->pi', FB_int, BBS)
                      - np.einsum('qp,qi->pi', BBS, FA_int)
                      + np.einsum('qp,qi,ij->pj', BBS, FA_int, BBS)
    
    The total long-range interaction potential becomes:
      V_lr_A = WA_elst_exch 
    
    For monomer B, analogous expressions hold using AAS instead of BBS.
    
    Parameters:
      monomer (str): "A" or "B"
      h_sapt (helper_SAPT): An instance of the helper_SAPT class.
      CA_occ (numpy.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (numpy.ndarray): Occupied MO coefficients for monomer B.
      D (numpy.ndarray): Inverse MO overlap matrix.
      S (numpy.ndarray): AO overlap matrix.
      FA_int (numpy.ndarray): Interaction Fock matrix for monomer A.
      FB_int (numpy.ndarray): Interaction Fock matrix for monomer B.
      input_dict (dict): Dictionary controlling debug print options.
      
    Returns:
      WX_elst_exch_lr (numpy.ndarray): The long-range electrostatic + exchange potential.
    """
    # Set default input_dict if not provided
    if input_dict is None:
        input_dict = fqsart_inputs_default
    verbosity = input_dict.get("debug_verbosity_level", 0)
    debug_functions = input_dict.get("debug_functions", False)
    print_matrix_elements = input_dict.get("Print matrix elements", False) if input_dict else False
    
    # Only execute this debug function if debug_functions is True
    if not debug_functions:
        return None
    
    # Print function entry at verbosity level 4 or higher (debug functions)
    if verbosity >= 1:
        print(f"DEBUG build_elst_exch_pot_lr (Verbosity {verbosity}, Monomer {monomer}): Building long-range electrostatic + exchange potential")
    
    if monomer.upper() == 'A':

        # Build projector from monomer B occupied orbitals:
        YYS = np.einsum('pa,qa,qk->pk', CB_occ, CB_occ, S, optimize=True)
        FX_int = FA_int
        FY_int = FB_int

        # --- Print matrix elements for debugging purposes ---
        term1 = FY_int.copy()
        term2 = -np.einsum('pq,qi->pi', FY_int, YYS)
        term3 = -np.einsum('qp,qi->pi', YYS, FX_int)
        term4 = np.einsum('qp,qi,ij->pj', YYS, FX_int, YYS)

        WA_elst_exch = term1 + term2 + term3 + term4
        V_lr = (WA_elst_exch+WA_elst_exch.T)/2 

        # Print components at verbosity level 4 or higher (debug functions)
        if verbosity >= 4 or print_matrix_elements:
            print('LR WA_elst_exch term1 = FB_int :\n', term1)
            print('LR WA_elst_exch term3 = - FB_int*BBS :\n', term2)
            print('LR WA_elst_exch term7 = - SBB*FA_int :\n', term3)
            print('LR WA_elst_exch term9 = SBB*FA_int*BBS :\n', term4)
    elif monomer.upper() == 'B':

        YYS = np.einsum('pa,qa,qk->pk', CA_occ, CA_occ, S, optimize=True)
        FX_int = FB_int
        FY_int = FA_int

        # --- Print matrix elements for debugging purposes ---
        term1 = FY_int.copy()
        term2 = -np.einsum('pq,qi->pi', FY_int, YYS)
        term3 = -np.einsum('qp,qi->pi', YYS, FX_int)
        term4 = np.einsum('qp,qi,ij->pj', YYS, FX_int, YYS)

        WB_elst_exch = term1 + term2 + term3 + term4
        V_lr = (WB_elst_exch+WB_elst_exch.T)/2 

        # Print components at verbosity level 4 or higher (debug functions)
        if verbosity >= 4 or print_matrix_elements:
            print('LR WB_elst_exch term1 = FA_int :\n', term1)
            print('LR WB_elst_exch term3 = - FA_int*AAS :\n', term2)
            print('LR WB_elst_exch term7 = - SAA*FB_int :\n', term3)
            print('LR WB_elst_exch term9 = SAA*FB_int*AAS :\n', term4)
    
    else:
        raise ValueError("monomer must be either \"A\" or \"B\"")

    if verbosity >=2 or print_matrix_elements:
        print(f"DEBUG LR Elst + Exch potential for monomer {monomer} :\n{V_lr}")
    
    return V_lr

def build_LM_pot(monomer, h_sapt, CA_occ, CB_occ, D, S, FA_int, TA, FB_int, TB, input_dict):
    """
    Constructs the LM potential for a specified monomer.
    
    Parameters:
      monomer (str): "A" or "B"
      h_sapt (helper_SAPT): An instance of the helper_SAPT class.
      CA_occ (numpy.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (numpy.ndarray): Occupied MO coefficients for monomer B.
      D (numpy.ndarray): Inverse MO overlap matrix.
      S (numpy.ndarray): AO overlap matrix.
      FA_int (numpy.ndarray): Interaction Fock matrix for monomer A.
      TA (numpy.ndarray): Kinetic energy matrix for monomer A.
      FB_int (numpy.ndarray): Interaction Fock matrix for monomer B.
      TB (numpy.ndarray): Kinetic energy matrix for monomer B.
      input_dict (dict): Dictionary controlling debug print options.
      
    Returns:
      LMX_pot (numpy.ndarray): The LM potential.
    """
    # Set default input_dict if not provided
    if input_dict is None:
        input_dict = fqsart_inputs_default
    verbosity = input_dict.get("debug_verbosity_level", 0)
    full_sym = input_dict.get("Full symmetrization", False)
    half_sym = input_dict.get("Half symmetrization", False)
    print_matrix_elements = input_dict.get("Print matrix elements", False)

    if verbosity >= 1:
        print(f"--Building {monomer} LM potential--")
        if full_sym:
            print(f"- Full symmetrization scheme -")
        elif half_sym:
            print(f"- Half symmetrization scheme -")
        else:
            print(f"- No symmetrization scheme -")

    nocc_A = h_sapt.ndocc_A
    if monomer.upper() == "A":
        D_AB = D[:nocc_A, nocc_A:]
        D_BB = D[nocc_A:, nocc_A:]
        XDYS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AB, CB_occ, S, optimize=True) # ADBS
        YDYS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BB, CB_occ, S, optimize=True) # BDBS
        FXTX = FA_int + TA # For A FX and TX ar FA_int and TA
        FYTY = FB_int + TB # For A FY and TY ar FB_int and TB
    elif monomer.upper() == "B":
        D_AA = D[:nocc_A, :nocc_A]
        D_BA = D[nocc_A:, :nocc_A]
        XDYS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BA, CA_occ, S, optimize=True) # BDAS
        YDYS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AA, CA_occ, S, optimize=True) # ADAS
        FXTX = FB_int + TB # For B FX and TX ar FB_int and TB
        FYTY = FA_int + TA # For B FY and TY ar FA_int and TA
    else:
        raise ValueError("monomer must be either \"A\" or \"B\"")

    if full_sym:
        FXTX = (FXTX + FXTX.T)/2
        FYTY = (FYTY + FYTY.T)/2
        XDYS = (XDYS+XDYS.T)/2 # Symmetrizing mapping operators for full_sym
        YDYS = (YDYS+YDYS.T)/2
        if verbosity >= 3 or print_matrix_elements:
            print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): FXTX, FYTY, XDYS, YDYS symmetrized for full_sym.")
    elif half_sym:
        FXTX = (FXTX + FXTX.T)/2
        FYTY = (FYTY + FYTY.T)/2
        if verbosity >= 3 or print_matrix_elements:
            print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): FXTX, FYTY symmetrized for half_sym.")

    term1 = FXTX.copy()
    term2 = -np.einsum("pq,qi -> pi", FXTX, XDYS)
    term3 = -np.einsum("pq,qi -> pi", FXTX, YDYS)
    term4 = -np.einsum("qp,qi -> pi", XDYS, FXTX)
    term5 = -np.einsum("qp,qi -> pi", YDYS, FYTY) 
    term6 = np.einsum("qp,qi,ij -> pj", XDYS, FXTX, XDYS)
    term7 = np.einsum("qp,qi,ij -> pj", XDYS, FXTX, YDYS)
    term8 = np.einsum("qp,qi,ij -> pj", YDYS, FYTY, XDYS) 
    term9 = np.einsum("qp,qi,ij -> pj", YDYS, FYTY, YDYS) 

    LMX_pot = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

    # Print total potential at verbosity level 2 or higher
    if verbosity >= 2:
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): Total LM potential (LMX_pot):\n{LMX_pot}")

    # Print components at verbosity level 3 or higher
    if verbosity >= 3 or print_matrix_elements:
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): XDYS (mapping op):\n{XDYS}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): YDYS (mapping op):\n{YDYS}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): FXTX (FX_int+TX):\n{FXTX}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): FYTY (FY_int+TY):\n{FYTY}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term1 (FXTX):\n{term1}") # This is just a copy and could be removed
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term2 (-FXTX*XDYS):\n{term2}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term3 (-FXTX*YDYS):\n{term3}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term4 (-XDYS*FXTX):\n{term4}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term5 (-YDYS*FYTY):\n{term5}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term6 (XDYS*FXTX*XDYS):\n{term6}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term7 (XDYS*FXTX*YDYS):\n{term7}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term8 (YDYS*FYTY*XDYS):\n{term8}")
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): term9 (YDYS*FYTY*YDYS):\n{term9}")

    return LMX_pot

def build_sq_potentials(monomer, h_sapt, CA, CB, S, input_dict):
    """
    Calculate SQ potentials for a specified monomer.
    
    Parameters:
      monomer (str): "A" or "B"
      h_sapt (helper_SAPT): An instance of the helper_SAPT class.
      CA (numpy.ndarray): MO coefficients for monomer A.
      CB (numpy.ndarray): MO coefficients for monomer B.
      S (numpy.ndarray): AO overlap matrix.
      input_dict (dict): Dictionary controlling calculation options.
      
    Returns:
      dict: Dictionary containing the calculated potentials.
    """
    # Set default input_dict if not provided
    if input_dict is None:
        input_dict = fqsart_inputs_default
    verbosity = input_dict.get("debug_verbosity_level", 0)
    print_matrix_elements = input_dict.get("Print matrix elements", False)
    
    # Extract potential calculation flags
    calc_elst_pot = input_dict.get("calc_elst_pot", True)
    calc_exch_pot = input_dict.get("calc_exch_pot", True)
    calc_lm = input_dict.get("calc_lm", True)
    
    # Initialize result dictionary
    result = {}
    if calc_elst_pot:
        if monomer.upper() == "A":
            nocc_B = h_sapt.ndocc_B
            CB_occ = CB[:, :nocc_B]
            D_B = 2*np.einsum('pi,qi->pq', CB_occ, CB_occ)
            mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())
            I = h_sapt.I.swapaxes(1, 2)  # ERI tensor
            VB  = np.asarray(mintsB.ao_potential()) # N-e potential for B
            JB = np.einsum('pqrs,rs->pq', I, D_B)
            WA_elst_AO = VB + JB
            result["elst_pot"] = WA_elst_AO
        elif monomer.upper() == "B":
            nocc_A = h_sapt.ndocc_A
            CA_occ = CA[:, :nocc_A]
            D_A = 2*np.einsum('pi,qi->pq', CA_occ, CA_occ)
            mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())
            I = h_sapt.I.swapaxes(1, 2)  # ERI tensor
            VA  = np.asarray(mintsA.ao_potential()) # N-e potential for A
            JA = np.einsum('pqrs,rs->pq', I, D_A)
            WB_elst_AO = VA + JA 
            result["elst_pot"] = WB_elst_AO
        else:
            raise ValueError("monomer must be either \"A\" or \"B\"")
        
        # Print total potential at verbosity level 2 or higher
        if verbosity >= 2 and not calc_exch_pot:
            print(f"DEBUG calculate_sq_potentials (Verbosity {verbosity}, Monomer {monomer}): SQ Elst potential:\n{result['elst_pot']}")
        if verbosity >= 3 and calc_exch_pot:
            print(f"DEBUG calculate_sq_potentials (Verbosity {verbosity}, Monomer {monomer}): SQ Elst potential:\n{result['elst_pot']}")
        
    # Import SQ modules only when needed
    if calc_exch_pot:
        # Import SQ modules
        from utils.omega_exch_utils import form_omega_exchange_w_sym

        # Get SQ electrostatic + exchange potential
        
        WB_exch_MO_sq, WA_exch_MO_sq = form_omega_exchange_w_sym(sapt=h_sapt, 
                                                                ca=CA, 
                                                                cb=CB,
                                                                oo_vv='S4',
                                                                ov_vo='Sinf')
        if monomer.upper() == "A":
            result["exch_pot"] = S.dot(CA).dot(WA_exch_MO_sq).dot(CA.T).dot(S)
        elif monomer.upper() == "B":
            result["exch_pot"] = S.dot(CB).dot(WB_exch_MO_sq).dot(CB.T).dot(S)
        else:
            raise ValueError("monomer must be either \"A\" or \"B\"")
        
        # Print total potential at verbosity level 2 or higher
        if verbosity >= 2 and not calc_elst_pot:
            print(f"DEBUG calculate_sq_potentials (Verbosity {verbosity}, Monomer {monomer}): SQ Elst+Exch potential:\n{result['exch_pot']}")
        if verbosity >= 3 and calc_elst_pot:
            print(f"DEBUG calculate_sq_potentials (Verbosity {verbosity}, Monomer {monomer}): SQ Elst potential:\n{result['elst_pot']}")

    
    # Calculate LM potential if requested
    if calc_lm:
        # Import SQ modules
        from utils.lm_utils import form_lm_terms_w_sym
        
        # Get SQ LM potential
        LMB_pot_MO_sq, LMA_pot_MO_sq = form_lm_terms_w_sym(sapt=h_sapt,
                                                    ca=CA,
                                                    cb=CB,
                                                    s_option='S4')
        
        if monomer.upper() == "A":
            result["lm_pot"] = S.dot(CA).dot(LMA_pot_MO_sq).dot(CA.T).dot(S)
        elif monomer.upper() == "B":
            result["lm_pot"] = S.dot(CB).dot(LMB_pot_MO_sq).dot(CB.T).dot(S)
        else:
            raise ValueError("monomer must be either \"A\" or \"B\"")
        
        # Print total potential at verbosity level 2 or higher
        if verbosity >= 2:
            print(f"DEBUG calculate_sq_potentials (Verbosity {verbosity}, Monomer {monomer}): SQ LM potential:\n{result['lm_pot']}")
    
    return result

def do_sart_fq(dimer: psi4.geometry, 
                sapt: helper_SAPT,
                input_dict: dict = fqsart_inputs_default):
    """
    Perform the SART-FQ iterative optimization of monomer orbitals.

    Parameters:
      dimer (psi4.geometry): The dimer geometry.
      sapt (helper_SAPT): An instance of the helper_SAPT class.
      input_dict (dict): Dictionary controlling calculation options.
      
    Returns:
      sart_results (dict): A dictionary containing all relevant results from the SART-FQ calculation.
        "job_name": job_name,
        "job_method": job_method,
        "job_basis": job_basis,
        "job_identifier": job_identifier,
        "system_name": system_name,
        "CA": None, # final MO matrix for A
        "CB": None, # final MO matrix for B
        "E_A": None, # final monomer A energy
        "E_B": None, # final monomer B energy
        "E1_elst": None, # electrostatic energy at iteration 1
        "E1_exch": None, # exchange energy at iteration 1
        "E_elst": None, # electrostatic energy at final iteration
        "E_exch": None, # exchange energy at final iteration
        "E_LMA": None, # DeltaLM for monomer A at final iteration
        "E_LMB": None, # DeltaLM for monomer B at final iteration
        "E_LM": None, # total LM energy at final iteration
        "dEA": None, # deformation energy for monomer A
        "dEB": None, # deformation energy for monomer B
        "E_int": None, # Interaction energy at final iteration = dEA+dEB+LM+elst+exch
        "iterations": 0, # Number of iterations performed
        "converged": False, # Converged?
        "energy_iterations_data": [], # only used if "collect_iter_data_for_plot": True
        "error": None,
        "iteration_details": [] # List of dictionaries to store detailed energy components per iteration
           
    """
    # Set default input_dict if not provided
    if input_dict is None:
        input_dict = fqsart_inputs_default
    # Extract parameters from input_dict
    verbosity = input_dict.get("debug_verbosity_level", 0)
    print_matrix_elements = input_dict.get("Print matrix elements", False)
    max_iter = input_dict.get("max_iter", 100)
    tol = input_dict.get("tol", 1.0e-8)
    
    # Extract method selection flags
    FQ_elst_exch = input_dict.get("FQ_elst_exch", True)
    FQ_LM = input_dict.get("FQ_LM", True)
    SQ_elst_exch = input_dict.get("SQ_elst_exch", False)
    SQ_LM = input_dict.get("SQ_LM", False)
    
    # Extract potential calculation flags
    calc_elst_pot = input_dict.get("calc_elst_pot", True)
    calc_exch_pot = input_dict.get("calc_exch_pot", True)
    calc_lm = input_dict.get("calc_lm", True)
    
    # Extract other options
    lr = input_dict.get("long range", False)
    plot_iter_data_collection = True  # Always collect data for potential plotting
    
    #====================================================================
    # Define all fields in the output dictionary:
    #====================================================================
    sart_results = {
        "job_name": input_dict.get("job_name", "SART-FQ"),
        "job_method": input_dict.get("job_method", "SART-FQ"),
        "job_basis": input_dict.get("job_basis", "Unknown"),
        "job_identifier": input_dict.get("job_identifier", "Unknown"),
        "system_name": input_dict.get("system_name", "Unknown"),
        "CA": None, # final MO matrix for A
        "CB": None, # final MO matrix for B
        "E_A": None, # final monomer A energy
        "E_B": None, # final monomer B energy
        "E1_elst": None, # electrostatic energy at iteration 1
        "E1_exch": None, # exchange energy at iteration 1
        "E_elst": None, # electrostatic energy at final iteration
        "E_exch": None, # exchange energy at final iteration
        "E_LMA": None, # DeltaLM for monomer A at final iteration
        "E_LMB": None, # DeltaLM for monomer B at final iteration
        "E_LM": None, # total LM energy at final iteration
        "dEA": None, # deformation energy for monomer A
        "dEB": None, # deformation energy for monomer B
        "E_int": None, # Interaction energy at final iteration = dEA+dEB+LM+elst+exch
        "iterations": 0, # Number of iterations performed
        "converged": False, # Converged?
        "iteration_details": [],
        "error": None,
        "energy_iterations_data": []
    }
    
    # Print input information at verbosity level 0 or higher
    if verbosity >= 0:
        print("\n=== Starting SART-FQ calculation ===")
        print(f"Method: {'FQ' if FQ_elst_exch else 'SQ'} for Elst+Exch, {'FQ' if FQ_LM else 'SQ'} for LM")
        print(f"Calculating: {'Elst' if calc_elst_pot else ''}{'+' if calc_elst_pot and calc_exch_pot else ''}{' Exch' if calc_exch_pot else ''}{'+' if (calc_elst_pot or calc_exch_pot) and calc_lm else ''}{' LM' if calc_lm else ''}")
        print(f"Max iterations: {max_iter}")
        print(f"Convergence tolerance: {tol}")
    
    # Get helper_SAPT object
    monomerA = dimer.extract_subsets(1,2)
    monomerB = dimer.extract_subsets(2,1)
    h_sapt = sapt
    rhfA = h_sapt.rhfA
    rhfB = h_sapt.rhfB
    
    # Intializing Energies to monomer energies
    EA = rhfA
    EB = rhfB
    
    # --- Initialize quantities for first iteration: ---
    # Initialize CA_new and CB_new with the initial MO coefficients
    CA_new = h_sapt.C_A.copy() # Ensure we work with copies
    CB_new = h_sapt.C_B.copy()
    nocc_A = h_sapt.ndocc_A
    nocc_B = h_sapt.ndocc_B
    # Get potentials and integrals:
    mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())
    mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())
    VA  = np.asarray(mintsA.ao_potential()) # N-e potential for A
    VB  = np.asarray(mintsB.ao_potential()) # N-e potential for B
    TA = np.asarray(mintsA.ao_kinetic())
    TB = np.asarray(mintsB.ao_kinetic())
    HA = VA + TA
    HB = VB + TB
    I = h_sapt.I.swapaxes(1, 2)  # ERI tensor (mu nu | p q)
    S = h_sapt.S  # AO overlap
    
    # Extract occupied MO coefficients
    CA_occ = CA_new[:, :nocc_A]
    CB_occ = CB_new[:, :nocc_B]
    
    # Compute density matrices for closed shell using einsum
    D_A = 2*np.einsum('pi,qi->pq', CA_occ, CA_occ)
    D_B = 2*np.einsum('pi,qi->pq', CB_occ, CB_occ)

    # Compute Coulomb J and Exchange K for A
    JA = np.einsum('pqrs,rs->pq', I, D_A)
    KA = np.einsum('prqs,rs->pq', I, D_A)
    FA = HA + JA - 0.5 * KA

    # Compute Coulomb J and Exchange K for B
    JB = np.einsum('pqrs,rs->pq', I, D_B)
    KB = np.einsum('prqs,rs->pq', I, D_B)
    FB = HB + JB - 0.5 * KB
    
    # Compute inverse MO overlap matrix
    D = compute_D_matrix(CA_occ, CB_occ, S)
    
    # Compute E1 with initial MO coefficients
    Eelex_init, Eelst_fq_init, Eexch_fq_init = compute_Eelex(CA_new, CB_new, dimer, h_sapt, D, input_dict)
    
    # Store initial E1 values
    sart_results["E1_elex"] = Eelex_init
    sart_results["E1_elst"] = Eelst_fq_init
    sart_results["E1_exch"] = Eexch_fq_init
    
    # Initialize variables for iteration
    E_int_prev = None
    converged = False
    iteration = 0
    
    # Print initial values at verbosity level 1 or higher
    if verbosity >= 1:
        print("\n--- Initial values ---")
        print(f"E1_init = {Eelex_init} Eh")
        print(f"Eelst_fq_init = {Eelst_fq_init} Eh")
        print(f"Eexch_fq_init = {Eexch_fq_init} Eh")
        print(f"EA_init = {EA} Eh")
        print(f"EB_init = {EB} Eh")
    
    # Start SART-FQ iterations
    while not converged and iteration < max_iter:
        iteration += 1
        
        if verbosity >= 1:
            print(f"\n###### Iteration {iteration} ######")
                
        # Compute interaction operators
        # Pass I_AO (mu nu | p q) to build_F_int, it will handle contractions
        FA_int = build_F_int(CA_new, CB_new, h_sapt, D, monomer="A", input_dict=input_dict)
        FB_int = build_F_int(CA_new, CB_new, h_sapt, D, monomer="B", input_dict=input_dict)

        if lr and verbosity >=1: # Long range checks are for debug 
            print("\n--- DEBUG: Computing the long range of the potentials ---")
            FA_int_lr = build_F_int_lr(h_sapt, monomer="A", input_dict=input_dict)
            FB_int_lr = build_F_int_lr(h_sapt, monomer="B", input_dict=input_dict)
            V_lr_A = build_elst_exch_pot_lr("A", CA_occ, CB_occ, FA_int_lr, FB_int_lr, S, input_dict=input_dict)
            V_lr_B = build_elst_exch_pot_lr("B", CA_occ, CB_occ, FA_int_lr, FB_int_lr, S, input_dict=input_dict)
        
        # Compute Elst + Exch potentials
        if verbosity >= 1:
            print("\n--- Computing potentials based on calculation flags ---")
        # Compute potentials based on selected methods and flags
        WA_pot = None
        WB_pot = None
        LMA_pot = None
        LMB_pot = None
        
        # Calculate electrostatic and exchange potentials
        if calc_elst_pot or calc_exch_pot:
            if FQ_elst_exch:
                # Use FQ method for electrostatic and exchange potentials
                if calc_elst_pot and calc_exch_pot:
                    # Calculate combined electrostatic + exchange potential
                    WA_pot = build_elst_exch_pot("A", h_sapt, CA_occ, CB_occ, D, S, FA_int, FB_int, input_dict)
                    WB_pot = build_elst_exch_pot("B", h_sapt, CA_occ, CB_occ, D, S, FA_int, FB_int, input_dict)
                    # Print the components at vrbosity level 3 or higher
                    if verbosity >=3:
                        # Calculate only electrostatic potential
                        WA_elst = build_elst_pot("A", h_sapt, CA_occ, CB_occ, input_dict)
                        WB_elst = build_elst_pot("B", h_sapt, CA_occ, CB_occ, input_dict)
                        print(f'DEBUG Elst pot (Verbosity {verbosity}, Monomer A): WA_elst:\n{WA_elst}')
                        print(f'DEBUG Elst pot (Verbosity {verbosity}, Monomer B): WB_elst:\n{WB_elst}')
                        WA_exch = WA_pot - WA_elst
                        WB_exch = WB_pot - WB_elst
                        print(f"DEBUG Exch pot (Verbosity {verbosity}, Monomer A): \n{WA_exch}")
                        print(f"DEBUG Exch pot (Verbosity {verbosity}, Monomer B): \n{WB_exch}")
                    # Print total potential at verbosity level 2 or higher
                    if verbosity >= 2:
                        print(f'DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer A): Total Elst+Exch potential (WA_elst_exch):\n{WA_pot}')
                        print(f'DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer B): Total Elst+Exch potential (WB_elst_exch):\n{WB_pot}')

                elif calc_elst_pot:
                    # Calculate only electrostatic potential
                    WA_pot = build_elst_pot("A", h_sapt, CA_occ, CB_occ, input_dict)
                    WB_pot = build_elst_pot("B", h_sapt, CA_occ, CB_occ, input_dict)
                    # Print total potential at verbosity level 2 or higher
                    if verbosity >= 2:
                        print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer A): Total electrostatic potential (WA_elst):\n{WA_pot}")
                        print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer B): Total electrostatic potential (WB_elst):\n{WB_pot}")
                elif calc_exch_pot:
                    # Calculate exchange potential by subtracting electrostatic from total
                    WA_elst = build_elst_pot("A", h_sapt, CA_occ, CB_occ, input_dict)
                    WB_elst = build_elst_pot("B", h_sapt, CA_occ, CB_occ, input_dict)
                    WA_elst_exch = build_elst_exch_pot("A", h_sapt, CA_occ, CB_occ, D, S, FA_int, FB_int, input_dict)
                    WB_elst_exch = build_elst_exch_pot("B", h_sapt, CA_occ, CB_occ, D, S, FA_int, FB_int, input_dict)
                    WA_pot = WA_elst_exch - WA_elst
                    WB_pot = WB_elst_exch - WB_elst
                    # Print total potential at verbosity level 2 or higher
                    if verbosity >= 2:
                        print(f"DEBUG build_elst_exch_pot - build_elst_pot (Verbosity {verbosity}, Monomer A): Total Exchange potential :\n{WA_tot}")
                        print(f"DEBUG build_elst_exch_pot - build_elst_pot (Verbosity {verbosity}, Monomer B): Total Exchange potential :\n{WB_tot}")
            
            elif SQ_elst_exch:
                # Use SQ method for electrostatic and exchange potentials

                if verbosity >= 1:
                    print(f"Calculate_sq_potentials (Verbosity {verbosity}, Monomer A): Calculating SQ Elst + Exch potential")
                sq_potentials_A = build_sq_potentials("A", h_sapt, CA_new, CB_new, S, input_dict)
                if verbosity >= 1:
                    print(f"Calculate_sq_potentials (Verbosity {verbosity}, Monomer B): Calculating SQ Elst + Exch potential")
                sq_potentials_B = build_sq_potentials("B", h_sapt, CA_new, CB_new, S, input_dict)
                
                if "elst_pot" in sq_potentials_A and "elst_pot" in sq_potentials_B:
                    WA_pot = sq_potentials_A["elst_pot"]
                    WB_pot = sq_potentials_B["elst_pot"]
                if "exch_pot" in sq_potentials_A and "exch_pot" in sq_potentials_B:
                    WA_pot += sq_potentials_A["exch_pot"]
                    WB_pot += sq_potentials_B["exch_pot"]
                if calc_elst_pot and calc_exch_pot and verbosity >= 2:
                    print(f"DEBUG calculate_sq_potentials (Verbosity {verbosity}, Monomer A): SQ Elst Exch potential:\n{WA_pot}")
                    print(f"DEBUG calculate_sq_potentials (Verbosity {verbosity}, Monomer B): SQ Elst Exch potential:\n{WB_pot}")
        
        # Calculate LM potentials
        if calc_lm:
            if FQ_LM:
                # Use FQ method for LM potential
                LMA_pot = build_LM_pot("A", h_sapt, CA_occ, CB_occ, D, S, FA_int, TA, FB_int, TB, input_dict) - FA
                LMB_pot = build_LM_pot("B", h_sapt, CA_occ, CB_occ, D, S, FA_int, TA, FB_int, TB, input_dict) - FB
            
            elif SQ_LM:
                # Use SQ method for LM potential
                if verbosity >= 1:
                    print(f"Calculate_sq_potentials (Verbosity {verbosity}, Monomer A): Calculating SQ LM potential")
                sq_potentials_A = build_sq_potentials("A", h_sapt, CA_new, CB_new, S, input_dict)

                if verbosity >= 1:
                    print(f"Calculate_sq_potentials (Verbosity {verbosity}, Monomer B): Calculating SQ LM potential")
                sq_potentials_B = build_sq_potentials("B", h_sapt, CA_new, CB_new, S, input_dict)
                
                if "lm_pot" in sq_potentials_A and "lm_pot" in sq_potentials_B:
                    LMA_pot = sq_potentials_A["lm_pot"]
                    LMB_pot = sq_potentials_B["lm_pot"]
        
        # Combine potentials
        if WA_pot is None:
            WA_pot = np.zeros_like(S)
        if WB_pot is None:
            WB_pot = np.zeros_like(S)
        
        if calc_lm:
            WA_tot = WA_pot + LMA_pot
            WB_tot = WB_pot + LMB_pot
        else:
            WA_tot = WA_pot
            WB_tot = WB_pot
        
        # Symmetrize total potential
        WA_tot_sym = (WA_tot + WA_tot.T)/2
        WB_tot_sym = (WB_tot + WB_tot.T)/2
        
        if verbosity >=2:
            print(f"---- Total Interaction potential for A ----\n{WA_tot_sym}")
            print(f"---- Total Interaction potential for B ----\n{WB_tot_sym}")
        
        # Print iteration info at verbosity level 1 or higher
        if verbosity >= 1:
            print(f"--- Iteration {iteration}: SCF Update ---")

        # =====================
        #  SCF for Monomer A  #
        # =====================
        if verbosity >=1:
            scf_A_timer = sapt_timer('SCF-iterations, A') 
        scfA_psi4_text = f"""
========================================== 
=               Monomer A                
=          Update itr {iteration}           
==========================================
        """
        psi4.core.print_out(scfA_psi4_text)
        psi4.core.print_out('\n ...Psi4 will run SCF iterations now')
        try:      
            wfnA_scf = do_scf_itr(monomer= monomerA,
                                        reference= 'RHF',
                                        guess= (CA_occ, CA_occ),
                                        omega= WA_tot_sym,#V_lr_A,
                                        maxiter= 300,
                                        input_dict = input_dict
                                        )
            # Get updated MO coefficients
            # These will be used in (i+1)th itr
            CA_new = wfnA_scf.Ca().to_array()
            # Get updated energies
            EA_new = wfnA_scf.compute_energy()
        except Exception as e:
            print(e)
            continue            # continues to the later part of the code
        psi4.core.print_out(scfA_psi4_text)
        psi4.core.print_out('\n ...Finished SCF iterations now')
        if verbosity >=1:
            scf_A_timer.stop()        
        
        # =====================
        #  SCF for Monomer B  #
        # =====================

        if verbosity >=1:
            scf_B_timer = sapt_timer('SCF-iterations, B') 
        scfB_psi4_text = f"""
========================================== 
=               Monomer B                
=          Update itr {iteration}           
==========================================
"""
        psi4.core.print_out(scfB_psi4_text)
        psi4.core.print_out('\n ...Psi4 will run SCF iterations now')
        # Modified C coefficient matrix for Monomer B
        try:
            wfnB_scf = do_scf_itr(monomer= monomerB,
                                        reference= 'RHF',
                                        guess= (CB_occ, CB_occ),
                                        omega= WB_tot_sym,#V_lr_B,
                                        maxiter= 300,
                                        input_dict = input_dict
                                        )
            # Get updated MO coefficients
            # These will be used in (i+1)th itr
            CB_new = wfnB_scf.Ca().to_array()        
            # Get updated energies
            EB_new = wfnB_scf.compute_energy()
        except Exception as e:
            print(e)
            continue
        psi4.core.print_out(scfB_psi4_text)
        psi4.core.print_out('\n ...Finished SCF iterations now')
        if verbosity >=1:
            scf_B_timer.stop()    
    
        # --- Initialize quantities for next iteration: ---
        # Use CA_new and CB_new in the  next computation
        CA_occ = CA_new[:, :nocc_A]
        CB_occ = CB_new[:, :nocc_B]

        # Compute density matrices using einsum
        D_A = 2*np.einsum('pi,qi->pq', CA_occ, CA_occ)
        D_B = 2*np.einsum('pi,qi->pq', CB_occ, CB_occ)

        # Compute Coulomb J and Exchange K for A
        JA = np.einsum('pqrs,rs->pq', I, D_A)
        KA = np.einsum('prqs,rs->pq', I, D_A)
        # Build the next FA operator
        FA = HA + JA - 0.5 * KA

        # Compute Coulomb J and Exchange K for B
        JB = np.einsum('pqrs,rs->pq', I, D_B)
        KB = np.einsum('prqs,rs->pq', I, D_B)
        # Build the next FB operator
        FB = HB + JB - 0.5 * KB

        # Recompute D matrix with the updated occupied orbitals
        D = compute_D_matrix(CA_occ, CB_occ, S)    

        # Compute deformation energies
        dEA = EA_new - rhfA
        dEB = EB_new - rhfB
        
        # Compute E1 with updated MO coefficients
        Eelex_fq, Eelst_fq_iter, Eexch_fq_iter = compute_Eelex(CA_new, CB_new, dimer, h_sapt, D, input_dict)
        
        # Compute delta LM energies
        delta_LMA_iter = compute_delta_LMX("A", CA_new, CB_new, h_sapt, D, HA, I, FA, input_dict)
        delta_LMB_iter = compute_delta_LMX("B", CA_new, CB_new, h_sapt, D, HB, I, FB, input_dict)
        
        delta_LM_iter = delta_LMA_iter + delta_LMB_iter
        # Compute total interaction energy
        E_int_current = dEA + dEB + Eelex_fq + delta_LM_iter
        
        # Print iteration results at verbosity level 1 or higher
        if verbosity >= 1:
            print(f"\n===Iteration {iteration} results===")
            print(f"E_int = {E_int_current} Eh")
            print(f"E_elst = {Eelst_fq_iter} Eh")
            print(f"E_exch = {Eexch_fq_iter} Eh")
            print(f"dEA = {dEA} Eh")
            print(f"dEB = {dEB} Eh")
            print(f"delta_LMA = {delta_LMA_iter} Eh")
            print(f"delta_LMB = {delta_LMB_iter} Eh")
            print(f"delta_LM = {delta_LM_iter} Eh")

            if E_int_prev is not None:
                print(f"Energy change: {E_int_current - E_int_prev} Eh")
        
        # Store data for plotting/analysis
        iter_details = {
            "Iteration": iteration,
            "dEA": dEA,
            "dEB": dEB,
            "E_elst": Eelst_fq_iter,
            "E_exch": Eexch_fq_iter,
            "E_LM": delta_LM_iter, # Combined LM energy
            "E_LMA": delta_LMA_iter,
            "E_LMB": delta_LMB_iter,
            "E_int": E_int_current,
            "E_def_tot[A]": dEA + delta_LMA_iter,
            "E_def_tot[B]": dEB + delta_LMB_iter
        }
        sart_results["iteration_details"].append(iter_details)

        #if plot_iter_data_collection:
        #    # This format matches what plot_energy_iterations_direct expects
        #    sart_results["energy_iterations_data"].append([
        #        iteration,
        #        dEA,
        #        dEB,
        #        Eelst_fq_iter, 
        #        Eexch_fq_iter, 
        #        delta_LM_iter, # Combined Delta LM
        #        E_int_current # Total Interaction Energy
        #    ])
        # Store detailed iteration data in a standardized dictionary format:
        if plot_iter_data_collection:
            energy_iter_row = {
                "Iteration": iteration,
                "dEA": dEA,
                "dEB": dEB,
                "E_elst": Eelst_fq_iter,
                "E_exch": Eexch_fq_iter,
                "E_LM": delta_LM_iter,      # Combined LM energy (or however you want to store it)
                "E_LMA": delta_LMA_iter,
                "E_LMB": delta_LMB_iter,
                "E_int": E_int_current,
                "E_def_tot[A]": dEA + delta_LMA_iter,
                "E_def_tot[B]": dEB + delta_LMB_iter
            }
            sart_results["energy_iterations_data"].append(energy_iter_row)

        # Check for convergence
        if E_int_prev is not None and abs(E_int_current - E_int_prev) < tol:
            converged = True
            if verbosity >= 0: 
                print(f"\nSART-FQ converged in {iteration} iterations.")
        
        E_int_prev = E_int_current
        
        if not converged and iteration == max_iter and verbosity >= 1:
            print(f"\nSART-FQ did not converge after {max_iter} iterations.")
    
    # Final values to store
    sart_results["CA"] = CA_new
    sart_results["CB"] = CB_new
    sart_results["E_A"] = EA_new
    sart_results["E_B"] = EB_new
    sart_results["E_elst"] = Eelst_fq_iter # Last computed values
    sart_results["E_exch"] = Eexch_fq_iter
    sart_results["E_LMA"] = delta_LMA_iter
    sart_results["E_LMB"] = delta_LMB_iter
    sart_results["E_LM"] = delta_LM_iter # Total LM energy
    sart_results["E_int"] = E_int_current
    sart_results["iterations"] = iteration
    sart_results["converged"] = converged
    sart_results["dEA"] = dEA # Deformation energy A
    sart_results["dEB"] = dEB # Deformation energy B

    # Print final results at verbosity level 0 or higher (minimal output)
    if verbosity >= 0:
        print("\n--- Final SART-FQ Results ---")
        print(f"Converged: {sart_results['converged']}")
        print(f"Iterations: {sart_results['iterations']}")
        print(f"Final Interaction Energy (E_int): {sart_results['E_int']} Eh")
        print(f"Final HF Energy of A (EA): {sart_results['E_A']} Eh")
        print(f"Final HF Energy of B (EB): {sart_results['E_B']} Eh")
        print(f"Final Deformation Energy of A (dEA): {sart_results['dEA']} Eh")
        print(f"Final Deformation Energy of B (dEB): {sart_results['dEB']} Eh")
        print(f"Final Electrostatic Energy (E_elst): {sart_results['E_elst']} Eh")
        print(f"Final Exchange Energy (E_exch): {sart_results['E_exch']} Eh")
        print(f"Final Delta LM Energy for A (D_LMA): {sart_results['E_LMA']} Eh")
        print(f"Final Delta LM Energy for B (D_LMB): {sart_results['E_LMB']} Eh")
        print(f"Final Delta LM Energy (= D_LMA + D_LMB): {sart_results['E_LM']} Eh")
        print(f"Iteration 1 Electrostatic Energy (E_elst_1): {sart_results['E1_elst']} Eh")
        print(f"Iteration 1 Exchange Energy (E_exch_1): {sart_results['E1_exch']} Eh")
    
    return sart_results

def do_scf_itr(monomer: psi4.geometry, 
                reference: str,
                guess: tuple,
                omega: np.ndarray, 
                maxiter: int = 100, 
                input_dict: dict = fqsart_inputs_default,
                diis_bool: bool = True):
    """
    Takes the monomer info + Creates a wavefunction object 
    Creates RHF/HF/DFT object from the wavefunction
    Assigns the guess orbitals explicitly
    Adds Omega as the external perturbation
    Runs SCF iteration 
    Returns the wavefuncn(/RHF?) object 
    
    **** Read/Write wavefunc is outside this method.
    """
    verbosity = input_dict.get("debug_verbosity_level", 0)
    
    # Print function entry at verbosity level 1 or higher
    if verbosity >= 1:
        print('Entering DO SCF...')
        print('........will start scf-iterations')
    
    psi4.set_options({'MAXITER': maxiter})
    
    # Constructing base wavefunction and then RHF/HF object
    base_wfn = psi4.core.Wavefunction.build(monomer, 
                                        psi4.core.get_global_option('BASIS'))
    
    # Print debug info at verbosity level 3 or higher
    if verbosity >= 3:
        print('Base WFN constructed...')
    
    wfn_ref_obj = psi4.driver.scf_wavefunction_factory('SCF', 
                                                ref_wfn=base_wfn,
                                                reference=reference)
    
    # Print debug info at verbosity level 3 or higher
    if verbosity >= 3:
        print('RHF object constructed...')
    
    # Access the GUESS and set these
    Ca = guess[0]
    Cb = guess[1]
    Ca_psi4_mat = psi4.core.Matrix.from_array(Ca)
    Cb_psi4_mat = psi4.core.Matrix.from_array(Cb)
    
    # Print debug info at verbosity level 3 or higher
    if verbosity >= 3:
        print('GUESS are extracted...')
    
    wfn_ref_obj.guess_Ca(Ca_psi4_mat)
    wfn_ref_obj.guess_Cb(Cb_psi4_mat)
    
    # Print debug info at verbosity level 3 or higher
    if verbosity >= 3:
        print('GUESS are set...')

    # Initialize for SCF Run
    wfn_ref_obj.initialize()
    
    # Print debug info at verbosity level 3 or higher
    if verbosity >= 3:
        print('After initializing.....Check if it has the correct GUESS loaded')
        print('CA matrix =', wfn_ref_obj.Ca().to_array())
        print('CB matrix =', wfn_ref_obj.Cb().to_array())

    # Prepare the Omega matrix
    if verbosity >= 3:
        print('Omega to be added to Fock')
    
    Omega_psi4_mat = psi4.core.Matrix.from_array(omega)
    wfn_ref_obj.push_back_external_potential(Omega_psi4_mat)

    # Start the SCF runs and save
    wfn_ref_obj.iterations()
    wfn_ref_obj.save_density_and_energy()

    # Print debug info at verbosity level 3 or higher
    if verbosity >= 3:
        print('SCF iterations done')
        #print('Energy = ', wfn_ref_obj.energy())
        #print('CA matrix =', wfn_ref_obj.Ca().to_array())
        #print('CB matrix =', wfn_ref_obj.Cb().to_array())
    
    return wfn_ref_obj

def plot_energy_iterations_direct(energy_iterations_data, title=None, save_path=None):
    """
    Plot energy components over iterations directly from data.
    
    Parameters:
      energy_iterations_data (list): List of lists containing energy data for each iteration.
                                    Format: [iteration, dEA, dEB, E_elst, E_exch, E_lm, E_int]
      title (str): Title for the plot.
      save_path (str): Path to save the plot.
      
    Returns:
      None
    """
    # Convert data to numpy array for easier manipulation
    data = np.array(energy_iterations_data)
    
    # Extract data columns
    iterations = data[:, 0]
    dEA = data[:, 1]
    dEB = data[:, 2]
    E_elst = data[:, 3]
    E_exch = data[:, 4]
    E_lm = data[:, 5]
    E_int = data[:, 6]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot energy components
    ax.plot(iterations, E_elst, 'b-', label='E_elst')
    ax.plot(iterations, E_exch, 'r-', label='E_exch')
    ax.plot(iterations, dEA, 'g--', label='dEA')
    ax.plot(iterations, dEB, 'm--', label='dEB')
    ax.plot(iterations, E_lm, 'c-', label='E_lm')
    ax.plot(iterations, E_int, 'k-', linewidth=2, label='E_int')
    
    # Set labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy (Eh)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('SART-FQ Energy Components')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show plot
    plt.tight_layout()
    plt.show()

def from_file(file_path, title=None, save_path=None):
    """
    Read energy data from file and plot energy components over iterations.
    
    Parameters:
      file_path (str): Path to the file containing energy data.
      title (str): Title for the plot.
      save_path (str): Path to save the plot.
      
    Returns:
      None
    """
    # Initialize data list
    energy_iterations_data = []
    
    # Read data from file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse data
    for line in lines:
        if line.strip() and not line.startswith('#'):
            try:
                # Try to parse line as space-separated values
                values = [float(val) for val in line.strip().split()]
                if len(values) >= 7:  # Ensure we have all required values
                    energy_iterations_data.append(values)
            except ValueError:
                # Skip lines that can't be parsed
                continue
    
    # Plot data
    if energy_iterations_data:
        plot_energy_iterations_direct(energy_iterations_data, title, save_path)
    else:
        print(f"No valid data found in {file_path}")
