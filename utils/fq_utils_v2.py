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
from utils.helper_SAPT import helper_SAPT, sapt_timer
from utils.pb_utils import get_elst, get_exch
from utils.omega_exch_utils import form_omega_exchange_w_sym
from utils.lm_utils import form_lm_terms_w_sym

fqsart_inputs_default = {
    'long range': False,  # If True compute and prints the long range form of the interaction operator, as well as the long range form of the potential
    'SQ elst exch': False, # If True compute, print and use in the iterations the SQ elst+exch potential and compare it with FQ
    'SQ LM': False,        # If True compute, print and use in the iterations the SQ LM potential and compare it with FQ
    'FQ elst exch': False, # If this and SQ Elst Exch are both True, both potential will be computed, but the FQ will be used instead, as default 
    'FQ LM': False,       # If this and SQ LM are both True, both potential will be computed, but the FQ will be used instead, as default
    'Full symmetrization': False,  # Symmetrize both the FX_int operator as well as the "mapping operators" when computing the potentials
    'Half symmetrization': False,  # Symmetrize only the FX_int operator when computing the potentials
    'Print matrix elements': False, # kept for retrocpatibility, but debug_verbosity_level=3 is instead reccomanded
    'debug_print_potentials': False, # For printing FQ/SQ potentials during iterations
    'debug_print_energies_iter': False, # For printing energy components at each SART iteration
    'debug_print_sq_comparison': False, # For printing comparisons between FQ and SQ results if SQ is computed
    'debug_check_asymptotic_behaviour': False, # For printing related to asymptotic behaviour checks (e.g. long-range potentials)
    'debug_verbosity_level': 0, # 0: Minimal, 1: Scalars + Totals, 2: Scalars + Totals + Components, 3: All the details (equivalent to Print matrix elements)
    'compute_sq_potential_also': False # Flag to control if SQ potential is computed for comparison
}

def compute_E1(CA, CB, dimer, sapt, D, input_dict, maxiter=10, geom_index=None):
    """
    Computes E^{(1)} for the given dimer geometry.
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
    - E1 (float): The computed value of E^{(1)}.
    - Eelst_fq (float): The first-order electrostatic energy (FQ).
    - Eexch_fq (float): The first-order exchange energy (FQ).
    """
    h_sapt = sapt

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
    D_rp = D[:, :nocc_A]  # Shape: nocc_total x nocc_A
    D_sq = D[:, nocc_A:]  # Shape: nocc_total x nocc_B

    # Compute Term1 = <a|U_B|r>*D_ra and Term2 = <b|U_A|r>*D_rb
    term1 = 2*np.einsum('rp, rp ->', D_rp, UB_MO.T)
    term2 = 2*np.einsum('sq, sq ->', D_sq, UA_MO.T)

    # Compute two-electron integrals in AO basis from h_sapt
    I_orig = h_sapt.I 
    I = I_orig.swapaxes(1, 2) # Adjust axes to match (n_basis x n_basis x n_basis x n_basis)

    # Prepare MO coefficient matrices for occupied MOs
    C_p = CA_occ  # n_basis x nocc_A (p in monomer A)
    C_q = CB_occ  # n_basis x nocc_B (q in monomer B)
    C_r = C_occ   # n_basis x nocc_total (r in all occupied MOs)
    C_s = C_occ   # n_basis x nocc_total (s in all occupied MOs)

    # Compute (pr|qs)
    eri_prqs = oe.contract('prqs,pi,rj,qk,sl->ijkl',
                           I, C_p, C_r, C_q, C_s)
    # Shape: nocc_A x nocc_total x nocc_B x nocc_total

    # Compute (ps|qr)
    eri_psqr = oe.contract('prqs,pi,rj,qk,sl->ilkj',
                           I, C_p, C_r, C_q, C_s)
    # Shape: nocc_A x nocc_total x nocc_B x nocc_total
    
    # Electrostatic FQ term component from two-electron integrals
    elst_2e_fq = 4 * np.einsum('rp, sq, prqs ->', D_rp, D_sq, eri_prqs)
    
    Eelst_fq = W_AB + term1 + term2 + elst_2e_fq
    Eexch_fq = -2 * np.einsum('rp, sq, prqs ->', D_rp, D_sq, eri_psqr)
    
    E1 = Eelst_fq + Eexch_fq

    verbosity = input_dict.get("debug_verbosity_level", 0)

    if verbosity >= 1:
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): W_AB = {W_AB} Eh")
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): FQ E_elst (total) = {Eelst_fq} Eh")
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): FQ E_exch (total) = {Eexch_fq} Eh")
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): FQ E1 (Eelst_fq + Eexch_fq) = {E1} Eh")

    if verbosity >= 2:
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): term1_val (<a|U_B|r> part) = {term1}")
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): term2_val (<b|U_A|s> part) = {term2}")
        print(f"DEBUG compute_E1 (Verbosity {verbosity}): FQ E_elst component (2e) = {elst_2e_fq}")

    # SQ comparison prints remain conditional on their specific flags, independent of main verbosity for now
    if input_dict.get("SQ elst exch", False) and input_dict.get("debug_print_sq_comparison", False):
        sqEelst = get_elst(CA_occ, CB_occ, I, VA, VB, W_AB) 
        sqEexch = get_exch(h_sapt, CA, CB)
        
        print(f"DEBUG compute_E1 (SQ Comparison): SQ E_elst = {sqEelst}")
        print(f"DEBUG compute_E1 (SQ Comparison): SQ E_exch = {sqEexch}")
        sqE1_val = sqEelst + sqEexch
        print(f"DEBUG compute_E1 (SQ Comparison): SQ E1 = {sqE1_val}")
        diff_fq_sq = E1 - sqE1_val
        print(f"DEBUG compute_E1 (SQ Comparison): E1 diff fq-sq = {diff_fq_sq}")
        if sqE1_val != 0:
            ratio_fq_sq = E1 / sqE1_val
            print(f"DEBUG compute_E1 (SQ Comparison): E1 ratio fq/sq = {ratio_fq_sq}")
        else:
            print("DEBUG compute_E1 (SQ Comparison): SQ E1 is zero, ratio fq/sq cannot be computed.")

    return E1, Eelst_fq, Eexch_fq

def compute_delta_LMX(monomer, CA, CB, h_sapt, D, H, I, F, input_dict):
    """
    Computes the sum of Murell and Landshoff delta term for a selected monomer using the supplied one-electron
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
    
    Returns:
      delta_LX (float): The computed Landshoff delta value for the given monomer.
    """
    #print(f'\n--- Entering Compute Delta L{monomer}--- \n')
    # Retrieve number of occupied orbitals for each monomer.
    nocc_A = h_sapt.ndocc_A   # Number of occupied MOs in monomer A
    nocc_B = h_sapt.ndocc_B   # Number of occupied MOs in monomer B
    # Extract the occupied parts of the MO coefficient matrices.
    CA_occ = CA[:, :nocc_A]               # Shape: (n_basis x nocc_A)
    CB_occ = CB[:, :nocc_B]               # Shape: (n_basis x nocc_B)

    # Form the full occupied block from both monomers.
    C_occ = np.hstack((CA_occ, CB_occ))    # Shape: (n_basis x n_occ_total)
    
    #-- set the D matrix to be the identity matrix for debugging --
    #nocc = nocc_A + nocc_B
    #D = np.eye(nocc)
    if monomer.upper() == 'A':
        # Select the slice of D corresponding to monomer A's occupied orbitals.
        D_block = D[:, :nocc_A]           # Shape: (n_occ_total x nocc_A)
        
        # Compute the common intermediate object for monomer A.
        X = np.einsum('qr,ra,pa->qp', C_occ, D_block, CA_occ)  # Shape: (n_basis x nbasis)

        # Diagonal (trace) contribution using the Fock operator.
        term2  = np.einsum('mp,mn,np->', CA_occ, F, CA_occ) 
        term2  += np.einsum('mp,mn,np->', CA_occ, H, CA_occ)

        ## Compute half the density matrix for alternatives and other purposes
        #Den = np.einsum('pr,qr->pq', CA_occ, CA_occ)
        ## --- Alternative computation ---
        ## Let's try to compute the delta L in a different way, by first transforming the matrices 
        ## and then contracting them with D block:
        #H_MO1 = np.einsum('pa,pq,qr->ar', CA_occ, H, CA_occ)
        #H_MO2 = np.einsum('pa,pq,qr->ar', CA_occ, H, CB_occ)
        #I_MO1 = np.einsum('pqlm,pa,qr,lb,ms->arbs', I, CA_occ, CA_occ, CA_occ, CA_occ)
        #I_MO2 = np.einsum('pqlm,pa,qr,lb,ms->arbs', I, CA_occ, CA_occ, CA_occ, CB_occ)
        #I_MO3 = np.einsum('pqlm,pa,qr,lb,ms->arbs', I, CA_occ, CB_occ, CA_occ, CA_occ)
        #I_MO4 = np.einsum('pqlm,pa,qr,lb,ms->arbs', I, CA_occ, CB_occ, CA_occ, CB_occ)


        ## Compute the common intermediates by blocks of the D matrix
        #D_1 = D[:nocc_A, :nocc_A] # AxA
        #D_2 = D[nocc_A:, :nocc_A] # BXA
        #X1 = np.einsum('qr,ra,pa->qp', CA_occ, D_1, CA_occ)  # Shape: (n_basis x nbasis)
        #X2 = np.einsum('qr,ra,pa->qp', CB_occ, D_2, CA_occ)  # Shape: (n_basis x nbasis)
        #X_alt = X1 + X2
        ## --- print X and X alt ---
        #print(f'Test if the product is equivalent using X or X1 and X2:')
        #print('X for A:\n', X)
        #print('X_alt = X1 + X2 for A:\n', X_alt)

    elif monomer.upper() == 'B':
        # Select the D block corresponding to monomer B's occupied orbitals.
        D_block = D[:, nocc_A:]           # Shape: (n_occ_total x nocc_B)
        
        # Compute the common intermediate object for monomer B.
        X = np.einsum('qr,rb,pb->qp', C_occ, D_block, CB_occ)  # Shape: (n_basis x nbasis)

        # Diagonal (trace) contribution using the Fock operator.
        term2  = np.einsum('mp,mn,np->', CB_occ, F, CB_occ) 
        term2  += np.einsum('mp,mn,np->', CB_occ, H, CB_occ)

        ## Compute the density matrix for alternatives and other purposes
        #Den = np.einsum('pr,qr->pq', CB_occ, CB_occ)
        ## --- Alternative computation ---
        ## Let's try to compute the delta L in a different way, by first transforming the matrices 
        ## and then contracting them with D block:
        #H_MO1 = np.einsum('pb,pq,qr->br', CB_occ, H, CA_occ)
        #H_MO2 = np.einsum('pb,pq,qr->br', CB_occ, H, CB_occ)
        #I_MO1 = np.einsum('pqlm,pa,qr,lb,ms->arbs', I, CB_occ, CA_occ, CB_occ, CA_occ)
        #I_MO2 = np.einsum('pqlm,pa,qr,lb,ms->arbs', I, CB_occ, CA_occ, CB_occ, CB_occ)
        #I_MO3 = np.einsum('pqlm,pa,qr,lb,ms->arbs', I, CB_occ, CB_occ, CB_occ, CA_occ)
        #I_MO4 = np.einsum('pqlm,pa,qr,lb,ms->arbs', I, CB_occ, CB_occ, CB_occ, CB_occ)

        ## Compute the common intermediates by blocks of the D matrix
        #D_1 = D[:nocc_A, nocc_A:] # AxB
        #D_2 = D[nocc_A:, nocc_A:] # BxB
        #X1 = np.einsum('qr,ra,pa->qp', CA_occ, D_1, CB_occ)  # Shape: (n_basis x nbasis)
        #X2 = np.einsum('qr,ra,pa->qp', CB_occ, D_2, CB_occ)  # Shape: (n_basis x nbasis)
        #X_alt = X1 + X2
        ## --- Print X and X alt ---
        ##print(f'Test if the product is equivalent using X or X1 and X2:')
        ##print('X for B:\n', X)
        ##print('X_alt = X1 + X2 for B:\n', X_alt)
    else:
        raise ValueError("monomer must be 'A' or 'B'")
    
    # --- Standard computation ---
    # Compute term11, term12 and term13 using the combined intermediate. 
    # This should be the most computetional efficient way of computing it (!to be proved).
    term11 = 2 * np.einsum('pq,qp->', H, X)
    term12 = 2 * np.einsum('pqrs,qp,sr->', I, X, X)
    term13 = - np.einsum('psrq,qp,sr ->', I, X, X) # written with the convential chemist index ordering for exchange.
    term1  = term11 + term12 + term13
    

    # Compute the final Landshoff Delta.
    delta_LMX = term1 - term2

    verbosity = input_dict.get("debug_verbosity_level", 0)
    print_matrix_elements = input_dict.get("Print matrix elements", False)

    if verbosity >= 1:
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): delta_LMX = {delta_LMX} Eh")

    if verbosity >= 2:
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term1 (H,I component sum) = {term1}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term2 (F,H trace component) = {term2}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term11 (2*<H*X>) = {term11}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term12 (2*<X*J*X>) = {term12}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): term13 (-<X*K*X>) = {term13}")
        print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): X matrix:\n{X if 'X' in locals() else 'X not computed in this path'}")
        # Example for other debug prints if they were to be activated:
        # if 'X_alt' in locals():
        #     print(f"DEBUG compute_delta_LMX (Verbosity {verbosity}, Monomer {monomer}): X_alt matrix:\n{X_alt}")


    ## --- term2 alternative computation ---
    ## Rebuild the Fock operator for comparison for term 2:
    #J = np.einsum('pqrs,rs->pq', I, Den)
    #K = np.einsum('prqs,rs->pq', I, Den)
    ## 2*tr(H_aa):
    #trH = 2*np.einsum('br,rb->', H, Den)
    ## 2*tr(J_aa)
    #trJ = 2*np.einsum('pq,pq->', J, Den)
    ## - tr(K_aa)
    #trK = -np.einsum('pq,pq->', K, Den)
    ## -- print the individual traces of H, J and K --

    #term2_alt = trH + trJ + trK

    ## --- Alternative computation 1 ---  
    #HD1 = 2*np.einsum('ar,ra->', H_MO1, D_1)
    #HD2= 2*np.einsum('ar,ra->', H_MO2, D_2)  
    #JD11 = 2*np.einsum('arbs,ra,sb->', I_MO1, D_1, D_1)
    #JD12 = 2*np.einsum('arbs,ra,sb->', I_MO2, D_1, D_2)
    #JD21 = 2*np.einsum('arbs,ra,sb->', I_MO3, D_2, D_1)
    #JD22 = 2*np.einsum('arbs,ra,sb->', I_MO4, D_2, D_2)
    #KD11 = - np.einsum('asbr,ra,sb->', I_MO1, D_1, D_1)
    #KD12 = - np.einsum('asbr,ra,sb->', I_MO2, D_1, D_2)
    #KD21 = - np.einsum('asbr,ra,sb->', I_MO3, D_2, D_1)
    #KD22 = - np.einsum('asbr,ra,sb->', I_MO4, D_2, D_2)
    #term1_alt = HD1 + HD2 + JD11 + JD12 + JD21 + JD22 + KD11 + KD12 + KD21 + KD22 

    #delta_LX_alt = term1_alt - term2
    ## --- Alternative computation 2 ---
    ## Compute term11, term12 and term13 with the operators in atomic base 
    ## and the modified density matrices components
    #term11_alt2 =  2*np.einsum('pq,qp->', H, X1)
    #term11_alt2 += 2*np.einsum('pq,qp->', H, X2)
    #term12_alt2 =  2*np.einsum('pqrs,qp,sr->', I, X1, X1)
    #term12_alt2 += 2*np.einsum('pqrs,qp,sr->', I, X1, X2)
    #term12_alt2 += 2*np.einsum('pqrs,qp,sr->', I, X2, X1)
    #term12_alt2 += 2*np.einsum('pqrs,qp,sr->', I, X2, X2)
    #term13_alt2 =  - np.einsum('psrq,qp,sr ->', I, X1, X1) # written with the convential chemist index ordering for exchange.
    #term13_alt2 -=   np.einsum('psrq,qp,sr ->', I, X1, X2) 
    #term13_alt2 -=   np.einsum('psrq,qp,sr ->', I, X2, X1) 
    #term13_alt2 -=   np.einsum('psrq,qp,sr ->', I, X2, X2) 
    #term1_alt2  = term11_alt2 + term12_alt2 + term13_alt2

    ## --- print terms and alternatives ---
    #print('D block for A:\n', D_block)
    #print(f'1st term for {monomer} in compute Delta LX:', term1)
    #print(f'alternative 1st term for monomer={monomer}', term1_alt)
    #print(f'alternative 2 1st term for {monomer} in compute Delta LX:', term1_alt2)
    #print(f'\nWhat is the ratio of all of this terms?\n')
    #ratio_norm_alt = term1/term1_alt
    #ratio_norm_alt2 = term1/term1_alt2
    #ratio_alt_alt2 = term1_alt/term1_alt_2
    #print(f'Ratio normal vs alternative for {monomer}:', ratio_norm_alt)
    #print(f'Ratio normal vs alternative 2 for {monomer}:', ratio_norm_alt2)
    #print(f'Ratio alternative vs alternative 2 for {monomer}:', ratio_alt_alt2)

    #print('computation of term 2: Take the fock operator and trace it with the density matrix*2')
    #print(f'2nd term for {monomer} in compute Delta LX:', term2)
    #print('alternative computation 1: transform matrix element in MO, trace them*2 with density matrix and sum them.')
    #print(f'2nd term aternative 1 for {monomer}:', term2_alt)
    #J_MO = np.einsum('pqrs,rs->pq', I, Den)
    #K_MO = np.einsum('prqs,rs->pq', I, Den)
    #FMO = H + J_MO - 0.5*K_MO
    #term2_old = 2*np.einsum('pq,pq->',FMO, Den)
    #print('alternative computation 2: transform matrix elements in MO, sum them and the trace them with the density matrix.')
    #print(f'2nd term alternative 2 for {monomer}', term2_old)

    # --- print the Delta terms ---
    #print(f'delta L{monomer}:\n', delta_LX)
    #print(f'delta L{monomer} alt:\n', delta_LX_alt)
    #delta_LX_ratio = delta_LX/delta_LX_alt
    #print(f'Ratio delta L{monomer} alt vs normal =', delta_LX_ratio)
    return delta_LMX



def compute_D_matrix(CA, CB, h_sapt, S):
    """
    Computes the inverse of the overlap matrix of all occupied molecular orbitals from two monomers.

    Parameters:
    - CA (numpy.ndarray): Coefficient matrix of occupied MOs for monomer A (shape: n_basis x nmo_A).
    - CB (numpy.ndarray): Coefficient matrix of occupied MOs for monomer B (shape: n_basis x nmo_B).
    - S (numpy.ndarray): Overlap matrix of the atomic orbital basis functions (shape: n_basis x n_basis).

    Returns:
    - D (numpy.ndarray): Inverse of the overlap matrix of all occupied MOs (shape: n_occ_total x n_occ_total).
    """
    # Get number of MOs and occupied orbitals from h_sapt
    nocc_A = h_sapt.ndocc_A  # Number of occupied MOs in monomer A
    nocc_B = h_sapt.ndocc_B  # Number of occupied MOs in monomer B
    # Extract occupied MO coefficients
    CA_occ = CA[:, :nocc_A]  # Shape: n_basis x nocc_A
    CB_occ = CB[:, :nocc_B]  # Shape: n_basis x nocc_B

    # Compute the overlap matrices within monomers (should be close to identity)
    S_AA = np.dot(CA_occ.T, np.dot(S, CA_occ))  # Overlap of monomer A MOs
    S_BB = np.dot(CB_occ.T, np.dot(S, CB_occ))  # Overlap of monomer B MOs

    # Compute the off-diagonal overlap matrix between monomers
    S_AB = np.dot(CA_occ.T, np.dot(S, CB_occ))  # Overlap between monomer A and B MOs
    S_BA = S_AB.T
    
    S_MO = np.block([
        [S_AA, S_AB],
        [S_BA, S_BB]
    ])

    # Invert the overlap matrix
    try:
        D = np.linalg.inv(S_MO)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("The MO overlap matrix is singular and cannot be inverted.")
    
    # !! This prints needs to be activate for verbosity level >=2
    # Print the assembled overlap matrix
    #print("\nAssembled overlap matrix (S_MO):\n", S_MO)
    # Print the D matrix
    #print("\nInverse of the overlap matrix (D):\n", D)
    # Print the overlap matrices
    #print("\nOverlap matrix (S):\n", S)
    #print("\nOverlap matrix for Monomer A (S_AA):\n", S_AA)
    #print("\nOverlap matrix for Monomer B (S_BB):\n", S_BB)
    #print("\nOverlap matrix between Monomers A and B (S_AB):\n", S_AB)
    # Assemble the full overlap matrix

    return D

def build_F_int_lr(h_sapt, monomer='A', input_dict=fqsart_inputs_default):
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
    
    Returns:
      F_lr (np.ndarray): The long-range interaction Fock matrix.
    """
    # Retrieve the two-electron integrals in AO basis; adjust axes if needed.
    I = h_sapt.I.swapaxes(1,2)
    print(f'Building lr F_int for {monomer}')

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
    
    # !! this prints also need to be activated for verbosity level >=2. 
    # This should be true every time we are printing matrices that are components of a total potential
    # in this case we are also printing components of a debugging function, so we could add a verbosity level =3
    # to print matrix elements of debugging functions, while having a a verbosity level <=2 just prints the total potential
    # (if we are using a debugging function the user must also print the potential and the compnents, doesn't make any sense
    # to use a debugging function with a verbosity level of 0).
    print(f'{monomer} lr U:\n', U)
    print(f'{monomer} lr J:\n', term1)
    print(f'{monomer} lr K=0:\n', term2)
    F_lr = U + term1 - term2
    F_lr_noK = U + term1
    print(f'LR F{monomer}_int without K for :\n', F_lr_noK)
    #print(f"build_F_int_lr for monomer {monomer}:\n", F_lr)
    return F_lr

def build_elst_exch_pot_lr(monomer, CA_occ, CB_occ, FA_int, FB_int, S, input_dict=fqsart_inputs_default):
    """
    Computes the long-range interaction potential for a given monomer using the condensed form.
    
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
      monomer (str): 'A' or 'B'
      CA_occ (np.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (np.ndarray): Occupied MO coefficients for monomer B.
      FA_int (np.ndarray): Perturbed Fock operator for monomer A.
      FB_int (np.ndarray): Perturbed Fock operator for monomer B.
      S (np.ndarray): AO overlap matrix.
      
    Returns:
      V_lr (np.ndarray): The long-range interaction potential matrix.
    """
    import numpy as np
    
    if monomer.upper() == 'A':
        # Build projector from monomer B occupied orbitals:
        YYS = np.einsum('pa,qa,qk->pk', CB_occ, CB_occ, S, optimize=True)
        FX_int = FA_int
        FY_int = FB_int
    elif monomer.upper() == 'B':
        YYS = np.einsum('pa,qa,qk->pk', CA_occ, CA_occ, S, optimize=True)
        FX_int = FB_int
        FY_int = FA_int
    else:
        raise ValueError("monomer must be 'A' or 'B'")
    
    # --- Print matrix elements for debugging purposes ---
    term1 = FY_int.copy()
    term2 = -np.einsum('pq,qi->pi', FY_int, YYS)
    term3 = -np.einsum('qp,qi->pi', YYS, FX_int)
    term4 = np.einsum('qp,qi,ij->pj', YYS, FX_int, YYS)
    
    # Verbosity check
    verbosity = input_dict.get("debug_verbosity_level", 0) if input_dict else 0
    print_matrix_elements = input_dict.get("Print matrix elements", False) if input_dict else False
    
    if verbosity >= 3 or print_matrix_elements:
        print('LR WA_elst_exch term1 = FB_int :\n', term1)
        print('LR WA_elst_exch term3 = - FB_int*BBS :\n', term2)
        print('LR WA_elst_exch term7 = - SBB*FA_int :\n', term3)
        print('LR WA_elst_exch term9 = SBB*FA_int*BBS :\n', term4)
    WX_elst_exch = term1 + term2 + term3 + term4
    V_lr = (WX_elst_exch+WX_elst_exch.T)/2 

    return V_lr

def build_elst_pot(monomer, h_sapt, CA_occ, CB_occ, input_dict):
    """
    Constructs the total electrostatic potential only for a specified monomer.
    
    For monomer A, the routine follows these steps:
      -- Extract the nucleai-electron potential matrix in A.O:
            mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())
            VB  = 2*np.asarray(mintsB.ao_potential())
      -- Extract the two electron integral matrix:
            I = h_sapt.I.swapaxes(1,2)
      -- Form the electrostatic potential for B in MO:
            JB = 4*np.einsum('pqrs, rb, sb -> pq ', I, CB_occ, CB_occ)
      -- Form the total electrostatic potential:
            W_tot = VB + JB
    For monomer B, similar expressions are used.
    
    Parameters:
      monomer (str): 'A' or 'B'
      h_sapt (np.ndarray): Integrals and potentials object in AO
      CA_occ (np.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (np.ndarray): Occupied MO coefficients for monomer B.
      
    Returns:
      W_tot (np.ndarray): The total interaction potential for the specified monomer.
    """        
    I = h_sapt.I.swapaxes(1,2)
    verbosity = input_dict.get("debug_verbosity_level", 0)
    # print_matrix_elements = input_dict.get("Print matrix elements", False) # Retained for potential future granular control

    if monomer.upper() == 'A':
        if verbosity >= 1:
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer A): Building Elst interaction potential for A = VB + 2*JB")
        mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())
        VB  = np.asarray(mintsB.ao_potential())
        JB = np.einsum('pqrs,rb,sb->pq', I, CB_occ, CB_occ)
        W_tot = VB + 2*JB

        if verbosity >= 1: # W_tot is a total potential, so print at verbosity 1
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer A): Total Elst Potential (W_tot for A):\n{W_tot if verbosity >= 1 else ''}")
        if verbosity >= 2: # Components VB and JB at verbosity 2
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer A): VB component:\n{VB}")
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer A): JB component:\n{JB}")

        return W_tot
        
    elif monomer.upper() == 'B':
        if verbosity >= 1:
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer B): Building Elst interaction potential for B = VA + 2*JA")
        mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())
        VA  = np.asarray(mintsA.ao_potential())
        JA = np.einsum('pqrs,rb,sb->pq', I, CA_occ, CA_occ)
        W_tot = VA + 2*JA

        if verbosity >= 1: # W_tot is a total potential, so print at verbosity 1
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer B): Total Elst Potential (W_tot for B):\n{W_tot if verbosity >= 1 else ''}")
        if verbosity >= 2: # Components VA and JA at verbosity 2
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer B): VA component:\n{VA}")
            print(f"DEBUG build_elst_pot (Verbosity {verbosity}, Monomer B): JA component:\n{JA}")

        return W_tot
    else:
        raise ValueError("monomer must be either 'A' or 'B'")

def build_enuc_int_pot(monomer, h_sapt, CA_occ, CB_occ, D, S, VA, VB, input_dict):
    """
    Constructs the total interaction potential for a specified monomer.
    
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

    For monomer B, similar expressions are used with:
            BDAS = np.einsum('pq,qi,ji,jk -> pk', CB_occ, D_BA, CA_occ, S, optimize=True)
            ADAS = np.einsum('pq,qi,ji,jk -> pk', CA_occ, D_AA, CA_occ, S, optimize=True)
    
    Parameters:
      monomer (str): 'A' or 'B'
      CA_occ (np.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (np.ndarray): Occupied MO coefficients for monomer B.
      D (np.ndarray): The full inverse occupied–MO overlap matrix partitioned as [D_AA D_AB; D_BA D_BB].
      S (np.ndarray): AO overlap matrix.
      VA, VB (np.ndarray): one electron-nuc potential matrices used in the elst+exch e-nuc potential.
    Returns:
      WA_elst_exch, WB_elst_exch (np.ndarray): The total interaction e-nuc potential for the specified monomer.
    """
    nocc_A = h_sapt.ndocc_A
    verbosity = input_dict.get("debug_verbosity_level", 0)
    print_matrix_elements = input_dict.get("Print matrix elements", False) # Retained for backward compatibility

    if monomer.upper() == 'A':
        if verbosity >= 1:
            print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): Building E_Nuc Elst+Exch potential for A")
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
        
        WA_elst_exch = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
        # --- compute the exchange potential without LM ---
        WA_exch = WA_elst_exch - term1 + term3 + term7 - term9 

        if verbosity >= 1:
            print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): Total E_Nuc Elst+Exch potential (WA_elst_exch):\n{WA_elst_exch if verbosity >=1 else ''}")
        
        if verbosity >= 2:
            print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): Exchange part (WA_exch):\n{WA_exch}")
        
        if verbosity >= 3 or print_matrix_elements:
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): ADBS :\n', ADBS)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): BDBS :\n', BDBS)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term1 (VB) :\n', term1)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term2 (-VB*ADBS, expected 0):\n', term2)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term3 (-VB*BDBS) :\n', term3)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term4 (-ADBS*VB, expected 0) :\n', term4)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term5 (ADBS*VB*ADBS, expected 0) :\n', term5)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term6 (ADBS*VB*BDBS, expected 0) :\n', term6)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term7 (-BDBS*VA) :\n', term7)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term8 (BDBS*VA*ADBS, expected 0) :\n', term8)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer A): term9 (BDBS*VA*BDBS) :\n', term9)

        return WA_elst_exch

    if monomer.upper() == 'B':
        if verbosity >= 1:
            print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): Building E_Nuc Elst+Exch potential for B")
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

        WB_elst_exch = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
        # --- compute the exchange potential without LM ---
        WB_exch = WB_elst_exch - term1 + term3 + term7 - term9 # Original formula for exchange part

        if verbosity >= 1:
            print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): Total E_Nuc Elst+Exch potential (WB_elst_exch):\n{WB_elst_exch if verbosity >= 1 else ''}")

        if verbosity >= 2:
            print(f"DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): Exchange part (WB_exch):\n{WB_exch}")
            
        if verbosity >= 3 or print_matrix_elements:
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): BDAS :\n', BDAS)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): ADAS :\n', ADAS)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term1 (VA) :\n', term1)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term2 (-VA*BDAS, expected 0):\n', term2)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term3 (-VA*ADAS) :\n', term3)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term4 (-BDAS*VA, expected 0) :\n', term4)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term5 (BDAS*VA*BDAS, expected 0) :\n', term5)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term6 (BDAS*VA*ADAS, expected 0) :\n', term6)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term7 (-ADAS*VB) :\n', term7)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term8 (ADAS*VB*BDAS, expected 0) :\n', term8)
            print(f'DEBUG build_enuc_int_pot (Verbosity {verbosity}, Monomer B): term9 (ADAS*VB*ADAS) :\n', term9)
        
        return WB_elst_exch
    
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

    if verbosity >= 2:
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
        JX_contrib = 2 * oe.contract('sb,mnbs->mn', D_XY, eri_munu_bs)
        JX += JX_contrib
        if verbosity >= 3:
            print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): JX contribution from D_block {D_XY.shape} and CY_occ {CY_occ.shape}:\n{JX_contrib}")

    FX_int_AO += JX
    if verbosity >= 2:
        print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): Total JX:\n{JX}")
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
        KX += KX_contrib
        if verbosity >= 3:
            print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): KX contribution from D_block {D_XY.shape} and CY_occ {CY_occ.shape}:\n{KX_contrib}")

    FX_int_AO -= KX # Subtract exchange
    if verbosity >= 2:
        print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): Total KX:\n{KX}")
    if verbosity >=1:
        print(f"DEBUG build_F_int (Verbosity {verbosity}, Monomer {monomer}): Final FX_int_AO:\n{FX_int_AO}")

    return FX_int_AO

def build_elst_exch_pot(monomer, h_sapt, CA_occ, CB_occ, D, S, FA_int, FB_int, input_dict):
    """
    Constructs the total interaction potential for a specified monomer.
    
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
      CA_occ (np.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (np.ndarray): Occupied MO coefficients for monomer B.
      D (np.ndarray): The full inverse occupied–MO overlap matrix partitioned as [D_AA D_AB; D_BA D_BB].
      S (np.ndarray): AO overlap matrix.
      FA_int, FB_int (np.ndarray): The perturbed (interaction) Fock matrices used in the elst+exch potential.
      input_dict (dict): Dictionary controlling debug print options.
      
    Returns:
      WX_elst_exch (np.ndarray): The electrostatic + exchange interaction potential for the specified monomer.
    """        
    verbosity = input_dict.get("debug_verbosity_level", 0)
    full_sym = input_dict.get("Full symmetrization", False)
    half_sym = input_dict.get("Half symmetrization", False)
    print_matrix_elements = input_dict.get("Print matrix elements", False)

    nocc_A = h_sapt.ndocc_A

    if verbosity >= 1:
        print(f"--Building {monomer} Elst+Exch potential--")
        if full_sym:
            print(f"- Full symmetrization scheme -")
        elif half_sym:
            print(f"- Half symmetrization scheme -")
        else:
            print(f"- No symmetrization scheme -")

    if monomer.upper() == "A":
        D_AB = D[:nocc_A, nocc_A:]
        D_BB = D[nocc_A:, nocc_A:]
        XDYS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AB, CB_occ, S, optimize=True) # ADBS
        YDYS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BB, CB_occ, S, optimize=True) # BDBS
        FX_int = FA_int # FX_int for monomer A is FA_int
        FY_int = FB_int # FY_int for monomer A is FB_int
    elif monomer.upper() == "B":
        D_AA = D[:nocc_A, :nocc_A]
        D_BA = D[nocc_A:, :nocc_A]
        XDYS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BA, CA_occ, S, optimize=True) # BDAS
        YDYS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AA, CA_occ, S, optimize=True) # ADAS
        FX_int = FB_int # FX_int for monomer B is FB_int
        FY_int = FA_int # FY_int for monomer B is FA_int
    else:
        raise ValueError("monomer must be either \"A\" or \"B\"")
    
    if full_sym:
        print('- Full symmetrization scheme -')
        FY_int = (FY_int + FY_int.T)/2
        FX_int = (FX_int + FX_int.T)/2
        XDYS = (XDYS+XDYS.T)/2
        YDYS = (YDYS+YDYS.T)/2
    elif half_sym:
        print('- Half symmetrization scheme -')
        FY_int = (FY_int + FY_int.T)/2
        FX_int = (FX_int + FX_int.T)/2

    term1 = FY_int.copy()
    term2 = -np.einsum("pq,qi -> pi", FY_int, XDYS)
    term3 = -np.einsum("pq,qi -> pi", FY_int, YDYS)
    term4 = -np.einsum("qp,qi -> pi", XDYS, FY_int) 
    term5 = np.einsum("qp,qi,ij -> pj", XDYS, FY_int, XDYS) 
    term6 = np.einsum("qp,qi,ij -> pj", XDYS, FY_int, YDYS) 
    term7 = -np.einsum("qp,qi -> pi", YDYS, FX_int) 
    term8 = np.einsum("qp,qi,ij -> pj", YDYS, FX_int, XDYS) 
    term9 = np.einsum("qp,qi,ij -> pj", YDYS, FX_int, YDYS) 
    
    # --- Symmetrization of terms ---
    if full_sym or half_sym: # Symmetrize terms if any symmetrization is active
        term1 = (term1 + term1.T) / 2
        term2 = (term2 + term2.T) / 2
        term3 = (term3 + term3.T) / 2
        term4 = (term4 + term4.T) / 2
        term5 = (term5 + term5.T) / 2
        term6 = (term6 + term6.T) / 2
        term7 = (term7 + term7.T) / 2
        term8 = (term8 + term8.T) / 2
        term9 = (term9 + term9.T) / 2
        if verbosity >=2 and print_matrix_elements:
             print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): Terms have been symmetrized.")

    WX_elst_exch = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

    # Stampe dettagliate attivate con verbosity >= 3 o Print matrix elements
    if verbosity >= 3 or print_matrix_elements:
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): XDYS (mapping op):\n{XDYS}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): YDYS (mapping op):\n{YDYS}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term1 (current_FY_int):\n{term1}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term2 (-current_FY_int*XDYS):\n{term2}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term3 (-current_FY_int*YDYS):\n{term3}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term4 (-XDYS*current_FY_int):\n{term4}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term5 (XDYS*current_FY_int*XDYS):\n{term5}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term6 (XDYS*current_FY_int*YDYS):\n{term6}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term7 (-YDYS*current_FX_int):\n{term7}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term8 (YDYS*current_FX_int*XDYS):\n{term8}")
        print(f"DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): term9 (YDYS*current_FX_int*YDYS):\n{term9}")
    if verbosity >= 1:
        print(f'DEBUG build_elst_exch_pot (Verbosity {verbosity}, Monomer {monomer}): Total Elst+Exch potential (WX_elst_exch):\n{WX_elst_exch}')

    ## -- Print the total Elst + Exch pot --
    #print(f'fq standard total W{monomer}_elst_exch : \n', WX_elst_exch)
    ## --- compute the exchange potential without LM ---
    #WX_elst = build_elst_pot(monomer, h_sapt, CA_occ, CB_occ)
    #WX_exch = WX_elst_exch - WX_elst
    ## -- Print the exchange pot --
    #print(f'fq {monomer} Exchange potential only:\n', WX_exch)

    ## --- ALternative, but faster, computation ---
    ## -- !! needs to be updated !! --
    # Term1: (FB_total), No need to change it.
    # Term2: - (FB_total) * (B * D_BA * A.T * S) = - FB_total * BDAS
    #LHS_B_Term2 = -np.einsum('pq, qi -> pi', FB_total, BDAS)
    ## Term3: - (FB_total) * ADAS
    #LHS_B_Term3 = -np.einsum('pq, qi -> pi', FB_total, ADAS)
    ## Term4: - (BDAS)^T * FB_total
    #LHS_B_Term4 = -np.einsum('qp, qi-> pi', BDAS, FB_total)
    ## Term5: + (BDAS)^T * FB_total * BDAS
    #LHS_B_Term5 = np.einsum('qp, qi, ij -> pj', BDAS, FB_total, BDAS)
    ## Term6: + (BDAS)^T * FB_total * ADAS
    #LHS_B_Term6 = np.einsum('qp, qi, ij -> pj', BDAS, FB_total, ADAS)
    ## Term7: - (ADAS)^T * FB_int
    #LHS_B_Term7 = np.einsum('qp, qi -> pi', ADAS, FB_int)
    ## Term8: + (ADAS)^T * FB_int * BDAS
    #LHS_B_Term8 = np.einsum('qp, qi, ij -> pj', ADAS, FB_int, BDAS)
    ## Term9: + (ADAS)^T * FB_int * ADAS
    #LHS_B_Term9 = np.einsum('qp, qi, ij -> pj', ADAS, FB_int, ADAS)
    ## Sum all terms to get LHS_A
    #LHS_B = FB_total + LHS_B_Term2 + LHS_B_Term3 + LHS_B_Term4 + LHS_B_Term5 + LHS_B_Term6 + LHS_B_Term7 + LHS_B_Term8 + LHS_B_Term9
    # Ensure matrices are symmetric
    #LHS_B_sym = (LHS_B + LHS_B.T.conj()) / 2
    
    #print(f"int_pot for monomer {monomer}:\n", WX_elst_exch)
    return WX_elst_exch


def build_LM_pot(monomer, h_sapt, CA_occ, CB_occ, D, S, FA_int, TA, FB_int, TB, input_dict):
    """
    Constructs the total interaction potential for a specified monomer.
    
    For monomer A, the routine follows these steps:
      -- Extract D matrix blocks:
            D_AA = D[:nocc_A, :nocc_A]
            D_AB = D[:nocc_A, nocc_A:]
            D_BB = D[nocc_A:, nocc_A:]
      -- Compute mapping operators:
            XDYS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AB, CB_occ, S, optimize=True)
            YDYS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BB, CB_occ, S, optimize=True)
      -- Compute the sum of F int and T
            FXTX = FX_int + TX
      -- Form the Delta potential:
            LX_pot = FXTX\
              - np.einsum("pq, qi -> pi", FXTX, XDYS)\
              - np.einsum("pq, qi -> pi", FXTX, YDYS)\
              - np.einsum("qp, qi -> pi", XDYS, FXTX)\
              + np.einsum("qp, qi, ij -> pj", XDYS, FXTX, XDYS)\
              + np.einsum("qp, qi, ij -> pj", XDYS, FXTX, YDYS)
    
    For monomer B, similar expressions are used with:
            XDYS = np.einsum("pq,qi,ji,jk -> pk", CB_occ, D_BA, CA_occ, S, optimize=True)
            YDYS = np.einsum("pq,qi,ji,jk -> pk", CA_occ, D_AA, CA_occ, S, optimize=True)
    
    Parameters:
      monomer (str): "A" or "B"
      CA_occ (np.ndarray): Occupied MO coefficients for monomer A.
      CB_occ (np.ndarray): Occupied MO coefficients for monomer B.
      D (np.ndarray): The full inverse occupied–MO overlap matrix partitioned as [D_AA D_AB; D_BA D_BB].
      S (np.ndarray): AO overlap matrix.
      FX_int = FA_int, FB_int (np.ndarray): The perturbed (interaction) Fock matrices used in the elst+exch potential of the monomer of interest.
      TX = TA, TB (np.ndarray): The kinetik energy operator of the monomer of interest.
      FY_int = FA_int, FB_int (np.ndarray): The perturbed (interaction) Fock matrices used in the elst+exch potential of th partner.
      TY = TA, TB (np.ndarray): The kinetik energy operator of the partner.
      input_dict (dict): Dictionary controlling debug print options.
      
    Returns:
      LMX_pot (np.ndarray): The total delta LM potential for the specified monomer.
    """
    verbosity = input_dict.get("debug_verbosity_level", 0)
    full_sym = input_dict.get("Full symmetrization", False)
    half_sym = input_dict.get("Half symmetrization", False)
    print_matrix_elements = input_dict.get("Print matrix elements", False)

    if verbosity >= 1:
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): --Building {monomer} LM potential--")
        if full_sym:
            print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): - Full symmetrization scheme -")
        elif half_sym:
            print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): - Half symmetrization scheme -")
        else:
            print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): - No symmetrization scheme -")

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
        if verbosity >=2 and print_matrix_elements:
            print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): FXTX, FYTY, XDYS, YDYS symmetrized for full_sym.")
    elif half_sym:
        FXTX = (FXTX + FXTX.T)/2
        FYTY = (FYTY + FYTY.T)/2
        if verbosity >=2 and print_matrix_elements:
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

    if verbosity >= 1:
        print(f"DEBUG build_LM_pot (Verbosity {verbosity}, Monomer {monomer}): Total LM potential (LMX_pot):\n{LMX_pot if verbosity >= 1 else 'Matrix not printed, increase verbosity level'}")

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

def do_sart_fq(dimer: psi4.geometry, 
                sapt:helper_SAPT,
                input_dict: dict = fqsart_inputs_default,
                # plot_iter:  bool = False, # Controlled by input_dict["debug_verbosity_level"] or a new specific flag if needed for data collection
                # separation: float = None, # Should be part of input_dict or system_info
                # system_name=\"unknown" # Should be part of input_dict or system_info
                ):
    """
    Perform the SART-FQ iterative optimization of monomer orbitals.

    Parameters:
      dimer (psi4.geometry): The dimer geometry.
      sapt (helper_SAPT): Helper object containing integrals, wavefunctions, initial energies, etc.
      input_dict (dict): Dictionary controlling calculation parameters and debug print options.
                         Expected keys include: max_iter, tol, debug_verbosity_level, 
                         Print matrix elements, job_name, job_method, job_basis, etc.
      
    Returns:
      sart_results (dict): A dictionary containing all relevant results from the SART-FQ calculation.
                           Keys include: "CA_final", "CB_final", "E_int_final", "E_elst_final", "E_exch_final",
                           "E_delta_LMA_final", "E_delta_LMB_final", "iterations", "converged", 
                           "E_elst_1", "E_exch_1", "energy_iterations_data", "job_identifier", etc.
    """
    
    # Extract parameters from input_dict
    max_iter = input_dict.get("max_iter", 10)
    tol = input_dict.get("tol", 1.0E-8)
    verbosity = input_dict.get("debug_verbosity_level", 0)
    print_matrix_elements = input_dict.get("Print matrix elements", False)
    plot_iter_data_collection = input_dict.get("collect_iter_data_for_plot", True) # New flag to control data collection for plotting
    system_name = input_dict.get("system_name", "unknown_system")
    job_name = input_dict.get("job_name", "SART-FQ calculation")
    job_method = input_dict.get("job_method", "SART-FQ")
    job_basis = "Implicitly defined by sapt object" # Or extract if available
    # Create a unique job identifier, e.g., using a timestamp or a UUID
    import uuid
    job_identifier = input_dict.get("job_identifier", str(uuid.uuid4()))

    sart_results = {
        "job_name": job_name,
        "job_method": job_method,
        "job_basis": job_basis,
        "job_identifier": job_identifier,
        "system_name": system_name,
        "E_elst_1": None,
        "E_exch_1": None,
        "energy_iterations_data": [],
        "iterations_details": [] # To store detailed energy components per iteration
    }

    lr             = input_dict.get("long range", False)
    SQ_elst_exch   = input_dict.get("SQ elst exch", False)
    SQ_LM          = input_dict.get("SQ LM", False)
    FQ_elst_exch   = input_dict.get("FQ elst exch", True) # Defaulting to True for FQ calculations
    FQ_LM          = input_dict.get("FQ LM", True)       # Defaulting to True for FQ calculations

    monomerA = dimer.extract_subsets(1,2)
    monomerB = dimer.extract_subsets(2,1)
    
    h_sapt = sapt
    rhfA = h_sapt.rhfA
    rhfB = h_sapt.rhfB

    if verbosity >= 0:
        print(f"SART-FQ Calculation for: {system_name}")
        print(f"Job Identifier: {job_identifier}")
        print(f"Max iterations: {max_iter}, Tolerance: {tol}")
        print("Monomer Energies for A and B before iterations:")
        print(f"RHF A =", rhfA,"\nRHF B =", rhfB)

    # Intializing Energies to monomer energies
    EA = rhfA
    EB = rhfB

    # --- Initialize quantities for first iteration: ---
    # Initialize CA_new and CB_new with the initial MO coefficients
    CA_new = h_sapt.C_A.copy() # Ensure we work with copies
    CB_new = h_sapt.C_B.copy()
    nocc_A = h_sapt.ndocc_A
    nocc_B = h_sapt.ndocc_B

    nmo_A = h_sapt.nmo # wfnA.nmo()
    nmo_B = h_sapt.nmo # wfnB.nmo()

    if verbosity >= 1:
        print("Total number of MOs (in DCBS):", nmo_A)
        print("Number of occupied MO(A):", nocc_A)
        print("Number of occupied MO(B):", nocc_B)

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

    # Use CA_new and CB_new in the computation
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

    # --- Initialize quantities for first iteration: ---
    CA_occ = CA_new[:, :nocc_A]
    CB_occ = CB_new[:, :nocc_B]

    D = compute_D_matrix(CA_occ, CB_occ, h_sapt, S)

    converged = False
    iteration = 0
    E_int_prev = 0.0 # Initialize to 0.0 or a suitable large number

    if verbosity >= 1:
        print(f"\nStarting SART-FQ iterations...")

    while not converged and iteration < max_iter:
        iteration += 1
        if verbosity >= 1:
            print(f"\n###### Iteration {iteration} #######\n")
        
        # Compute interaction operators
        # Pass I_AO (mu nu | p q) to build_F_int, it will handle contractions
        FA_int = build_F_int(CA_new, CB_new, h_sapt, D, monomer="A", input_dict=input_dict)
        FB_int = build_F_int(CA_new, CB_new, h_sapt, D, monomer="B", input_dict=input_dict)

        if lr and verbosity >=1: # Long range checks are for debug 
            print("\n--- DEBUG: Computing the long range of the potentials ---")
            FA_int_lr = build_F_int_lr(h_sapt, monomer="A", input_dict=input_dict)
            FB_int_lr = build_F_int_lr(h_sapt, monomer="B", input_dict=input_dict)
            if verbosity >=2 or print_matrix_elements:
                print("DEBUG Standard FA_int =\n", (FA_int+FA_int.T)/2)
                print("DEBUG Long-range FA_int_lr =\n", FA_int_lr)
                print("DEBUG Standard FB_int =\n", (FB_int+FB_int.T)/2)
                print("DEBUG Long-range FB_int_lr =\n", FB_int_lr)
            V_lr_A = build_elst_exch_pot_lr("A", CA_occ, CB_occ, FA_int_lr, FB_int_lr, S, input_dict=input_dict)
            if verbosity >=2 or print_matrix_elements:
                print(f"DEBUG LR Elst + Exch potential for monomer A :\n{V_lr_A}")
            V_lr_B = build_elst_exch_pot_lr("B", CA_occ, CB_occ, FA_int_lr, FB_int_lr, S, input_dict=input_dict)
            if verbosity >=2 or print_matrix_elements:
                print(f"DEBUG LR Elst + Exch potential for monomer B :\n{V_lr_B}")
        
        # Compute Elst + Exch potentials
        if verbosity >= 1:
            print("\n--- Computing Elst+Exch potentials (FQ or SQ based on flags) ---")
        
        WA_elst_exch_fq = build_elst_exch_pot("A", h_sapt, CA_occ, CB_occ, D, S, FA_int, FB_int, input_dict)
        WB_elst_exch_fq = build_elst_exch_pot("B", h_sapt, CA_occ, CB_occ, D, S, FA_int, FB_int, input_dict)

        # For comparison or if SQ is chosen
        WA_elst_exch_sq, WB_elst_exch_sq = None, None
        if SQ_elst_exch:
            if verbosity >= 1: print("DEBUG: Computing SQ Elst+Exch for comparison")
            # Density matrices for isolated monomers for J, K terms in SQ
            JA = np.einsum("pqrs,rs->pq", I, D_A)
            # KA = np.einsum("prqs,rs->pq", I_AO, D_A_iso) # Not needed for Elst
            JB = np.einsum("pqrs,rs->pq", I, D_B)
            # KB = np.einsum("prqs,rs->pq", I_AO, D_B_iso) # Not needed for Elst

            WA_elst_AO_sq = VB + JB # Potential on A due to B (elst part)
            WB_elst_AO_sq = VA + JA # Potential on B due to A (elst part)
            
            # Get SQ exchange from omega_exch_utils (ensure it uses current CAs, CBs if iterative)
            # For now, assume it uses initial CAs, CBs as per typical SAPT(0) context
            # If it needs current CAs, CBs, these need to be passed to form_omega_exchange_w_sym
            WB_exch_MO_sq, WA_exch_MO_sq = form_omega_exchange_w_sym(sapt=h_sapt, ca=CA_new, cb=CB_new, oo_vv="S4", ov_vo="Sinf")
            WA_exch_AO_sq = S.dot(CA_new).dot(WA_exch_MO_sq).dot(CA_new.T).dot(S)
            WB_exch_AO_sq = S.dot(CB_new).dot(WB_exch_MO_sq).dot(CB_new.T).dot(S)
            WA_elst_exch_sq = WA_elst_AO_sq + WA_exch_AO_sq
            WB_elst_exch_sq = WB_elst_AO_sq + WB_exch_AO_sq
            if verbosity >=2 or print_matrix_elements:
                print(f"DEBUG Iteration {iteration} SQ Elst + Exchange potential for monomer A :\n{WA_elst_exch_sq}")
                print(f"DEBUG Iteration {iteration} SQ Elst + Exchange potential for monomer B :\n{WB_elst_exch_sq}")

        # Decide which elst_exch potential to use for the SCF update
        WA_elst_exch = WA_elst_exch_fq if FQ_elst_exch else WA_elst_exch_sq
        WB_elst_exch = WB_elst_exch_fq if FQ_elst_exch else WB_elst_exch_sq
        if WA_elst_exch is None or WB_elst_exch is None:
            raise ValueError("Elst+Exch potential not computed. Check FQ_elst_exch/SQ_elst_exch flags.")

        # Compute LM potentials
        if verbosity >= 1:
            print("\n--- Computing LM potentials (FQ or SQ based on flags) ---")
        LMA_pot_fq = build_LM_pot("A", h_sapt, CA_occ, CB_occ, D, S, FA_int, TA, FB_int, TB, input_dict) - FA 
        LMB_pot_fq = build_LM_pot("B", h_sapt, CA_occ, CB_occ, D, S, FA_int, TA, FB_int, TB, input_dict) - FB 

        LMA_pot_sq, LMB_pot_sq = None, None
        if SQ_LM:
            if verbosity >= 1: print("DEBUG: Computing SQ LM for comparison")
            LMB_pot_MO_sq, LMA_pot_MO_sq = form_lm_terms_w_sym(sapt=h_sapt, ca=CA_new, cb=CB_new, s_option="S4")
            LMA_pot_sq = S.dot(CA_new).dot(LMA_pot_MO_sq).dot(CA_new.T).dot(S)
            LMB_pot_sq = S.dot(CB_new).dot(LMB_pot_MO_sq).dot(CB_new.T).dot(S)
            if verbosity >=2 and print_matrix_elements:
                print(f"DEBUG Iteration {iteration} SQ LM potential for A :\n{LMA_pot_sq}")
                print(f"DEBUG Iteration {iteration} SQ LM potential for B :\n{LMB_pot_sq}")

        # Decide which LM potential to use
        LMA_pot = LMA_pot_fq if FQ_LM else LMA_pot_sq
        LMB_pot = LMB_pot_fq if FQ_LM else LMB_pot_sq
        if LMA_pot is None or LMB_pot is None:
            raise ValueError("LM potential not computed. Check FQ_LM/SQ_LM flags.")

        # Total interaction potential (Omega)
        WA_tot = WA_elst_exch + LMA_pot
        WB_tot = WB_elst_exch + LMB_pot
        
        # Symmetrize total potential
        WA_tot_sym = (WA_tot + WA_tot.T)/2
        WB_tot_sym = (WB_tot + WB_tot.T)/2

        if verbosity >= 1:
            print(f"--- Iteration {iteration}: SCF Update ---")
        # SCF for Monomer A
        if verbosity >= 1: psi4.core.print_out(f"\nSCF Iterations for Monomer A (SART Iteration {iteration})")

        try:      
            wfnA_scf = do_scf_itr(monomer= monomerA,
                                        reference= 'RHF',
                                        guess= (CA_occ, CA_occ),
                                        omega= WA_tot_sym, #V_lr_A,
                                        maxiter=300,
                                        input_dict= input_dict
                                        )
            #wfnA_scf = do_scf_itr(monomer=monomerA, reference="RHF", basis_S=S, guess_Ca=CA_new, guess_Cb=CA_new, omega=WA_tot_sym, input_dict=input_dict)
            CA_new = wfnA_scf.Ca().to_array()
            EA_new = wfnA_scf.compute_energy()
        except Exception as e:
            if verbosity >=0: print(f"Error during SCF for Monomer A in iteration {iteration}: {e}")
            sart_results["error"] = f"SCF_A_failure_iter_{iteration}: {e}"
            sart_results["converged"] = False
            return sart_results 
        
        # SCF for Monomer B
        if verbosity >= 1: psi4.core.print_out(f"\nSCF Iterations for Monomer B (SART Iteration {iteration})")
        try:
            wfnB_scf = do_scf_itr(monomer= monomerB,
                                        reference= 'RHF',
                                        guess= (CB_occ, CB_occ),
                                        omega= WB_tot_sym,#V_lr_A,
                                        maxiter=300,
                                        input_dict= input_dict
                                        )
           # wfnB_scf = do_scf_itr(monomer=monomerB, reference="RHF", basis_S=S, guess_Ca=CB_new, guess_Cb=CB_new, omega=WB_tot_sym, input_dict=input_dict)
            CB_new = wfnB_scf.Ca().to_array()        
            EB_new = wfnB_scf.compute_energy()
        except Exception as e:
            if verbosity >=0: print(f"Error during SCF for Monomer B in iteration {iteration}: {e}")
            sart_results["error"] = f"SCF_B_failure_iter_{iteration}: {e}"
            sart_results["converged"] = False
            return sart_results

        # Update occupied MOs for the next iteration or final energy calculation
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

        # Recompute D matrix with new orbitals for energy calculation and next iteration
        D = compute_D_matrix(CA_occ, CB_occ, h_sapt, S)
        
        # Compute interaction energy components with current orbitals
        E1, Eelst_fq_iter, Eexch_fq_iter = compute_E1(CA_new, CB_new, dimer, h_sapt, D, input_dict)
        delta_LMA_iter = compute_delta_LMX("A", CA_new, CB_new, h_sapt, D, HA, I, FA, input_dict)
        delta_LMB_iter = compute_delta_LMX("B", CA_new, CB_new, h_sapt, D, HB, I, FB, input_dict)

        # Deformation Energies
        dEA = EA_new - rhfA
        dEB = EB_new - rhfB
        
        E_int_current = dEA + dEB + E1 + delta_LMA_iter + delta_LMB_iter

        # Store E1_elst and E1_exch from the first iteration
        if iteration == 1:
            sart_results["E_elst_1"] = Eelst_fq_iter
            sart_results["E_exch_1"] = Eexch_fq_iter

        if verbosity >= 1:
            print(f"=== Iteration {iteration}===")
            print(f"E_A = {EA_new} Eh \nE_B = {EB_new} Eh")
            print(f"E_elst = {Eelst_fq_iter} Eh\nE_exch = {Eexch_fq_iter} Eh")
            print(f"Delta_LMA = {delta_LMA_iter} Eh\nDelta_LMB = {delta_LMB_iter} Eh")
            print(f"E_int_current = {E_int_current} Eh")

        # Store data for plotting/analysis
        iter_details = {
            "iteration": iteration,
            "EA": EA_new,
            "EB": EB_new,
            "dEA": dEA,
            "dEB": dEB,
            "E_elst": Eelst_fq_iter,
            "E_exch": Eexch_fq_iter,
            "delta_LMA": delta_LMA_iter,
            "delta_LMB": delta_LMB_iter,
            "E_int_total": E_int_current
        }
        sart_results["iterations_details"].append(iter_details)
        if plot_iter_data_collection:
            # This format matches what plot_energy_iterations_direct expects
            sart_results["energy_iterations_data"].append([
                iteration,
                dEA,
                dEB,
                Eelst_fq_iter, 
                Eexch_fq_iter, 
                delta_LMA_iter + delta_LMB_iter, # Combined Delta LM
                E_int_current # Total Interaction Energy
            ])

        # Check for convergence
        if E_int_prev is not None and abs(E_int_current - E_int_prev) < tol:
            converged = True
            if verbosity >= 0: print(f"\nSART-FQ converged in {iteration} iterations.")
        
        E_int_prev = E_int_current
        EA = EA_new
        EB = EB_new

        if not converged and iteration == max_iter and verbosity >= 0:
            print(f"\nSART-FQ did not converge after {max_iter} iterations.")

    # Final values to store
    sart_results["CA_final"] = CA_new
    sart_results["CB_final"] = CB_new
    sart_results["E_A_final"] = EA
    sart_results["E_B_final"] = EB
    sart_results["E_elst_final"] = Eelst_fq_iter # Last computed values
    sart_results["E_exch_final"] = Eexch_fq_iter
    sart_results["E_delta_LMA_final"] = delta_LMA_iter
    sart_results["E_delta_LMB_final"] = delta_LMB_iter
    sart_results["E_int_final"] = E_int_current
    sart_results["iterations"] = iteration
    sart_results["converged"] = converged
    sart_results["dEA"] = dEA # Deformation energy A
    sart_results["dEB"] = dEB # Deformation energy B
    sart_results["E_lm_final"] = delta_LMA_iter + delta_LMB_iter # Total LM energy

    # The plot function is removed from here and will be called by intermediate-level code
    # if plot_iter and energy_iter_data:
    #     plot_energy_iterations_direct(energy_iter_data, separation, system_name)

    if verbosity >= 0:
        print("\n--- Final SART-FQ Results ---")
        print(f"Converged: {sart_results['converged']}")
        print(f"Iterations: {sart_results['iterations']}")
        print(f"Final E_int: {sart_results['E_int_final']} Eh")
        print(f"Final E_A: {sart_results['E_A_final']} Eh")
        print(f"Final E_B: {sart_results['E_B_final']} Eh")
        print(f"Final dE_A: {sart_results['dEA']} Eh")
        print(f"Final dE_B: {sart_results['dEB']} Eh")
        print(f"Final E_elst: {sart_results['E_elst_final']} Eh")
        print(f"Final E_exch: {sart_results['E_exch_final']} Eh")
        print(f"Final E_delta_LMA: {sart_results['E_delta_LMA_final']} Eh")
        print(f"Final E_delta_LMB: {sart_results['E_delta_LMB_final']} Eh")
        print(f"E_elst_1: {sart_results['E_elst_1']} Eh")
        print(f"E_exch_1: {sart_results['E_exch_1']} Eh")

    return sart_results           

def do_scf_itr(monomer:psi4.geometry, 
                reference:str,
                guess:tuple,
                omega: np.ndarray, 
                maxiter:int=None, 
                input_dict:dict=fqsart_inputs_default,
                diis_bool:bool= True):
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
    if verbosity >= 1:
        print('Entering DO SCF...')
        print('........will start scf-iterations')
    psi4.set_options({'MAXITER': maxiter})
    
    # Constructing base wavefunction and then RHF/HF object
    base_wfn = psi4.core.Wavefunction.build(monomer, 
                                        psi4.core.get_global_option('BASIS'))
    if verbosity >=3:
        print('Base WFN constructed...')
    wfn_ref_obj= psi4.driver.scf_wavefunction_factory('SCF', 
                                                ref_wfn=base_wfn,
                                                reference= reference)
    if verbosity >=3:
        print('RHF object constructed...')
    # Access the GUESS and set these
    Ca = guess[0]
    Cb = guess[1]
    Ca_psi4_mat = psi4.core.Matrix.from_array(Ca)
    Cb_psi4_mat = psi4.core.Matrix.from_array(Cb)
    if verbosity >=3:
        print('GUESS are extracted...')
    wfn_ref_obj.guess_Ca(Ca_psi4_mat)
    wfn_ref_obj.guess_Cb(Cb_psi4_mat)
    if verbosity >=3:
        print('GUESS are set...')

    # Initialize for SCF Run
    wfn_ref_obj.initialize()
    if verbosity >=3:
        print('After initializing.....Check if it has the correct GUESS loaded')
        print('CA matrix =', wfn_ref_obj.Ca().to_array())
        print('CB matrix =',wfn_ref_obj.Cb().to_array())

    # Prepare the Omega matrix
    if verbosity >=3:
        print('Omega to be added to Fock')
    Omega_psi4_mat = psi4.core.Matrix.from_array(omega)
    wfn_ref_obj.push_back_external_potential(Omega_psi4_mat)

    # Start the SCF runs and save
    wfn_ref_obj.iterations()
    wfn_ref_obj.save_density_and_energy()

    if verbosity >=3:
        print('After SCF iterations.....Modified')
        #print(wfn_ref_obj.Ca().to_array())
        #e_scfomega = wfn_ref_obj.compute_energy()
        #print('Energy with the modified orbitals:....')
        #print(e_scfomega)

    if verbosity >=1:
        print('.... Finished scf-iterations')

    return wfn_ref_obj

#def plot_energy_iterations_direct(energy_data, separation, scale=1, zoom_y=True, custom_y_limits=None):
#    """
#    Given a list of dictionaries (each for one iteration) with keys like:
#       "iteration", "dEA", "dEB", "Elst+Exch", "Delta LA", "Delta LB", 
#       "Total Interaction Energy", and optionally "Energy change",
#    this function generates line plots for these energy components versus iteration.
#    
#    The energy values defined in Hartree are multiplied by 'scale' (default = 1000) 
#    so the differences are expressed in milli–Hartree units. Then, if zoom_y is True,
#    the function computes suitable y-axis limits (with a small margin) for each subplot,
#    effectively "magnifying" the region where the variations occur.
#    
#    Optionally, a tuple custom_y_limits=(lower, upper) can be provided to force the same y-axis range for all plots.
#    
#    The resulting figures are saved as PNG files.
#    """
#    # Define energy components and assign each a unique color.
#    energy_keys = [
#        "dEA", 
#        "dEB", 
#        "Elst+Exch", 
#        "Delta LA", 
#        "Delta LB", 
#        "Total Interaction Energy"
#    ]
#    colors = ['blue', 'green', 'red', 'magenta', 'cyan', 'black']
#    
#    iterations = [entry["iteration"] for entry in energy_data]
#
#    # Create a figure with 6 subplots.
#    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
#    axs = axs.flatten()
#
#    for idx, key in enumerate(energy_keys):
#        # Get energy values (in Eh) and convert to mEh.
#        y_vals = [entry.get(key, None) for entry in energy_data]
#        scaled_y = [value * scale if value is not None else None for value in y_vals]
#        axs[idx].plot(iterations, scaled_y, marker='o', linestyle='-', color=colors[idx])
#        axs[idx].set_xlabel("Iteration")
#        axs[idx].set_ylabel(f"{key} (mEh)")
#        axs[idx].set_title(f"{key} vs Iteration (R = {separation} bohr)")
#        # --- optional --- Set the x-axis limit to zoom into iterations 1-12
#        axs[idx].set_xlim(1, 12)
#        axs[idx].set_xlim(0, 6)
#        # --- optional ---
#        axs[idx].grid(True)
#        
#        # Zoom in the y-axis to magnify small differences.
#        if zoom_y:
#            # Filter out any None values.
#            numeric_vals = [y for y in scaled_y if y is not None]
#            if numeric_vals:
#                y_min = min(numeric_vals)
#                y_max = max(numeric_vals)
#                # Compute a padding as 10% of the range (or a small constant if values are nearly constant)
#                if abs(y_max - y_min) < 1e-6:
#                    pad = 0.1  # a default small padding in mEh
#                else:
#                    pad = 0.1 * abs(y_max - y_min)
#                
#                # Use custom limits if provided; otherwise, auto-compute.
#                lower = custom_y_limits[0] if custom_y_limits is not None else y_min - pad
#                upper = custom_y_limits[1] if custom_y_limits is not None else y_max + pad
#                axs[idx].set_ylim(lower, upper)
#
#    plt.tight_layout()
#    filename = f"energy_components_R{separation}_zoom.png"
#    fig.savefig(filename, format="png", dpi=300)
#    print(f"Zoomed energy components plot saved as {filename}")
#    plt.close(fig)
#
#    # Plot Energy Change if available, applying the same zoom strategy.
#    energy_change_vals = [entry.get("Energy change", None) for entry in energy_data if entry.get("Energy change", None) is not None]
#    if energy_change_vals:
#        iters_with_change = [entry["iteration"] for entry in energy_data if entry.get("Energy change", None) is not None]
#        scaled_ec = [ec * scale for ec in energy_change_vals]
#        fig2, ax2 = plt.subplots(figsize=(8, 5))
#        ax2.plot(iters_with_change, scaled_ec, marker='o', linestyle='-', color='orange')
#        ax2.set_xlabel("Iteration")
#        ax2.set_ylabel("Energy Change (mEh)")
#        ax2.set_title(f"Energy Change vs Iteration (R = {separation} bohr)")
#        # --- optional --- Set the x-axis limit to zoom into iterations 1-12
#        axs[idx].set_xlim(1, 12)
#        # --- optional ---
#        ax2.grid(True)
#        if zoom_y:
#            if scaled_ec:
#                y_min = min(scaled_ec)
#                y_max = max(scaled_ec)
#                if abs(y_max - y_min) < 1e-6:
#                    pad = 0.1
#                else:
#                    pad = 0.1 * abs(y_max - y_min)
#                lower = custom_y_limits[0] if custom_y_limits is not None else y_min - pad
#                upper = custom_y_limits[1] if custom_y_limits is not None else y_max + pad
#                ax2.set_ylim(lower, upper)
#        plt.tight_layout()
#        filename2 = f"energy_changes_R{separation}_zoom.png"
#        fig2.savefig(filename2, format="png", dpi=300)
#        print(f"Zoomed energy change plot saved as {filename2}")
#        plt.close(fig2)

def plot_energy_iterations_direct(energy_data, separation, system_name):
    """
    Given a list of dictionaries (each for one iteration) with keys:
       "iteration", "dEA", "dEB", "Elst+Exch", "Delta LA", "Delta LB",
       "Total Interaction Energy", and optionally "Energy change"
    this function generates line plots for the energy components versus iteration.
    It saves the figures as PNG images (since PNG is lossless and great for charts).
    """
    # Energy components to plot
    energy_keys = ["dEA", "dEB", "Elst+Exch", "Delta LA", "Delta LB", "Total Interaction Energy"]
    iterations = [entry["iteration"] for entry in energy_data]

    # Create a figure with subplots (3 rows x 2 columns)
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()

    for idx, key in enumerate(energy_keys):
        y_vals = [entry.get(key, None) for entry in energy_data]
        axs[idx].plot(iterations, y_vals, marker='o', linestyle='-', color='b')
        axs[idx].set_xlabel("Iteration")
        axs[idx].set_ylabel(f"{key} (Eh)")
        axs[idx].set_title(f"{key} vs Iteration (R = {separation} bohr) for {system_name}")
        
        # --- optional --- Set the x-axis limit to zoom into iterations 1-12
        #axs[idx].set_xlim(1, 12)
        #axs[idx].set_ylim(0, 6)
        # --- optional ---
        axs[idx].grid(True)

    plt.tight_layout()
    filename = f"energy_components_R{separation}_{system_name}.png"
    fig.savefig(filename, format="png", dpi=300)
    print(f"Energy components plot saved as {filename}")
    plt.close(fig)

    # If available, plot the energy change.
    energy_change_vals = [entry.get("Energy change", None) for entry in energy_data if entry.get("Energy change", None) is not None]
    if energy_change_vals:
        iters_with_change = [entry["iteration"] for entry in energy_data if entry.get("Energy change", None) is not None]
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(iters_with_change, energy_change_vals, marker='o', linestyle='-', color='orange')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Energy change (Eh)")
        ax2.set_title(f"Energy Change vs Iteration (R = {separation} bohr) for {system_name}")
        # --- optional --- Set the x-axis limit to zoom into iterations 1-12
        #axs[idx].set_xlim(1, 12)
        #axs[idx].set_ylim(0, 6)
        # --- optional ---
        ax2.grid(True)
        plt.tight_layout()
        filename2 = f"energy_changes_R{separation}_{system_name}.png"
        fig2.savefig(filename2, format="png", dpi=300)
        print(f"Energy change plot saved as {filename2}")
        plt.close(fig2)


def extract_separation_from_filename(filename):
    """
    Attempt to extract a separation value from the filename.
    For example, if the filename contains '_7_75', it returns '7.75';
    otherwise, returns 'unknown'.
    """
    m = re.search(r'_(\d+)[_.](\d+)', filename)
    if m:
        return m.group(1) + '.' + m.group(2)
    return "unknown"

def plot_energy_iterations_from_file(txt_filename, separation=None):
    """
    Parse a text file containing iterative energy-component data and generate
    plots of the energy components and (if available) the difference between the
    first-quantization (FQ) and second-quantization (SQ) results versus iteration.

    The expected format for each iteration block is something like:
    
      [Optional method tag:]
      FQ Energy components at iteration 1:
       dEA = 0.000013423123 Eh, 
       dEB = 0.000000000530 Eh, 
       Elst+Exch = -0.000100360839 Eh, 
       Delta LA = 0.000000854451 Eh, 
       Delta LB = 0.000000010779 Eh
      Total Interaction Energy at iteration 1: -0.000086071957 Eh

    Blocks for the SQ method (if available) should be similarly tagged with “SQ”.
    If no method tag is found, the block is assumed to belong to FQ.

    The function will produce:
      - One saved image with subplots of each energy component versus iteration.
      - If both FQ and SQ data are found, a second image showing the difference (FQ – SQ).

    The saved files will include the separation value in their names.
    """
    # If separation isn’t provided, try to extract it from the filename.
    if separation is None:
        separation = extract_separation_from_filename(txt_filename)
    else:
        separation = str(separation)
    
    # Dictionary to hold the data:
    # data = { "FQ": { 1: { 'dEA': val, ... }, 2: { ... } },
    #          "SQ": { 1: { ... } } }
    data = {}

    # Regular expression to detect the header.
    # It optionally accepts a method tag (FQ or SQ) followed by the standard header message.
    header_regex = re.compile(r"^(?:(FQ|SQ)\s+)?Energy components at iteration (\d+):", re.IGNORECASE)

    # Patterns for each energy component.
    patterns = {
        "dEA": re.compile(r"dEA\s*=\s*([-\d.e]+)", re.IGNORECASE),
        "dEB": re.compile(r"dEB\s*=\s*([-\d.e]+)", re.IGNORECASE),
        "Elst+Exch": re.compile(r"Elst\+Exch\s*=\s*([-\d.e]+)", re.IGNORECASE),
        "Delta LA": re.compile(r"Delta LA\s*=\s*([-\d.e]+)", re.IGNORECASE),
        "Delta LB": re.compile(r"Delta LB\s*=\s*([-\d.e]+)", re.IGNORECASE),
        "Total Interaction Energy": re.compile(r"Total Interaction Energy at iteration \d+:\s*([-\d.e]+)", re.IGNORECASE),
        "Energy change": re.compile(r"Energy change:\s*([-\d.e]+)", re.IGNORECASE)
    }
    
    current_method = None
    current_iter = None
    current_block = {}

    # Open and read the file.
    with open(txt_filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if the line starts a new energy block.
        header_match = header_regex.match(line)
        if header_match:
            # If there is a block already, save it.
            if current_iter is not None and current_method is not None and current_block:
                data.setdefault(current_method, {})[current_iter] = current_block
                current_block = {}
            
            # Get the method (if present) and iter number.
            method = header_match.group(1)
            # If not tagged, assume FQ.
            if not method:
                method = "FQ"
            else:
                method = method.upper()
            current_method = method
            current_iter = int(header_match.group(2))
            continue

        # Check if the line contains any of the energy values.
        for key, pattern in patterns.items():
            match = pattern.search(line)
            if match:
                try:
                    current_block[key] = float(match.group(1))
                except ValueError:
                    current_block[key] = None
                break  # Avoid matching a line more than once.

    # Save the last block if needed.
    if current_iter is not None and current_method is not None and current_block:
        data.setdefault(current_method, {})[current_iter] = current_block

    # Determine which energy components to plot.
    energy_keys = ["dEA", "dEB", "Elst+Exch", "Delta LA", "Delta LB", "Total Interaction Energy"]

    #############################
    # Plot energy components
    #############################
    # Create a figure with subplots (3 rows x 2 columns) for each energy component.
    nkeys = len(energy_keys)
    nrows = 3
    ncols = 2
    fig_components, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
    axs = axs.flatten()

    for idx, key in enumerate(energy_keys):
        ax = axs[idx]
        # Try to plot data for FQ and/or SQ if available.
        plotted_methods = []
        for method in ["FQ", "SQ"]:
            if method in data:
                # Get sorted iterations for which this component is present.
                iter_numbers = sorted(data[method].keys())
                # Get y-values if the component is present in a given iteration.
                y_vals = [data[method][it].get(key, None) for it in iter_numbers]
                if any(val is not None for val in y_vals):
                    ax.plot(iter_numbers, y_vals, marker='o', linestyle='-', label=method)
                    plotted_methods.append(method)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{key} (Eh)")
        ax.set_title(f"{key} vs Iteration (R = {separation} bohr)")
        if plotted_methods:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    
    components_filename = f"energy_components_R{separation}.png"
    fig_components.savefig(components_filename, format="png", dpi=300)
    print(f"Energy components plot saved as {components_filename}")

    #############################
    # Plot differences (if both methods are available)
    #############################
    if "FQ" in data and "SQ" in data:
        # Use the intersection of iteration numbers.
        common_iters = sorted(set(data["FQ"].keys()).intersection(set(data["SQ"].keys())))
        if common_iters:
            fig_diff, axs_diff = plt.subplots(nrows, ncols, figsize=(15, 10))
            axs_diff = axs_diff.flatten()
            for idx, key in enumerate(energy_keys):
                ax = axs_diff[idx]
                # For each common iteration get the value for FQ and SQ.
                y_FQ = []
                y_SQ = []
                for it in common_iters:
                    val_fq = data["FQ"].get(it, {}).get(key, None)
                    val_sq = data["SQ"].get(it, {}).get(key, None)
                    # Only include if both values are available.
                    if val_fq is not None and val_sq is not None:
                        y_FQ.append(val_fq)
                        y_SQ.append(val_sq)
                    else:
                        # For consistency in plotting, we append None (or could skip the iteration).
                        y_FQ.append(None)
                        y_SQ.append(None)
                # Now compute the difference for iterations where both values are available.
                diff_vals = []
                it_plot = []
                for i, it in enumerate(common_iters):
                    if y_FQ[i] is not None and y_SQ[i] is not None:
                        diff_vals.append(y_FQ[i] - y_SQ[i])
                        it_plot.append(it)
                if it_plot:
                    ax.plot(it_plot, diff_vals, marker='o', linestyle='-', color='purple')
                    ax.set_title(f"Difference in {key} (FQ - SQ)")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Difference (Eh)")
                else:
                    ax.text(0.5, 0.5, "No diff data", horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            diff_filename = f"energy_differences_R{separation}.png"
            fig_diff.savefig(diff_filename, format="png", dpi=300)
            print(f"Energy differences plot saved as {diff_filename}")
        else:
            print("No common iterations found for FQ and SQ methods. Skipping difference plots.")
    else:
        print("Data for both FQ and SQ methods not available; difference plots cannot be generated.")
    
    # Optionally, you can display the figures:
    # plt.show()
