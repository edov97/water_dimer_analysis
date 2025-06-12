"""
Utilities related to NO-PB iterations and Convergence
Current implementation considers MO basis for Omega construction 

##############################################################
SART calculation is performed iteratively for various schemes of
Omega potential and overlap expansion approximation
In each SART iteration:
    - Isolated monomer SCF is performed and orbitals are passed as the initial guess
    - Potential due to the partner is evaluated from previous iteration
    - Fock matrix is updated with the potential 
        - SCF iterations are performed for the perturbed Fock matrix
        - Relaxed orbitals are updated for next cycle of SART iteration
    - Convergence is checked for the SART interaction energy    
    
"""
import os
import csv
import copy
from pathlib import Path

import numpy as np
import opt_einsum as oe
import pandas as pd

import psi4
from psi4.driver.p4util.exceptions import ConvergenceError

from utils.helper_SAPT import helper_SAPT, sapt_timer
from utils.omega_exch_utils import (get_Exch_sinf, 
                                    form_omega_exchange_w_sym, 
                                    form_omega_exchange_no_sym
                                                    )

from utils.pb_utils import get_elst, diagonalise_fock
from utils.helper_HF import DIIS_helper
from utils.lm_utils import (get_delta_F, get_delta_W,
                                form_l_terms_w_sym,
                                form_m_terms_w_sym,
                                form_lm_terms_w_sym )

sart_inp_defaults = {
    'omega_elst': (False, False),    #(Omega_elst_present, only_ov_vo)
    'omega_exch': (False, False),
    'l_potential': (False, False), 
    'm_potential': (False, False), 
    'overlap_options': {
        'omega_exch_OO_VV': 'S2',
        'omega_exch_OV_VO': 'S_inf',
        'l': 'S2',
        'm': 'S2',
        'l+m': 'S2'
                },
    'sym_all': False,
    'end_run':'conv' # else, it will run till maxiter   
}

sart_out_defaults ={
    'output_dir': os.getcwd(),
    'calc_tag': 'SART',
    'to_write_matrices': False,
    'to_write_orbitals': False,
    'write_matrices_list': ['Omega', 'Omega(elst)', 'Omega(exch)', 'LM', 'Fock', 'Fock+Omega', 'Density'],
    
}
def do_sart_itr(dimer: psi4.geometry, 
                sapt:helper_SAPT,
                reference:str = 'RHF',
                sart_maxiter:int= 10,
                scf_maxiter:int= 100,
                E_conv= 1e-07, 
                input_dict:dict = sart_inp_defaults,
                output_dict:dict = sart_out_defaults,                
                # Geometry info passed with geom_index, in case of Energy-scan
                geom_index= None,
                *kwargs):

    """
    --takes the monomer info in psi4 geometry and helper_sapt(h_sapt) object,
    starts the zeroth iteration with unperturbed orbitals,:q
    In each iteration:
        --evaluates the omega and updates the perturned Fock matrices
        --diagonalize the focks to get modified orbitals
        --checks for convergence of E(int) = dEA + dEB + E_elst + E_exch
    --Returns the dictionary with zeroth and convergence values, 
        if convergence is reached
    """
    # Prints Geometry string and other important stuff at the begining
    print('--- Entering do_sart_itr ---')
    print(dimer.create_psi4_string_from_molecule())
    
    # Extracting monomers with Ghost atoms  
    monomerA = dimer.extract_subsets(1,2)
    monomerB = dimer.extract_subsets(2,1)

    access_var_timer = sapt_timer('Access Variables')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SART Potentials
    
    sym_all = input_dict.get('sym_all', True)
    print('sym_all:', sym_all)
    
    omega_elst = input_dict.get('omega_elst', (False,False))
    print('omega elst:', omega_elst)

    omega_elst_bool = omega_elst[0]
    omega_elst_all_blocks = omega_elst[1]
    
    omega_exch = input_dict.get('omega_exch', (False,False))
    omega_exch_bool = omega_exch[0]
    omega_exch_all_blocks = omega_exch[1]  
    
    l_potential = input_dict.get('l_potential', (False,False))
    l_bool = l_potential[0]
    l_all_blocks = l_potential[1]
    
    m_potential = input_dict.get('m_potential', (False,False))
    m_bool = m_potential[0]
    m_all_blocks = m_potential[1]
    
    overlap_options = input_dict.get('overlap_options')
    overlap_exch_oo_vv = overlap_options.get('omega_exch_OO_VV','S2')  
    overlap_exch_ov_vo = overlap_options.get('omega_exch_OV_VO','S_inf')  
    overlap_l = overlap_options.get('l','S2')
    overlap_m = overlap_options.get('m','S2')
    overlap_lm = overlap_options.get('l+m', 'S2')
    
    # SART calculation details
    end_run_condition = input_dict.get('end_run', 'conv')
    
    # SART output options
    output_dir = output_dict.get('output_dir', os.getcwd())
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
    calc_tag = output_dict.get('calc_prefix', dimer.name())
    to_write_orbitals = output_dict.get('to_write_orbitals', False)
    to_write_matrices = output_dict.get('to_write_matrices', False)
    write_matrices_list = output_dict.get('write_matrices_list', ['Omega', 'Omega(elst)', 'Omega(exch)', 'LM', 'Fock', 'Fock+Omega', 'Density'])
    
    # ======================= Printing calculation specifics ======================
    print('Printing Omega specifics started...............')
    if omega_elst_bool:
        print('\nOmega (Electrostatic) is present.')
        if omega_elst_all_blocks:
            print('Omega (Electrostatic) has all the blocks.')
        else:
            print('Omega (Electrostatic) has only OV/VO blocks.')
            
    if omega_exch_bool:
        print('\nOmega (Exchange) is present.')
        if omega_exch_all_blocks:
            print('Omega (Exchange) has all the blocks.')
        else:
            print('Omega (Exchange) has only OV/VO blocks.')
            
    if l_bool:
        print('\nLandshoff potential is present.')
        if l_all_blocks:
            print('Landshoff potential has all the blocks.')
        else:
            print('Landshoff potential has only OV/VO blocks.')  
            
    if m_bool:
        print('\nMurell potential is present.')
        if m_all_blocks:
            print('Murell potential has all the blocks.')
        else:
            print('Murell potential has only OV/VO blocks.')  
            
    print('Omega (Exchange) OO/VV blocks:', overlap_exch_oo_vv)
    print('Omega (Exchange) OV/VO blocks:', overlap_exch_ov_vo)
    print('Landshoff Potential, all', overlap_l)   
    print('Murell Potential, all', overlap_m)    

    access_var_timer.stop()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #======================= Printing calculation specifics DONE ====================
        
    print('\n############### Storing of Wavefunction variables started!')

    read_hsapt_timer = sapt_timer('Read helperSAPT object')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    h_sapt = sapt
    rhfA = h_sapt.rhfA
    rhfB = h_sapt.rhfB
    print('Monomer Energies for A and B:')
    print(rhfA, rhfB)

    # Intializing Energies to monomer energies
    Eprime_A = rhfA
    Eprime_B = rhfB

    EnucA = dimer.extract_subsets(1,2).nuclear_repulsion_energy()
    EnucB = dimer.extract_subsets(2,1).nuclear_repulsion_energy()
    EnucAB = dimer.nuclear_repulsion_energy()
    Enucint = EnucAB - EnucA - EnucB

    print('EnucAB:', EnucAB)
    print('EnucA:', EnucA)
    print('EnucB:', EnucB)

    print('Energy values extracted..........')

    ca = h_sapt.C_A
    cb = h_sapt.C_B

    # In DCBS, total number of MO's are same 
    # in both monomers
    nmo_A = h_sapt.nmo # wfnA.nmo()
    nmo_B = h_sapt.nmo # wfnB.nmo()

    ndocc_A = h_sapt.ndocc_A
    ndocc_B = h_sapt.ndocc_B

    print('Total number of MOs (in DCBS):', nmo_A)
    print('Number of occupied MO(A):', ndocc_A)
    print('Number of occupied MO(B):', ndocc_B)

    print('MO coefficients values extracted..........')

    mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())
    mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())

    A = mintsA.ao_overlap()
    A.power(-0.5, 1.e-16)
    A = np.asarray(A)

    S = h_sapt.S
    I = h_sapt.I.swapaxes(1,2)

    VA  = np.asarray(mintsA.ao_potential())
    TA = np.asarray(mintsA.ao_kinetic())
    VB  = np.asarray(mintsB.ao_potential())
    TB = np.asarray(mintsB.ao_kinetic())
    
    print('Kinetic and Potential matrices extracted.......')

    # TODO: use wavefunction object to extract info
    HA = TA + VA
    HB = TB + VB
    print('################### Storing from wavefunction variables completed!\n')    
    read_hsapt_timer.stop()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    CA_prev = ca
    CB_prev = cb
    E_int_old =  0.0
    E_int_old_w_lm = 0.0
    converged = False

    itr_list = []

    print('n\tm \tdEA\t\t dEB\t\t E_elst\t\t E_exch\t\t E_lm\t\t E_int\t\t E_int+lm\t\t dE\t\t dE(lm)' )
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)   
    
    # Write SART outputs
    if geom_index is not None:
        out_fname = f'sart_output_{calc_tag}_{geom_index}.txt'
    else:
        out_fname = f'sart_output_{calc_tag}.txt'
        
    # Writing Output file
    out_fpath = Path(output_dir)/out_fname
    with open(out_fpath, 'w') as f:
        f.write(f"\n#Dimer :{dimer.name()}\n")
        f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
        f.write(f"\n#Energy(Monomer A) :{rhfA}\n")
        f.write(f"\n#Energy(Monomer B) :{rhfB}\n")
        f.write(dimer.create_psi4_string_from_molecule())
        
        f.write(f"\n#Input Options :\n {input_dict}")
        f.write(f"\n#Output Options :\n{output_dict}")
        f.write('\nn\tm \tdEA\t\t dEB\t\t E_elst\t\t E_exch\t\t E_lm\t\t E_int\t\t E_int+lm\t\t dE\t\t dE(lm)' )

    # Write itr to CSV
    if geom_index is not None:
        csv_fname = f'sart_itr_{calc_tag}_{geom_index}.csv'
    else:
        csv_fname = f'sart_itr_{calc_tag}.csv'
    
    csv_fpath = Path(output_dir)/csv_fname   
    create_csv_file(filename= csv_fpath)
    
    # Writing matrices/Update step
    # NOTE: Used only for bug-testing
    if to_write_matrices:
        if geom_index is not None:
            sart_fname_matrices = f'sart_matrices_{calc_tag}_{geom_index}.txt'
        else:
            sart_fname_matrices = f'sart_matrices_{calc_tag}.txt'

        # Writing system info
        sart_fpath_matrices = Path(output_dir)/sart_fname_matrices
        with open(sart_fpath_matrices, 'w') as f:
            f.write(f"\n#Dimer :{dimer.name()}\n")
            f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
            f.write(f"\n#Energy(Monomer A) :{rhfA}\n")
            f.write(f"\n#Energy(Monomer B) :{rhfB}\n")
            f.write(dimer.create_psi4_string_from_molecule())

    # Writing Orbitals/Update step
    if to_write_orbitals:
        if geom_index is not None:            
            sart_fname_zero = f'sart_orbitals_{calc_tag}_{geom_index}_0.txt'      # Orbitals for itr= 0
            sart_fname_conv = f'sart_orbitals_{calc_tag}_{geom_index}_conv.txt'   # Orbitals for itr= conv
            sart_fname_end = f'sart_orbitals_{calc_tag}_{geom_index}_last.txt'    # Orbitals for itr= maxitr
            sart_fname_itr = f'sart_orbitals_{calc_tag}_{geom_index}_itr.txt'     # Orbitals at each itr           
            
        else:
            sart_fname_zero = f'sart_orbitals_{calc_tag}_0.txt'
            sart_fname_conv = f'sart_orbitals_{calc_tag}_conv.txt'
            sart_fname_end = f'sart_orbitals_{calc_tag}_last.txt'
            sart_fname_itr = f'sart_orbitals_{calc_tag}_itr.txt'
            
        sart_fpath_zero = Path(output_dir)/sart_fname_zero
        sart_fpath_conv = Path(output_dir)/sart_fname_conv
        sart_fpath_end = Path(output_dir)/sart_fname_end
        sart_fpath_itr = Path(output_dir)/sart_fname_itr

    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #                             NO-PB Iteration STARTS HERE!!!!                      #
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    sart_itr_timer = sapt_timer('NO-PB iterations')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(1, sart_maxiter + 1):
        if i == 1:
            fock_timer = sapt_timer('Constructing Fock Matrix in AO')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        CA_occ_prev = CA_prev[:,:ndocc_A]
        CB_occ_prev = CB_prev[:,:ndocc_B]
        
        DA_occ_prev = oe.contract('pi,qi->pq', CA_occ_prev, CA_occ_prev)
        DB_occ_prev = oe.contract('pi,qi->pq', CB_occ_prev, CB_occ_prev)
                
        JA_prev = oe.contract('pqrs,rs->pq', I, DA_occ_prev)
        JB_prev = oe.contract('pqrs,rs->pq', I, DB_occ_prev)

        if i == 1:
            fock_timer.stop()
            omega_elst_timer = sapt_timer('Constructing Omega Elst in AO, except to use OV-VO case')
        
        #============================================== 
        #         Omega(Electrostatic)                #
        # =============================================  
        WA_elst_ao_prev = None 
        WB_elst_ao_prev = None
        if omega_elst_bool:
            # Omega(elst) due to monomer A
            WA_elst_ao_prev = VA + 2*JA_prev 
            # Omega(elst) due to monomer B
            WB_elst_ao_prev = VB + 2*JB_prev

            if omega_elst_all_blocks:
                # By default, uses all blocks of Omega(elst)
                pass
            else:
                # ================ OV-VO block of Omega(electrostactic)=======================
                #------------ AO --> MO 
                WA0_elst_mo_prev = CB_prev.T.dot(WA_elst_ao_prev).dot(CB_prev)
                WB0_elst_mo_prev = CA_prev.T.dot(WB_elst_ao_prev).dot(CA_prev)
                    
                WA_elst_mo_prev = copy.deepcopy(WA0_elst_mo_prev)
                WB_elst_mo_prev = copy.deepcopy(WB0_elst_mo_prev)  

                # Omega(elst) due to monomer B
                ## VV block
                WB_elst_mo_prev[ndocc_A:, ndocc_A:] = np.zeros((nmo_A- ndocc_A, nmo_A- ndocc_A))
                ## OO block
                WB_elst_mo_prev[:ndocc_A, :ndocc_A] = np.zeros((ndocc_A, ndocc_A))

                # Omega(elst) due to monomer A
                ## VV block
                WA_elst_mo_prev[ndocc_B:, ndocc_B:] = np.zeros((nmo_B- ndocc_B, nmo_B- ndocc_B))
                ## OO block
                WA_elst_mo_prev[:ndocc_B, :ndocc_B] = np.zeros((ndocc_B, ndocc_B))

                #----------- MO --> AO
                WA_elst_ao_prev = S.dot(CB_prev).dot(WA_elst_mo_prev).dot(CB_prev.T).dot(S)
                WB_elst_ao_prev = S.dot(CA_prev).dot(WB_elst_mo_prev).dot(CA_prev.T).dot(S)

        if i ==1:
            omega_elst_timer.stop()
            omega_exch_timer = sapt_timer('Constructing Omega-Exch in MO, also LM Terms')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #================================ 
        #       Omega (Exchange)        #
        #================================
        WA_exch_mo_prev = None
        WB_exch_mo_prev = None
        if omega_exch_bool: 
            # if omega_exch_all_blocks:
                # *******************************************
                # NOTE:Symmetrization Scheme--> 1/2(PV +VP)
                # *******************************************
            if sym_all:
                # NOTE: If oo_vv is None, then we have NULL values in the diagonal blocks
                WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_w_sym(sapt= h_sapt,  # Use This
                                                                                ca= CA_prev,
                                                                                cb= CB_prev,
                                                                                oo_vv= overlap_exch_oo_vv,
                                                                                ov_vo= overlap_exch_ov_vo)
            else:
                WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_no_sym(sapt= h_sapt,
                                                                                ca= CA_prev,
                                                                                cb= CB_prev,
                                                                                oo_vv= overlap_exch_oo_vv,
                                                                                ov_vo= overlap_exch_ov_vo)
             
        if i == 1:
            omega_exch_timer.stop()
        
        #================================ 
        #           LM Potential        # 
        # ===============================        
        lA_mo_prev = None
        lB_mo_prev =None
        
        mA_mo_prev = None
        mB_mo_prev = None
        
        lmA_mo_prev = None
        lmB_mo_prev = None 
        if l_bool and m_bool:
            if overlap_lm is not None:
                if lmA_mo_prev is None and lmB_mo_prev is None:
                    lmA_mo_prev, lmB_mo_prev = form_lm_terms_w_sym(sapt= h_sapt, #Use This, They probably won't match
                                                                   ca= CA_prev,
                                                                   cb= CB_prev,
                                                                   s_option= overlap_lm
                                                                   )
        if l_bool:
            lA_mo_prev, lB_mo_prev = form_l_terms_w_sym(sapt= h_sapt,
                                                        ca= CA_prev,
                                                        cb= CB_prev,
                                                        s_option= overlap_l)
        if m_bool:            
            mA_mo_prev, mB_mo_prev = form_m_terms_w_sym(sapt= sapt,
                                                        ca= CA_prev,
                                                        cb= CB_prev) 
                    
        if lA_mo_prev is not None and mA_mo_prev is None:
            lmA_mo_prev = lA_mo_prev            
        elif lA_mo_prev is None and mA_mo_prev is not None:
            lmA_mo_prev = mA_mo_prev            
        elif lA_mo_prev is not None and mA_mo_prev is not None:
            lmA_mo_prev = lA_mo_prev + mA_mo_prev
            
        if lB_mo_prev is not None and mB_mo_prev is None:
            lmB_mo_prev = lB_mo_prev        
        elif lB_mo_prev is None and mB_mo_prev is not None:
            lmB_mo_prev = mB_mo_prev        
        elif lB_mo_prev is not None and mB_mo_prev is not None:
            lmB_mo_prev = lB_mo_prev + mB_mo_prev  

        # Omega Exchange + LM
        # =========================================
        WA_exch_lm_mo_prev = None
        WB_exch_lm_mo_prev = None
        WA_exch_lm_ao_prev = None     
        WB_exch_lm_ao_prev = None    
        
        if WA_exch_mo_prev is not None:
            WA_exch_lm_mo_prev = WA_exch_mo_prev.copy()
        if lmA_mo_prev is not None:
            WA_exch_lm_mo_prev += lmA_mo_prev
            
        if WB_exch_mo_prev is not None:
            WB_exch_lm_mo_prev = WB_exch_mo_prev.copy()            
        if lmB_mo_prev is not None:
            WB_exch_lm_mo_prev += lmB_mo_prev
            
        # MO --> AO
        # Omega Exchange + LM 
        # ========================================
        if WA_exch_lm_mo_prev is not None:
            WA_exch_lm_ao_prev = S.dot(CB_prev).dot(WA_exch_lm_mo_prev).dot(CB_prev.T).dot(S)
        
        if WB_exch_lm_mo_prev is not None:
            WB_exch_lm_ao_prev = S.dot(CA_prev).dot(WB_exch_lm_mo_prev).dot(CA_prev.T).dot(S)
            
        #================================================================
        #    Total = Omega(elst) + Omega(exch) + LM, in AO              #
        #================================================================  
        if WA_elst_ao_prev is not None:
            WA_tot_ao_prev = WA_elst_ao_prev
        if WA_exch_lm_ao_prev is not None:
            WA_tot_ao_prev += WA_exch_lm_ao_prev
        
        if WB_elst_ao_prev is not None:
            WB_tot_ao_prev = WB_elst_ao_prev
        if WB_exch_lm_ao_prev is not None:
            WB_tot_ao_prev += WB_exch_lm_ao_prev
        
        if WA_tot_ao_prev is None:
            WA_tot_ao_prev = np.zeros((nmo_B, nmo_B))
            
        if WB_tot_ao_prev is None:
            WB_tot_ao_prev = np.zeros((nmo_A, nmo_A))
        
        # NOTE: SCF iterations in Psi4
        #=======================================================  
        #           SCF Iterations(Monomer A)                  #
        #=======================================================
        # --- Added for debugging purposes ---
        printSQ_elst_exch = True
        if printSQ_elst_exch:
            WB_exch_ao_prev =  S.dot(CA_prev).dot(WB_exch_mo_prev).dot(CA_prev.T).dot(S)
            WA_exch_ao_prev =  S.dot(CB_prev).dot(WA_exch_mo_prev).dot(CB_prev.T).dot(S)
            WB_elst_exch_ao = WB_elst_ao_prev + WB_exch_ao_prev
            WA_elst_exch_ao = WA_elst_ao_prev + WA_exch_ao_prev
            WB_elst_exch_mo = CA_prev.T.dot(WB_elst_exch_ao).dot(CA_prev) 
            WA_elst_exch_mo = CB_prev.T.dot(WA_elst_exch_ao).dot(CB_prev)

        scf_A_timer = sapt_timer('SCF-iterations, A') 
        scfA_psi4_text = f"""
========================================== 
=               Monomer A                
=          Update itr {i-1}           
==========================================
        """

        scfB_psi4_text = f"""
========================================== 
=               Monomer B                
=          Update itr {i-1}           
==========================================
"""

        psi4.core.print_out(scfA_psi4_text)
        psi4.core.print_out('\n ...Psi4 will run SCF iterations now')
        try: 
            # Modified C coefficient matrix for Monomer A     
            wfnA_scf_conv = do_scf_itr(monomer= monomerA,
                                        reference= reference,
                                        guess= (CA_occ_prev, CA_occ_prev),
                                        omega= WB_tot_ao_prev,
                                        maxiter= scf_maxiter
                                        )
            # These will be used in (i+1)th itr
            CA_i = wfnA_scf_conv.Ca().to_array()
            eA_scf = wfnA_scf_conv.compute_energy()
        except Exception as e:
            print(e)
            continue            # continues to the later part of the code
        psi4.core.print_out(scfA_psi4_text)
        psi4.core.print_out('\n ...Finished SCF iterations now')
        scf_A_timer.stop()        

        #=======================================================  
        #           SCF Iterations(Monomer B)                  #
        #=======================================================
        scf_B_timer = sapt_timer('SCF-iterations, B') 

        psi4.core.print_out(scfB_psi4_text)
        psi4.core.print_out('\n ...Psi4 will run SCF iterations now')
        # Modified C coefficient matrix for Monomer B
        try:
            wfnB_scf_conv = do_scf_itr(monomer= monomerB,
                                        reference= reference,
                                        guess= (CB_occ_prev, CB_occ_prev),
                                        omega= WA_tot_ao_prev,
                                        maxiter= scf_maxiter
                                        )
            # These will be used in (i+1)th itr
            CB_i = wfnB_scf_conv.Ca().to_array()        
            eB_scf = wfnB_scf_conv.compute_energy()
        except Exception as e:
            print(e)
            continue
        psi4.core.print_out(scfB_psi4_text)
        psi4.core.print_out('\n ...Finished SCF iterations now')
        scf_B_timer.stop()        

        if i == 1:
            energy_calc_timer = sapt_timer('NO-PB Energy Calculation')

        # ===============================================
        #               Energy Calculation              #
        # ===============================================
        # ------------------itr i-1----------------------
        if i == 1:
            E_elst_timer = sapt_timer('Electrostatic Energy calc')   
        E_elst = 0 
        E_exch = 0 
        E_lm = 0
        
        # E(electrostatic + pol)
        if omega_elst_bool:
            E_elst = get_elst(Ca=CA_occ_prev, 
                                Cb=CB_occ_prev,
                                I=I,
                                VA=VA, VB=VB,  
                                Enucint= Enucint)
        if i == 1:
            E_elst_timer.stop()
            E_exch_timer = sapt_timer('Exchange Energy calc')
        
        # E(exch), Non-S2
        if omega_exch_bool:
            E_exch = get_Exch_sinf(
                            sapt= h_sapt,
                            ca= CA_prev,
                            cb=CB_prev
                        )
        if i == 1:
            E_exch_timer.stop()    

        # E(LM), Non-S2
        if l_bool is True or m_bool is True:
            if i == 1:
                E_lm_timer = sapt_timer('LM Energy calc')
            (del_FA, del_FB) = get_delta_F(sapt= h_sapt, ca= CA_prev, cb= CB_prev)
            (del_WA, del_WB) = get_delta_W(sapt=h_sapt, ca= CA_prev, cb= CB_prev)
            delta_A = del_FA + del_WA
            delta_B = del_FB + del_WB
            E_lm = delta_A + delta_B
            if i == 1:
                E_lm_timer.stop()

        if i == 1:
            E_def_timer = sapt_timer('Deformation Energy calc')

        # E(deformation of monomers) 
        dEA = (Eprime_A - rhfA)
        dEB = (Eprime_B - rhfB)

        Eprime_A = eA_scf
        Eprime_B = eB_scf

        if i == 1:
            E_def_timer.stop()

        # Total Interaction Energy Terms
        # =================================
        E_int = dEA + dEB + E_elst + E_exch
        E_int_w_lm = E_int + E_lm

        dE_int = E_int - E_int_old
        dE_int_lm = E_int_w_lm - E_int_old_w_lm

        if i ==1:
            energy_calc_timer.stop()            

        #============================================ 
        #               Write Options               #
        #============================================
        # TODO: Add itr=0 writing

        if to_write_orbitals:        
            # Writing orbitals/Update step                 
            with open(sart_fpath_itr, 'a') as f:
                f.write(f"\n#Dimer :{calc_tag}\n")
                f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
                f.write(dimer.create_psi4_string_from_molecule())
                f.write(f"\n#itr:{i-1}")
                f.write("\n#CA Orbitals(isolated):\n")
                np.savetxt(f, CA_prev, delimiter=','
                        )
                f.write("\n#CB Orbitals(isolated):\n")
                np.savetxt(f, CB_prev, delimiter=',', 
                        )

        if i == 1:
            write_itr_timer = sapt_timer('Writing matrices for each itr')

        ###############################################################################
        # TODO: CHECK? If used, this part has the options to write matrices/Update step
        if to_write_matrices:
            #============================================ 
            #           Added for Bug-testing           #           
            #============================================
            with open(sart_fpath_matrices, 'a') as f:
                f.write(f"\n#Dimer :{dimer.name()}\n")
                f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
                f.write(f"\n#=============== itr:{i-1}")
                f.write(f"\n#def EA:{dEA}")
                f.write(f"\n#def EB:{dEB}")
                f.write(f"\n#E_elst:{E_elst}")
                f.write(f"\n#E_exch:{E_exch}")
                f.write(f"\n#E_lm:{E_lm}")
                f.write(f"\n#E_int:{E_int}")
                
                f.write(f"\n#Dimer :{dimer.name()}\n")
                f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
                f.write(f"\n#itr:{i-1}")              
                
                # Monomer A
                #============================================================================
                # SART Omega potentials due to monomer B is projected on the MO's of monomer A
                WB_tot_mo_prev = np.zeros((nmo_A, nmo_A))
                
                if 'Omega' in write_matrices_list:  
                    f.write("\n#=================== Omega (in MO)====================\n")
                    #------------ AO --> MO 
                    if WB_elst_ao_prev is not None:
                        WB_elst_mo_prev = CA_prev.T.dot(WB_elst_ao_prev).dot(CA_prev)  
                        f.write("\n#Omega-elst(B),MO:\n")
                        np.savetxt(f, WB_elst_mo_prev, delimiter=',', fmt='%f')  
                    else:
                        WB_elst_mo_prev = np.zeros((nmo_A, nmo_A))
                    WB_tot_mo_prev += WB_elst_mo_prev
                    
                    if WB_exch_mo_prev is not None:                                
                        f.write("\n#Omega-exch(B),MO:\n")
                        np.savetxt(f, WB_exch_mo_prev, delimiter=',',fmt='%f')
                    else:
                        WB_exch_mo_prev = np.zeros((nmo_A, nmo_A))
                    WB_tot_mo_prev += WB_exch_mo_prev
                    
                if 'LM' in write_matrices_list:
                    if lB_mo_prev is not None:
                        f.write("\n#L(B),MO:\n")
                        np.savetxt(f, lB_mo_prev, delimiter=',',fmt='%f')
                        
                    if mB_mo_prev is not None:
                        f.write("\n#M(B),MO:\n")
                        np.savetxt(f, mB_mo_prev, delimiter=',',fmt='%f')
                        
                    if lmB_mo_prev is not None:
                        f.write("\n#LM(B),MO:\n")
                        np.savetxt(f, lmB_mo_prev, delimiter=',',fmt='%f')
                    else:
                        lmB_mo_prev = np.zeros((nmo_A, nmo_A))
                    WB_tot_mo_prev += lmB_mo_prev
                            
                f.write("\n#Omega-total(B),MO:\n")
                #------------ AO --> MO 
                np.savetxt(f, WB_tot_mo_prev, delimiter=',',fmt='%f')
                            
                f.write("\n#=================== Omega (in AO)====================\n")                
                f.write("\n#Omega-total(B),AO:\n")
                np.savetxt(f, WB_tot_ao_prev, delimiter=',',fmt='%f')
                
                if 'Fock' in write_matrices_list:
                    # Monomer A
                    f.write(f"\n#================= Fock ===================")
                    f.write(f"\n#SCF EA:{Eprime_A}")
                    
                    KA_prev = oe.contract('prqs,rs->pq', I, DA_occ_prev)
                    FA_ao_prev = HA + 2*JA_prev - KA_prev
                    f.write("\n#Fock(A),AO:\n")
                    np.savetxt(f, FA_ao_prev, delimiter=',',fmt='%f')  
                                        
                    f.write("\n#Fock-modified(A),AO = FA+Omega(B):\n")
                    np.savetxt(f, FA_ao_prev + WB_tot_ao_prev, delimiter=',',fmt='%f')                 
                    
                    # shape(CB_prev) = (nbas, nmo_B) 
                    # In DCBS, nmoA = nmoB = nbas/nao
                    # Transformation AO --> MO(A)
                    FA_mo_prev = CA_prev.T.dot(FA_ao_prev).dot(CA_prev)
                    f.write("\n#Fock(A),MO:")
                    f.write("\n#MO's of A:\n")
                    np.savetxt(f, FA_mo_prev, delimiter=',',fmt='%f')
                                        
                    f.write("\n#Fock-modified(A),MO:")
                    f.write("\n#MO's of A:\n")
                    np.savetxt(f, FA_mo_prev + WB_tot_mo_prev, delimiter=',',fmt='%f') 
                
                # Monomer B
                #============================================================================
                # SART Omega potentials due to monomer A is projected on the MO's of monomer B
                WA_tot_mo_prev = np.zeros((nmo_B, nmo_B))
                
                if 'Omega' in write_matrices_list:  
                    f.write("\n#=================== Omega (in MO)====================\n")
                    #------------ AO --> MO 
                    if WA_elst_ao_prev is not None:
                        WA_elst_mo_prev = CB_prev.T.dot(WA_elst_ao_prev).dot(CB_prev)  
                        f.write("\n#Omega-elst(A),MO:\n")
                        np.savetxt(f, WA_elst_mo_prev, delimiter=',', fmt='%f')  
                    else:
                        WA_elst_mo_prev = np.zeros((nmo_B, nmo_B))
                    WA_tot_mo_prev += WA_elst_mo_prev
                    
                    if WA_exch_mo_prev is not None:                                
                        f.write("\n#Omega-exch(A),MO:\n")
                        np.savetxt(f, WA_exch_mo_prev, delimiter=',',fmt='%f')
                    else:
                        WA_exch_mo_prev = np.zeros((nmo_B, nmo_B))
                    WA_tot_mo_prev += WA_exch_mo_prev
                    
                if 'LM' in write_matrices_list:
                    if lA_mo_prev is not None:
                        f.write("\n#L(A),MO:\n")
                        np.savetxt(f, lA_mo_prev, delimiter=',',fmt='%f')
                        
                    if mA_mo_prev is not None:
                        f.write("\n#M(A),MO:\n")
                        np.savetxt(f, mA_mo_prev, delimiter=',',fmt='%f')
                        
                    if lmA_mo_prev is not None:
                        f.write("\n#LM(A),MO:\n")
                        np.savetxt(f, lmA_mo_prev, delimiter=',',fmt='%f')
                    else:
                        lmA_mo_prev = np.zeros((nmo_B, nmo_B))
                    WA_tot_mo_prev += lmA_mo_prev
                            
                f.write("\n#Omega-total(A),MO:\n")
                #------------ AO --> MO 
                np.savetxt(f, WA_tot_mo_prev, delimiter=',',fmt='%f')
                            
                f.write("\n#=================== Omega (in AO)====================\n")                
                f.write("\n#Omega-total(A),AO:\n")
                np.savetxt(f, WA_tot_ao_prev, delimiter=',',fmt='%f')
                
                if 'Fock' in write_matrices_list:     
                    # Monomer B
                    f.write(f"\n#================= Fock ===================")
                    f.write(f"\n#SCF EB:{Eprime_A}")
                    
                    KB_occ_prev = oe.contract('prqs,rs->pq', I, DB_occ_prev)
                    FB_ao_prev = HB + 2*JB_prev - KB_occ_prev
                    f.write("\n#Fock(B),AO:\n")
                    np.savetxt(f, FB_ao_prev, delimiter=',',fmt='%f')  
                                        
                    f.write("\n#Fock-modified(B),AO:\n")
                    np.savetxt(f, FB_ao_prev + WA_tot_ao_prev, delimiter=',',fmt='%f')                 
                    
                    # shape(CB_prev) = (nbas, nmo_B) 
                    # In DCBS, nmoA = nmoB = nbas/nao
                    # Transformation AO --> MO(A)
                    FB_mo_prev = CB_prev.T.dot(FB_ao_prev).dot(CB_prev)
                    f.write("\n#Fock(B),MO:")
                    f.write("\n#MO's of B:\n")
                    np.savetxt(f, FB_mo_prev, delimiter=',',fmt='%f')
                                        
                    f.write("\n#Fock-modified(B),MO:")
                    f.write("\n#MO's of B:\n")
                    np.savetxt(f, FB_mo_prev + WA_tot_mo_prev, delimiter=',',fmt='%f')                                    
                    
                if 'Density' in write_matrices_list:            
                    pass
                
        #######################################################################################################        
        print(f'{(i-1):3d}\t{(i-1):3d}  {dEA:.8e}  {dEB:.8e}  {E_elst:.8e}  {E_exch:.8e} {E_lm:.8e} {E_int:.8e} {E_int_w_lm:.8e}  {(dE_int):.8e}  {(dE_int_lm):.8e}')
        with open(out_fpath, 'a') as f:
            f.write(f'\n{(i-1):3d}\t{(i-1):3d}  {dEA:.8e}  {dEB:.8e}  {E_elst:.8e}  {E_exch:.8e} {E_lm:.8e} {E_int:.8e} {E_int_w_lm:.8e}  {(dE_int):.8e}  {(dE_int_lm):.8e}')
        if i == 1:
            write_itr_timer.stop()
            data_handle_timer = sapt_timer('Handling the data in each itr')

        # Saving iterations to dataframe 
        # and return it at max itr value
        itr_dict = {
                'dEA': dEA,
                'dEB': dEB,
                'E_elst': E_elst,
                'E_exch': E_exch,
                'E_lm': E_lm,                
                'E_int': E_int,
                'E_int+lm': E_int_w_lm,
                'dE_int': dE_int,
                'dE_int_lm': dE_int_lm,
            }
        if l_bool is True or m_bool is True:
            itr_dict.update({
                'E_FA': del_FA,
                'E_FB': del_FB,
                'E_WA': del_WA,
                'E_WB': del_WB,
            })
        itr_list.append(itr_dict)
        E1 = E_elst + E_exch
        delta_LMA = del_FA + del_WA
        delta_LMB = del_FB + del_WB
        
        #================================================
        #          Write NO-PB Energy/Update Itr        #        
        #================================================
        # Update Iterations to CSV at every iteration dynamically
        write_itr_to_csv(data= itr_dict,
                         csv_fname= csv_fpath)
        
        # =========================================================
        #                       Zeroth itr 
        #           (with unperturbed monomer orbitals)
        # =========================================================
        # if i == 1:
        #     # NOTE:Commented out for single point calculation
        #     # ===============================================
        #     # JSON Data Handing
        #     write_itr_to_json(dimer_name= dimer.name(),
        #                       key= 'zeroth',
        #                     itr_dict= itr_dict)
            
        #============================= Check for convergence criteria
        #                       Converged itr 
        #               (with converged monomer orbitals)
        # (i>1) condition is added to avoid convergence on zeroth itr
        # ===========================================================
        #  Convergence for E_int(+ LM)
        if end_run_condition in ['conv', 'CONV']:        
            if (abs(dE_int_lm) < E_conv) and i>1 and not converged: 
                print('\n##############################################')
                print(f'!!!!       CONVERGED at itr  {i-1}         !!!!')
                print('################################################\n')
                sart_itr_timer.stop()

                converged = True
                CA_CONV = CA_i
                CB_CONV = CB_i

                # If convergence is reached, writes the conv orbitals
                # Also writes the conv itr dict to JSON
                # Writing Converged orbitals
                if to_write_orbitals:
                    with open(sart_fpath_conv, 'a') as f:
                        f.write(f"\n#Dimer :{dimer.name()}\n")
                        f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
                        f.write(f"\n#Total number of MOs (in DCBS):{nmo_A}")
                        f.write(f"\n#Number of occupied MO(A):{ndocc_A}")
                        f.write(f"\n#Number of occupied MO(B):{ndocc_B}")
                        f.write(f"\n#itr:{i-1}")
                        f.write(f"\n#CA Orbitals(itr: {i-1}):\n")
                        np.savetxt(f, CA_prev, delimiter=','
                                )
                        f.write(f"\n#CB Orbitals(itr: {i-1}):\n")
                        np.savetxt(f, CB_prev, delimiter=',', 
                                )
                                        # Remove the next variables when not using the FQ routines.
                return CA_CONV, CB_CONV, E_int_w_lm, dEA, dEB, E1, delta_LMA, delta_LMB, itr_list
            # NOTE:Commented out for single point calculation
            # ===============================================
            # JSON Data Handing
            # write_itr_to_json(dimer_name= dimer.name(),
            #                   key= 'conv',
            #                   itr_dict= itr_dict)
        else:
            pass
            

        # Update for next iteration
        CA_prev = CA_i
        CB_prev = CB_i

        E_int_old = E_int
        E_int_old_w_lm = E_int_w_lm

        if i == 1:
            data_handle_timer.stop()

        # Writes the last orbital sets 
        # and return all iterations as a list
        if i == sart_maxiter:  
            sart_itr_timer.stop()          
            # Writing last itr orbitals
            if to_write_orbitals:
                with open(sart_fpath_end, 'a') as f:
                    f.write(f"\n#Dimer :{dimer.name()}\n")
                    f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
                    f.write(f"\n#Total number of MOs (in DCBS):{nmo_A}")
                    f.write(f"\n#Number of occupied MO(A):{ndocc_A}")
                    f.write(f"\n#Number of occupied MO(B):{ndocc_B}")

                    f.write(f"\n#itr:{i-1}")
                    f.write(f"\n#CA Orbitals(itr: {i-1}):\n")
                    np.savetxt(f, CA_prev, delimiter=','
                            )
                    f.write(f"\n#CB Orbitals(itr: {i-1}):\n")
                    np.savetxt(f, CB_prev, delimiter=',', 
                            )
            psi4.core.clean()
            # NOTE: Raises Convergence Error for NO-PB iterations
            # Or returns itr_list (SART energy/itr dataset as a list)
            if end_run_condition in ['conv', 'CONV']:   
                raise ConvergenceError(eqn_description='NO-PB Interaction',
                                   iteration=i,
                                   additional_info="Maximum number of NO-PB iterations exceeded.")
            else:
                # itr_df = pd.DataFrame(itr_list)
                return CA_CONV, CB_CONV, E_int_w_lm, dEA, dEB, E1, delta_LMA, delta_LMB, itr_list        
            
#######################################################################################################

import json
import pathlib

def write_dimer_to_json(dimer_name:str, dimer_dict:dict=None):
    """
    Initiates the json object and writes to a json file
    TODO:Check for same keyError
    """
    new_dimer_obj = {
        dimer_name: {'sys_info': dimer_dict,
                    'calc_info': []
                    }
                    }
        
    json_obj = json.dumps(new_dimer_obj)

    json_fname = 'sart_en.json'
    # Writing to json
    with open(json_fname, "w") as outfile:
        outfile.write(json_obj)

def write_geom_to_json(dimer_name,geometry_dict:dict):
    """
    Updates the geometry_dict w/ geometry info
    Appends geometry item to the dimer_list
    """
    json_fname = 'sart_en.json'
    if pathlib.Path(json_fname).exists():
        with open(json_fname, 'r') as file:
            data = json.load(file)

        if dimer_name in data.keys():
            list_geom = data[dimer_name]['calc_info']    # list of different geometries for a given dimer
            list_geom.append(geometry_dict)
        else:
            raise KeyError(f'{dimer_name} is missing in json object.')
            # data.update(dict_of_dimers)
        json_obj = json.dumps(data, indent=4)
    else:
        raise FileNotFoundError(f'{json_fname} does not exist.')

    # Writing to json
    with open("sart_en.json", "w") as outfile:
        outfile.write(json_obj)             


def write_itr_to_json(dimer_name, 
                  key,
                  itr_dict
                  ):
    """
    Writes iteration values to json file
    key:str: zeroth/conv
    itr_dict:Dict: {'dEA': ...,
                    'dEB': ...,
                    'E_elst': ...,
                    'E_exch': ...,
                    'E_int': ...}
    """

    jsonpath_str = f'sart_en.json'
    if pathlib.Path(jsonpath_str).exists():
        with open('sart_en.json', 'r') as file:
            data = json.load(file)

        if dimer_name in data.keys():
            list_geom = data[dimer_name]['calc_info']     # list of different geometries for a given dimer

            # If the list has item(in between scan calculation), 
            # the last item is considered to be the current geometry
            if len(list_geom)>0:
                last_geom_item = list_geom[-1]  # last item is a dictionary for the last geometry
                last_geom_item.update({key: itr_dict})
            else:
                # If an empty list is found, [key:itr_dict] is added
                list_geom.append({key: itr_dict})
        else:
            raise KeyError(f'{dimer_name} is missing in json object.')
            # data.update(dict_of_dimers)
        json_obj = json.dumps(data, indent=4)
    else:
        raise FileNotFoundError(f'{jsonpath_str} does not exist.')

    # Writing to json
    with open("sart_en.json", "w") as outfile:
        outfile.write(json_obj) 

def create_csv_file(filename):
    itr_dict = {
                'dEA': None,
                'dEB': None,
                'E_elst': None,
                'E_exch': None,
                'E_lm': None,                
                'E_int': None,
                'E_int+lm': None,
                'dE_int': None,
                'dE_int_lm': None,
                'E_FA': None,
                'E_FB': None,
                'E_WA': None,
                'E_WB': None,
            }
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=itr_dict.keys())

        # Write header
        writer.writeheader()

def write_itr_to_csv(data, csv_fname):
    print('Will write to CSV')
    file_exists = os.path.exists(csv_fname) and os.path.getsize(csv_fname) > 0
    
    with open(csv_fname, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())

        # Write header only if the file is new/empty
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

#######################################################################

def do_scf_itr(monomer:psi4.geometry, 
                 reference:str,
                guess:tuple,
                omega: np.ndarray, 
                maxiter:int=None, 
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
    print('........will start scf-iterations')
    psi4.set_options({'MAXITER': maxiter})
    
    # Constructing base wavefunction and then RHF/HF object
    base_wfn = psi4.core.Wavefunction.build(monomer, 
                                        psi4.core.get_global_option('BASIS'))
    # print('Base WFN constructed...')
    wfn_ref_obj= psi4.driver.scf_wavefunction_factory('SCF', 
                                                ref_wfn=base_wfn,
                                                reference= reference)
    # print('RHF object constructed...')
    # Access the GUESS and set these
    Ca = guess[0]
    Cb = guess[1]
    Ca_psi4_mat = psi4.core.Matrix.from_array(Ca)
    Cb_psi4_mat = psi4.core.Matrix.from_array(Cb)
    # print('GUESS are extracted...')
    wfn_ref_obj.guess_Ca(Ca_psi4_mat)
    wfn_ref_obj.guess_Cb(Cb_psi4_mat)
    # print('GUESS are set...')

    # Initialize for SCF Run
    wfn_ref_obj.initialize()
    # print('After initializing.....Check if it has the correct GUESS loaded')
    # print(wfn_ref_obj.Ca().to_array())
    # print(wfn_ref_obj.Cb().to_array())

    # Prepare the Omega matrix
    #print('Omega to be added to Fock')
    #print(omega)
    Omega_psi4_mat = psi4.core.Matrix.from_array(omega)
    wfn_ref_obj.push_back_external_potential(Omega_psi4_mat)

    # Start the SCF runs and save
    wfn_ref_obj.iterations()
    wfn_ref_obj.save_density_and_energy()

    # print('After SCF iterations.....Modified')
    # print(wfn_ref_obj.Ca().to_array())
    # e_scfomega = wfn_ref_obj.compute_energy()
    # print('Energy with the modified orbitals:....')
    # print(e_scfomega)

    print('.... Finished scf-iterations')

    return wfn_ref_obj