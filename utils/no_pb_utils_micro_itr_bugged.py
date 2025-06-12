"""
Utilities related to NO-PB iterations, Convergence
Current implementation considers MO basis

**** Modified to store Matrices for NO-PB itr if needed
**** Modified to consider only omega_elst/exch boolean options
**** Exchange(OV-VO) default is Non-S2, Other blocks in S2

**** Modified to consider h_sapt object to access monomer info 
##############################################################

**** Modified to have the OMEGA-SYMMETRIZED scheme
**** Modified to update CSV dynamically, by default
**** Cleaned redundant Code, 
**** TODO: DOES not have writing matrices(commented out)
**** TODO: Add option to use the convergence criteria/run till maxiter
"""
import os
import csv
import pdb

import numpy as np
import opt_einsum as oe
import copy
import psi4
from psi4.driver.p4util.exceptions import ConvergenceError

from utils.helper_SAPT import helper_SAPT, sapt_timer
from utils.omega_exch_utils import (form_omega_exchange_s2_total, 
                                         form_omega_exchange_sinf_total,
                                         form_omega_exchange_s2_sym,
                                         form_omega_exchange_s2, 
                                         form_lm_terms_s2,
                                        get_Exch_s2, 
                                        form_omega_exchange_sinf, 
                                        form_omega_exchange_sinf_sym,
                                        get_Exch_sinf
                                                    )
from utils.pb_utils import get_elst, diagonalise_fock
from utils.helper_HF import DIIS_helper
from utils.lm_utils import get_delta_F, get_delta_W

def do_nopb_itr(dimer: psi4.geometry, 
                sapt:helper_SAPT,
                maxiter:int=10, 
                non_s2:bool=True, 
                omega_exch = (False, False),
                omega_elst= (False,False),
                lm_bool = False,
                sym_all = False,
                diis_bool:bool=False,  # diis_bool is redundant in case scf-itr, since Psi4 handles it
                to_write_matrices:bool = False,
                to_write_orbitals:bool = False,
                # Geometry info passed here, in case of Energy-scan
                geom_index= None):

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
    # TODO: Print Geometry string and other important stuff at the begining
    # Extracting monomers with Ghost atoms
    monomerA = dimer.extract_subsets(1,2)
    monomerB = dimer.extract_subsets(2,1)

    access_var_timer = sapt_timer('Access Variables')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # If true, only Omega(elst) is used 
    omega_elst_only = omega_elst[0]
    # If True, all blocks of Omega(elst) is used
    # else, OV-VO blocks will be used
    omega_elst_all_blocks = omega_elst[1]

    # If true, only Omega(exch) is used 
    omega_exch_only = omega_exch[0]
    # If True, all blocks of Omega(exch) is used
    # else, OV-VO blocks will be used
    omega_exch_all_blocks = omega_exch[1]

    # ======================= Printing calculation specifics ======================
    if not omega_elst_only:
        if omega_exch_only:
            print('\n ######### Omega is Only Exchange!')   
        else:
            print('\n ######### Omega is Electrostatic + Exchange!')
        
        if omega_exch_all_blocks:
            print('\n ######### Omega(exch) has all the blocks(OO-VV, S2 & OV-VO)')
        else:
            print('\n ######### Omega(exch) has only (OV-VO)blocks, (NOT OO-VV)')
    else:
        print('\n ######### Omega is Only Electrostatic. No Exchange potential!')

    if omega_elst_all_blocks:
        print('\n ######### Omega(elst) has all the blocks(including OO-VV)')
    else:
        print('\n ######### Omega(elst) has (OV-VO)blocks, (NOT OO-VV)') 

    if non_s2:
        print('\n ######### OV-VO blocks of Omega(exch) are Non-S2')
    else:
        print('\n ######### OV-VO blocks of Omega(exch) are S2')

    if sym_all:
        print('\n Uniform Symmetrization applied !!')
    else: 
        print('\n No Uniform Symmetrization applied !!')

    if lm_bool:
        print('\n ######### LM terms are added to the Omega potential.')
    else:
        print('\n ######### No LM terms are added to the Omega potential.')       

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
    E_conv = 1.0E-8
    converged = False

    itr_list = []

    print('n\tm \tdEA\t\t dEB\t\t E_elst\t\t E_exch\t\t E_lm\t\t E_int\t\t E_int+lm\t\t dE\t\t dE(lm)' )
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # Writing matrices/Update step
    # NOTE: Used only for bug-testing
    if to_write_matrices:
        if geom_index is not None:
            no_pb_fname_matrices = f'no_pb_matrices_{dimer.name()}_{geom_index}.txt'
        else:
            no_pb_fname_matrices = f'no_pb_matrices_{dimer.name()}.txt'

        # Writing system info
        with open(no_pb_fname_matrices, 'w') as f:
            f.write(f"\n#Dimer :{dimer.name()}\n")
            f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
            f.write(f"\n#Energy(Monomer A) :{rhfA}\n")
            f.write(f"\n#Energy(Monomer B) :{rhfB}\n")

    # Writing Orbitals/Update step
    if to_write_orbitals:
        if geom_index is not None:            
            no_pb_fname_zero = f'no_pb_orbitals_{dimer.name()}_{geom_index}_0.txt'      # Orbitals for itr= 0
            no_pb_fname_conv = f'no_pb_orbitals_{dimer.name()}_{geom_index}_conv.txt'   # Orbitals for itr= conv
            no_pb_fname_end = f'no_pb_orbitals_{dimer.name()}_{geom_index}_last.txt'    # Orbitals for itr= maxitr
            no_pb_fname_itr = f'no_pb_orbitals_{dimer.name()}_{geom_index}_itr.txt'     # Orbitals at each itr           
            
        else:
            no_pb_fname_zero = f'no_pb_orbitals_{dimer.name()}_0.txt'
            no_pb_fname_conv = f'no_pb_orbitals_{dimer.name()}_conv.txt'
            no_pb_fname_end = f'no_pb_orbitals_{dimer.name()}_last.txt'
            no_pb_fname_itr = f'no_pb_orbitals_{dimer.name()}_itr.txt'

    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #                             NO-PB Iteration STARTS HERE!!!!                      #
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    no_pb_itr_timer = sapt_timer('NO-PB iterations')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(1, maxiter + 1):
        if i == 1:
            fock_timer = sapt_timer('Constructing Fock Matrix in AO')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        CA_occ_prev = CA_prev[:,:ndocc_A]
        CB_occ_prev = CB_prev[:,:ndocc_B]
        
        DA_prev = oe.contract('pi,qi->pq', CA_occ_prev, CA_occ_prev)
        DB_prev = oe.contract('pi,qi->pq', CB_occ_prev, CB_occ_prev)

        JA_prev = oe.contract('pqrs,rs->pq', I, DA_prev)
        JB_prev = oe.contract('pqrs,rs->pq', I, DB_prev)

        if i == 1:
            fock_timer.stop()
            omega_elst_timer = sapt_timer('Constructing Omega Elst in AO, except to use OV-VO case')
        
        #============== Omega(Electrostatic)=============================================   
        WA_elst_ao_prev = VA + 2*JA_prev 
        WB_elst_ao_prev = VB + 2*JB_prev

        if omega_elst_all_blocks:
                pass
        else:
            # ================ OV-VO block of Omega(electrostactic)=======================
            #------------ AO --> MO 
            WA0_elst_mo_prev = CB_prev.T.dot(WA_elst_ao_prev).dot(CB_prev)
            WB0_elst_mo_prev = CA_prev.T.dot(WB_elst_ao_prev).dot(CA_prev)
                 
            WA_elst_mo_prev = copy.deepcopy(WA0_elst_mo_prev)
            WB_elst_mo_prev = copy.deepcopy(WB0_elst_mo_prev)  

            # # VV block
            WB_elst_mo_prev[ndocc_A:, ndocc_A:] = np.zeros((nmo_A- ndocc_A, nmo_A- ndocc_A))
            ## OO block
            WB_elst_mo_prev[:ndocc_A, :ndocc_A] = np.zeros((ndocc_A, ndocc_A))

            # VV block
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
        
        #========================== Omega (Exchange)================
        if not omega_elst_only: 
            if omega_exch_all_blocks:
                # *******************************************
                # NOTE:Symmetrization Scheme--> 1/2(PV +VP)
                # *******************************************
                if sym_all:
                    if non_s2:
                        # Partial Non-S2 : 
                        # OV-VO(Non-S2), OO-VV(S2)
                        WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_sinf_sym(h_sapt, 
                                                                                    ca= CA_prev, 
                                                                                    cb=CB_prev,
                                                                                    ov_vo= False)
                        #  LM-terms (S2)
                        if lm_bool:
                            lmA_mo_prev, lmB_mo_prev = form_lm_terms_s2(h_sapt,
                                                                        ca= CA_prev,
                                                                        cb=CB_prev, 
                                                                        sym_all=True)
                            WA_exch_mo_prev= WA_exch_mo_prev + lmA_mo_prev
                            WB_exch_mo_prev = WB_exch_mo_prev + lmB_mo_prev
                    else:
                        # All blocks (S2)
                        WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_s2_sym(h_sapt, 
                                                                                    ca= CA_prev, 
                                                                                    cb=CB_prev)
                        #  LM-terms (S2)
                        if lm_bool:
                            lmA_mo_prev, lmB_mo_prev = form_lm_terms_s2(h_sapt,
                                                                        ca= CA_prev,
                                                                        cb=CB_prev, 
                                                                        sym_all=True)
                            WA_exch_mo_prev= WA_exch_mo_prev + lmA_mo_prev
                            WB_exch_mo_prev = WB_exch_mo_prev + lmB_mo_prev
                else:
                    # ******************************************************
                    # NOTE:Symmetrization Scheme--> <VP> for OV, <PV> for VO
                    # ******************************************************
                    # Partial Non-S2 : 
                    # OV-VO(Non-S2), OO-VV(S2)
                    if non_s2:
                        WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_sinf_total(h_sapt, 
                                                                                    ca= CA_prev, 
                                                                                    cb=CB_prev)
                        #  LM-terms (S2)
                        if lm_bool:
                            lmA_mo_prev, lmB_mo_prev = form_lm_terms_s2(h_sapt,
                                                                        ca= CA_prev,
                                                                        cb=CB_prev)
                            WA_exch_mo_prev= WA_exch_mo_prev + lmA_mo_prev
                            WB_exch_mo_prev = WB_exch_mo_prev + lmB_mo_prev
                    else:
                        # All blocks (S2)
                        WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_s2_total(h_sapt, 
                                                                                    ca= CA_prev, 
                                                                                    cb=CB_prev)
                        #  LM-terms (S2)
                        if lm_bool:
                            lmA_mo_prev, lmB_mo_prev = form_lm_terms_s2(h_sapt,
                                                                        ca= CA_prev,
                                                                        cb=CB_prev)
                            WA_exch_mo_prev= WA_exch_mo_prev + lmA_mo_prev
                            WB_exch_mo_prev = WB_exch_mo_prev + lmB_mo_prev

            # else, only OV-VO blocks of Omega-Exchange are present
            else:
                # Currently only have this scheme for OV-VO Case
                # ******************************************************
                # NOTE:Symmetrization Scheme--> <VP> for OV, <PV> for VO
                # ******************************************************
                if non_s2:
                    WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_sinf(h_sapt, ca= CA_prev, cb=CB_prev)
                else:
                    WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_s2(h_sapt, 
                                                                                  ca= CA_prev, 
                                                                                  cb=CB_prev)
                    if lm_bool:
                        pass
                        # NOTE: LM matrices do not have OV-VO option,
                        # TODO: Needs change in.......form_lm_terms_s2()
            
            # MO --> AO
            # Omega Exchange (AO)
            WA_exch_ao_prev = S.dot(CB_prev).dot(WA_exch_mo_prev).dot(CB_prev.T).dot(S)
            WB_exch_ao_prev = S.dot(CA_prev).dot(WB_exch_mo_prev).dot(CA_prev.T).dot(S)

        # No Omega-exchange will be added
        else:
            WA_exch_ao_prev = np.zeros((nmo_B, nmo_B))
            WB_exch_ao_prev = np.zeros((nmo_A, nmo_A)) 

        if i == 1:
            omega_exch_timer.stop()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        WA_tot_ao_prev = WA_elst_ao_prev + WA_exch_ao_prev
        WB_tot_ao_prev = WB_elst_ao_prev + WB_exch_ao_prev

        
        # NOTE: SCF iterations in Psi4
        #=======================================================  
        #           SCF Iterations(Monomer A)                  #
        #=======================================================
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

        pdb.run("do_scf_itr(monomer= monomerA, reference= 'RHF', guess= (CA_occ_prev, CA_occ_prev), omega= WB_tot_ao_prev, maxiter= 300)")

        try:      
            wfnA_scf_conv = do_scf_itr(monomer= monomerA,
                                        reference= 'RHF',
                                        guess= (CA_occ_prev, CA_occ_prev),
                                        omega= WB_tot_ao_prev,
                                        maxiter= 300
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
                                        reference= 'RHF',
                                        guess= (CB_occ_prev, CB_occ_prev),
                                        omega= WA_tot_ao_prev,
                                        maxiter= 300
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
        
        # E(electrostatic + pol)
        E_elst = get_elst(Ca=CA_occ_prev, 
                            Cb=CB_occ_prev,
                            I=I,
                            VA=VA, VB=VB,  
                            Enucint= Enucint)
        if i == 1:
            E_elst_timer.stop()
            E_exch_timer = sapt_timer('Exchange Energy calc')
        
        # E(exch), Non-S2
        if not omega_elst_only:
            E_exch = get_Exch_sinf(
                            sapt= h_sapt,
                            ca= CA_prev,
                            cb=CB_prev
                        )
        else:
            E_exch = 0  
        if i == 1:
            E_exch_timer.stop()    

        # E(LM), Non-S2
        E_lm = 0
        if lm_bool:
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
        E1 = E_elst + E_exch
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
            with open(no_pb_fname_itr, 'a') as f:
                        f.write(f"\n#Dimer :{dimer.name()}\n")
                        f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
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
        # if to_write_matrices:
        #     #============================================ 
        #     #           Added for Bug-testing           #           
        #     #============================================
        #     with open(no_pb_fname_matrices, 'a') as f:
        #         f.write(f"\n#Dimer :{dimer.name()}\n")
        #         f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
        #         f.write(f"\n#=============== itr:{i-1}")
        #         f.write(f"\n#def EA:{dEA}")
        #         f.write(f"\n#def EB:{dEB}")
        #         f.write(f"\n#E_elst:{E_elst}")
        #         f.write(f"\n#E_exch:{E_exch}")
        #         f.write(f"\n#E_int:{E_int}")

        #         # if write_itr_dict['fock']:

        #         f.write(f"\n#================= Fock (in MO)")
        #         f.write(f"\n#SCF EA:{Eprime_A}")
        #         f.write("\n#Fock(A),MO:\n")
        #         np.savetxt(f, FA_mo_prev, delimiter=',',
        #                     fmt='%f'
        #                     )
                
        #         f.write(f"\n#SCF EB:{Eprime_B}")
        #         f.write("\n#Fock(B),MO:\n")
        #         np.savetxt(f, FB_mo_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         f.write(f"\n#SCF EA:{Eprime_A}")
            
        #         f.write(f"\n#================= Fock (in AO)")                   
        #         f.write("\n#Density(A),AO:\n")
        #         np.savetxt(f, DA_prev, delimiter=',',
        #                     fmt='%f'
        #                     )
                
        #         f.write("\n#Fock(A),AO:\n")
        #         np.savetxt(f, FA_prev, delimiter=',',
        #                     fmt='%f'
        #                     )
                
        #         f.write(f"\n#SCF EB:{Eprime_B}")
        #         f.write("\n#Density(B),AO:\n")
        #         np.savetxt(f, DB_prev, delimiter=',',
        #                     fmt='%f'
        #                     )
        #         f.write("\n#Fock(B),AO:\n")
        #         np.savetxt(f, FB_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
                
        #         # ================ Omega(A)
        #         f.write(f"\n#Dimer :{dimer.name()}\n")
        #         f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
        #         f.write(f"\n#itr:{i-1}")

        #         f.write("\n#=================== Omega (in MO)====================\n")
        #         f.write("\n#Omega-elst(A),MO:\n")
        #         np.savetxt(f, WA_elst_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         f.write("\n#Omega-exch(A),MO:\n")
        #         np.savetxt(f, WA_exch_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         if lm_bool:
        #             f.write("\n#LM(A),MO:\n")
        #             np.savetxt(f, lmA_mo_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         f.write("\n#Omega-total(A),MO:\n")
        #         np.savetxt(f, WA_tot, delimiter=',', 
        #                     fmt='%f'
        #                     )
                
        #         f.write("\n#=================== Omega (in AO)====================\n")
                
        #         f.write("\n#Omega-total(A),AO:\n")
        #         np.savetxt(f, WA_ao_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
                
        #         # ================ Omega(B)
        #         f.write("\n#=================== Omega (in MO)====================\n")
        #         f.write("\n#Omega-elst(B),MO:\n")
        #         np.savetxt(f, WB_elst_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         f.write("\n#Omega-exch(B),MO:\n")
        #         np.savetxt(f, WB_exch_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         if lm_bool:
        #             f.write("\n#LM(B),MO:\n")
        #             np.savetxt(f, lmB_mo_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         f.write("\n#Omega-total(B),MO:\n")
        #         np.savetxt(f, WB_tot, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         f.write("\n#=================== Omega (in AO)====================\n")
                
        #         f.write("\n#Omega-total(B),AO:\n")
        #         np.savetxt(f, WB_ao_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         # ============= Modified Focks===============
        #         f.write(f"\n#========== Fock + Omega (in MO)============")
        #         f.write("\n#Fock(A) + Omega(B),MO:\n")
        #         np.savetxt(f, FAB_mo_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         f.write("\n#Fock(B) + Omega(A),MO:\n")
        #         np.savetxt(f, FBA_mo_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
                
        #         f.write(f"\n#========== Fock + Omega (in AO)============")
        #         f.write("\n#Fock(A) + Omega(B),AO:\n")
        #         np.savetxt(f, FAB_ao_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
        #         f.write("\n#Fock(B) + Omega(A),AO:\n")
        #         np.savetxt(f, FBA_ao_prev, delimiter=',', 
        #                     fmt='%f'
        #                     )
                
        #         f.write(f"\n#========== Energy Values ============")

        #         f.write('\nn\tm \tdEA\t\t dEB\t\t E_elst\t\t E_exch\t\t E_lm\t\t E_int\t\t E_int+lm\t\t dE\t\t dE_lm' )
        #         f.write(f'\n{(i-1):3d}\t{(i-1):3d}  {dEA:.8e}  {dEB:.8e}  {E_elst:.8e}  {E_exch:.8e}  {E_lm:.8e}  {E_int:.8e}  {E_int_w_lm:.8e} {(dE_int):.8e} {(dE_int_lm):.8e}')
        #######################################################################################################
        
        print(f'{(i-1):3d}\t{(i-1):3d}  {dEA:.8e}  {dEB:.8e}  {E_elst:.8e}  {E_exch:.8e} {E_lm:.8e} {E_int:.8e} {E_int_w_lm:.8e}  {(dE_int):.8e}  {(dE_int_lm):.8e}')
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
        if lm_bool:
            itr_dict.update({
                'E_FA': del_FA,
                'E_FB': del_FB,
                'E_WA': del_WA,
                'E_WB': del_WB,
            })
        itr_list.append(itr_dict)
        
        #================================================
        #          Write NO-PB Energy/Update Itr        #        
        #================================================
        # Update Iterations to CSV at every iteration dynamically
        if geom_index is not None:
            csv_fname = f'no_pb_itr_{dimer.name()}_{geom_index}.csv'
        else:
            csv_fname = f'no_pb_itr_{dimer.name()}.csv'
        write_itr_to_csv(data= itr_dict,
                         csv_fname= csv_fname)
        
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
        if (abs(dE_int_lm) < E_conv) and i>1 and not converged: 
            print('\n##############################################')
            print(f'!!!!       CONVERGED at itr  {i-1}         !!!!')
            print('################################################\n')
            no_pb_itr_timer.stop()

            converged = True
            CA_CONV = CA_i
            CB_CONV = CB_i

            # If convergence is reached, writes the conv orbitals
            # Also writes the conv itr dict to JSON
            # Writing Converged orbitals
            if to_write_orbitals:
                with open(no_pb_fname_conv, 'a') as f:
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
                    
            # NOTE:Commented out for single point calculation
            # ===============================================
            # JSON Data Handing
            # write_itr_to_json(dimer_name= dimer.name(),
            #                   key= 'conv',
            #                   itr_dict= itr_dict)
            
            # Breaks the loop in case convergence is reached!!
            # If not, continues to the following itr till maxiter is reached 

            return CA_CONV, CB_CONV, E_int, dEA, dEB, E1, del_FA, del_FB

        # Update for next iteration
        CA_prev = CA_i
        CB_prev = CB_i

        E_int_old = E_int
        E_int_old_w_lm = E_int_w_lm

        if i == 1:
            data_handle_timer.stop()

        # Writes the last orbital sets 
        # and return all iterations as a list
        if i == maxiter:  
            no_pb_itr_timer.stop()          
            # Writing last itr orbitals
            if to_write_orbitals:
                with open(no_pb_fname_end, 'a') as f:
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
            raise ConvergenceError(eqn_description='NO-PB Interaction',
                                   iteration=i,
                                   additional_info="Maximum number of NO-PB iterations exceeded.")        
            

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

    json_fname = 'no_pb_en.json'
    # Writing to json
    with open(json_fname, "w") as outfile:
        outfile.write(json_obj)

def write_geom_to_json(dimer_name,geometry_dict:dict):
    """
    Updates the geometry_dict w/ geometry info
    Appends geometry item to the dimer_list
    """
    json_fname = 'no_pb_en.json'
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
    with open("no_pb_en.json", "w") as outfile:
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

    jsonpath_str = f'no_pb_en.json'
    if pathlib.Path(jsonpath_str).exists():
        with open('no_pb_en.json', 'r') as file:
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
    with open("no_pb_en.json", "w") as outfile:
        outfile.write(json_obj) 

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
    print('entering humas code')
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
    print('GUESS are extracted...')
    wfn_ref_obj.guess_Ca(Ca_psi4_mat)
    wfn_ref_obj.guess_Cb(Cb_psi4_mat)
    print('GUESS are set...')

    # Initialize for SCF Run
    wfn_ref_obj.initialize()
    print('After initializing.....Check if it has the correct GUESS loaded')
    print(wfn_ref_obj.Ca().to_array())
    print(wfn_ref_obj.Cb().to_array())

    # Prepare the Omega matrix
    print('Omega to be added to Fock')
    print(omega)
    Omega_psi4_mat = psi4.core.Matrix.from_array(omega)
    wfn_ref_obj.push_back_external_potential(Omega_psi4_mat)

    # Start the SCF runs and save
    wfn_ref_obj.iterations()
    wfn_ref_obj.save_density_and_energy()

    print('After SCF iterations.....Modified')
    print(wfn_ref_obj.Ca().to_array())
    e_scfomega = wfn_ref_obj.compute_energy()
    print('Energy with the modified orbitals:....')
    print(e_scfomega)

    print('.... Finished scf-iterations')

    return wfn_ref_obj
