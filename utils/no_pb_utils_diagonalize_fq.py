"""
Utilities related to NO-PB iterations, Convergence
Edo: This is the first quantization draft, based on the second quantization routines.

**** Modified to store Matrices for NO-PB itr if needed
**** Modified to consider only omega_elst/exch boolean options
**** Exchange(OV-VO) default is Non-S2, Other blocks in S2

**** Modified to consider h_sapt object to access monomer info 
##############################################################

**** Modified to have the OMEGA-SYMMETRIZED scheme
"""

import numpy as np
import opt_einsum as oe
import copy
import psi4
from psi4.driver.p4util.exceptions import ConvergenceError

from utils.helper_SAPT import helper_SAPT, sapt_timer
from utils.omega_exch_utils_test import (form_omega_exchange_s2_total, 
                                         form_omega_exchange_sinf_total,
                                         form_omega_exchange_s2_sym,
                                         form_omega_exchange_sinf_sym,
                                         form_omega_exchange_s2, 
                                         form_lm_terms_s2,
                                        get_Exch_s2, 
                                        form_omega_exchange_sinf, 
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
                diis_bool:bool=False,
                to_write_itr:bool = False,
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
    # This needs editing in a second moment.
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

    if lm_bool:
        print('\n ######### LM terms are added to the Omega potential.')
    else:
        print('\n ######### No LM terms are added to the Omega potential.')       

    if diis_bool:
        print('\n========= With DIIS')
    else:
        print('\n========= Without DIIS')  

    access_var_timer.stop()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ======================= Printing calculation specifics DONE ======================
        
    print('\n############### Storing of Wavefunction variables started!')

    read_hsapt_timer = sapt_timer('Read helperSAPT object')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    h_sapt = sapt
    rhfA = h_sapt.rhfA
    rhfB = h_sapt.rhfB
    print('Monomer Energies for A and B:')
    print(rhfA, rhfB)

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
    # this might not be necessary:
    A = mintsA.ao_overlap()
    A.power(-0.5, 1.e-16)
    A = np.asarray(A)

    S = h_sapt.S
    # This appears to be defined in A.O. :
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

    if diis_bool:
        diisA = DIIS_helper()
        diisB = DIIS_helper()

    En_dict = {}
    itr_list = []

    print('n\tm \tdEA\t\t dEB\t\t E_elst\t\t E_exch\t\t E_lm\t\t E_int\t\t E_int+lm\t\t dE\t\t dE(lm)' )
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    if to_write_itr:
        if geom_index is not None:
            no_pb_fname_matrices = f'no_pb_matrices_{dimer.name()}_{geom_index}.txt'
        else:
            no_pb_fname_matrices = f'no_pb_matrices_{dimer.name()}.txt'

        with open(no_pb_fname_matrices, 'w') as f:
            f.write(f"\n#Dimer :{dimer.name()}\n")
            f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
            f.write(f"\n#Energy(Monomer A) :{rhfA}\n")
            f.write(f"\n#Energy(Monomer B) :{rhfB}\n")

    if to_write_orbitals:
        if geom_index is not None:            
            no_pb_fname_zero = f'no_pb_orbitals_{dimer.name()}_{geom_index}_0.txt'
            no_pb_fname_conv = f'no_pb_orbitals_{dimer.name()}_{geom_index}_conv.txt'
            no_pb_fname_end = f'no_pb_orbitals_{dimer.name()}_{geom_index}_last.txt'
            
        else:
            no_pb_fname_zero = f'no_pb_orbitals_{dimer.name()}_0.txt'
            no_pb_fname_conv = f'no_pb_orbitals_{dimer.name()}_conv.txt'
            no_pb_fname_end = f'no_pb_orbitals_{dimer.name()}_last.txt'
            
        # NOTE: Commented--> Orbitals to be written for zeroth as well as following iterations, 
        # though filename is with suffix '0'
        # # Writing zeroth order orbitals   
           
        # with open(no_pb_fname_zero, 'a') as f:
        #             f.write(f"\n#Dimer :{dimer.name()}\n")
        #             f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
        #             f.write(f"\n#itr:{0}")
        #             f.write("\n#CA Orbitals(isolated):\n")
        #             np.savetxt(f, CA_prev, delimiter=','
        #                     )
        #             f.write("\n#CB Orbitals(isolated):\n")
        #             np.savetxt(f, CB_prev, delimiter=',', 
        #                     )

    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #                             NO-PB Iteration STARTS HERE!!!!                      #
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    no_pb_itr_timer = sapt_timer('NO-PB iterations')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(1, maxiter + 1):
        if i == 1:
            fock_timer = sapt_timer('Constructing Fock Matrix in AO')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # C_{\alpha a} \alpha = a.o. a = occupied m.o.
        CA_occ_prev = CA_prev[:,:ndocc_A]
        # C_{\beta b} (but who actually knows...)
        CB_occ_prev = CB_prev[:,:ndocc_B]
        # \gamma_{\alpha \alpha'}
        DA_prev = oe.contract('pi,qi->pq', CA_occ_prev, CA_occ_prev)
        # \gamma_{\beta \beta'}
        DB_prev = oe.contract('pi,qi->pq', CB_occ_prev, CB_occ_prev)

        # She is transformig everything to the A.O. basis.
        # F_{\alpha \alpha'} = h_{\alpha \alpha'} +  \sum_{\alpha'' \alpha'''} P_{\alpha'' \alpha'''} (\alpha \alpha' || \alpha'' \alpha''')
        # JA_{\mu\nu} = \sum_{\alpha \alpha'} P_{\alpha \alpha'} (\mu \nu | \alpha \alpha')
        JA_prev = oe.contract('pqrs,rs->pq', I, DA_prev) 
        KA_prev = oe.contract('prqs,rs->pq', I, DA_prev)
        FA_prev = HA + 2*JA_prev - KA_prev

        JB_prev = oe.contract('pqrs,rs->pq', I, DB_prev)
        KB_prev = oe.contract('prqs,rs->pq', I, DB_prev)
        FB_prev = HB + 2*JB_prev - KB_prev

        if i == 1:
            fock_timer.stop()
            omega_elst_timer = sapt_timer('Constructing Omega Elst in MO')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #============== Omega(Electrostatic)=============================================    
        # NOTE: In MO: Initial Omega matrices for A and B
        # C_B^T V_A C_B ==> V_A is what we called U_A.
        # \sum_{\beta \beta'} B_{b \beta}^T VA_{\beta \beta'} B_{\beta' b'}
        # This \beta indices are actually indices of the dimer centered basis set. You end up with a Matrix 
        # that depends only on occupied orbitals of B, but is optained by contracting over all atomic orbitals.
        # In my case, with a monomer centered basis set, I need to compute VA_{\beta \alpha}. We'll see...
    
        #VA_mo_prev = CB_prev.T.dot(VA).dot(CB_prev) # This needs to be changed to CB^T VA CA and doubled, since for B the equations are different:
        VAA_mo_prev = CB_prev.T.dot(VA).dot(CA_prev) # UA for A
        VAB_mo_prev = CA_prev.T.dot(VA).dot(CB_prev) # UA for B
        # \sum_{a a'}(b b' | a a')
        # C_{b \beta}^T JA_prev_{\beta \beta'} C_{\beta' b'} 
        # JA0_mo_prev = 2*CB_prev.T.dot(JA_prev).dot(CB_prev) # Same as above. We will keep the old code and update:
        JAA0_mo_prev = 2*CB_prev.T.dot(JA_prev).dot(CA_prev) # JA for A
        JAB0_mo_prev = 2*CA_prev.T.dot(JA_prev).dot(CB_prev) # JA for B
        WAA_elst_prev = VAA_mo_prev + JAA0_mo_prev 
        WAB_elst_prev = VAB_mo_prev + JAB0_mo_prev 

        #VB_mo_prev = CA_prev.T.dot(VB).dot(CA_prev)
        VBA_mo_prev = CB_prev.T.dot(VB).dot(CA_prev) # UB for A
        VBB_mo_prev = CA_prev.T.dot(VB).dot(CB_prev) # UB for B
        #JB0_mo_prev = 2*CA_prev.T.dot(JB_prev).dot(CA_prev) 
        JBA0_mo_prev = 2*CB_prev.T.dot(JB_prev).dot(CA_prev) # JB for A 
        JBB0_mo_prev = 2*CA_prev.T.dot(JB_prev).dot(CB_prev) # JB for B
        WBA_elst_prev = VBA_mo_prev + JBA0_mo_prev
        WBB_elst_prev = VBB_mo_prev + JBB0_mo_prev
        
        # This part right here can probably be eliminated:
        #if omega_elst_all_blocks:
        #        WA_elst_prev = WA0_elst_mo_prev
        #        WB_elst_prev = WB0_elst_mo_prev
        #else:
        #    # ================ OV-VO block of Omega(electrostactic)==========================     
        #    WA_elst_mo_prev = copy.deepcopy(WA0_elst_mo_prev)
        #    WB_elst_mo_prev = copy.deepcopy(WB0_elst_mo_prev)  

        #    # # VV block
        #    WB_elst_mo_prev[ndocc_A:, ndocc_A:] = np.zeros((nmo_A- ndocc_A, nmo_A- ndocc_A))
        #    ## OO block
        #    WB_elst_mo_prev[:ndocc_A, :ndocc_A] = np.zeros((ndocc_A, ndocc_A))

        #    # VV block
        #    WA_elst_mo_prev[ndocc_B:, ndocc_B:] = np.zeros((nmo_B- ndocc_B, nmo_B- ndocc_B))
        #    ## OO block
        #    WA_elst_mo_prev[:ndocc_B, :ndocc_B] = np.zeros((ndocc_B, ndocc_B))

        #    WA_elst_prev = WA_elst_mo_prev
        #    WB_elst_prev = WB_elst_mo_prev

        #if i ==1:
        #    omega_elst_timer.stop()
        #    omega_exch_timer = sapt_timer('Constructing Omega-Exch in MO, also LM Terms')
        #    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if not omega_elst_only:  
            #============== Omega (Exchange)================
            if omega_exch_all_blocks:

                # *******************************************
                # NOTE:Symmetrization Scheme--> 1/2(PV +VP)
                # *******************************************
                if sym_all:
                    if non_s2:
                        # Partial Non-S2 : 
                        # OV-VO(Non-S2), OO-VV(S2)
                        # This is where you need to modify the form omega exchange.
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
                    if non_s2:
                        WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_sinf_total(h_sapt, 
                                                                                    ca= CA_prev, 
                                                                                    cb=CB_prev)
                        if lm_bool:
                            lmA_mo_prev, lmB_mo_prev = form_lm_terms_s2(h_sapt,
                                                                        ca= CA_prev,
                                                                        cb=CB_prev)
                            WA_exch_mo_prev= WA_exch_mo_prev + lmA_mo_prev
                            WB_exch_mo_prev = WB_exch_mo_prev + lmB_mo_prev
                    else:
                        WA_exch_mo_prev, WB_exch_mo_prev = form_omega_exchange_s2_total(h_sapt, 
                                                                                    ca= CA_prev, 
                                                                                    cb=CB_prev)
                        if lm_bool:
                            lmA_mo_prev, lmB_mo_prev = form_lm_terms_s2(h_sapt,ca= CA_prev,cb=CB_prev)
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
                        # Needs change in.......form_lm_terms_s2()
            WA_exch_prev = WA_exch_mo_prev
            WB_exch_prev = WB_exch_mo_prev

        else:
            WA_exch_prev = np.zeros((nmo_B, nmo_B))
            WB_exch_prev = np.zeros((nmo_A, nmo_A)) 

        if i == 1:
            omega_exch_timer.stop()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble the omegas (or interaction fock operators).
        WAA_tot_0 = WAA_elst_prev + WA_exch_prev # + WAA_exch_prev
        WAB_tot_0 = WAB_elst_prev + WA_exch_prev # + WAB_exch_prev
        WBA_tot_0 = WBA_elst_prev + WB_exch_prev # + WBA_exch_prev
        WBB_tot_0 = WBB_elst_prev + WB_exch_prev # + WBB_exch_prev
        # Now add the omegas convert them in AA or BB sets:
        WA_tot_0 = WBA_tot_0 - WAA_tot_0
        WB_tot_0 = WAB_tot_0 - WBB_tot_0
        WA_tot = S.dot(B).dot(B).T.dot(WA_tot_0)
        ################ Omegas in AO 
        # We don't need this:
        #if i == 1:
        #    omega_ao_transform_timer = sapt_timer('MO-->AO Transformation of Omega total')       
        #WA_ao_prev = S.dot(CB_prev).dot(WA_tot).dot(CB_prev.T).dot(S)
        #WB_ao_prev = S.dot(CA_prev).dot(WB_tot).dot(CA_prev.T).dot(S)
        #if i == 1:
        #    omega_ao_transform_timer.stop()

        ################ Perturbed Fock Matrices in MO
        if i == 1:
            fock_mo_transform_timer = sapt_timer('AO --> MO, Fock + Omega')
        FA_mo_prev = CA_prev.T.dot(FA_prev).dot(CA_prev)
        FB_mo_prev = CB_prev.T.dot(FB_prev).dot(CB_prev)
        # Instead we need this transformation:
        WB_tot = WBA_tot - WAA_tot        


        FAB_mo_prev = FA_mo_prev + WB_tot 
        FBA_mo_prev = FB_mo_prev + WA_tot  
        if i == 1:
            fock_mo_transform_timer.stop()      

        ################ Perturbed Fock Matrices in AO
        # How the heck does this works mathematically???
        # This cannot be right with non orthogonal orbitals!!!!! This is why:
        # In standard HF F^{MO} = C^T F^{AO} C. So to get F^{AO} we compute:
        # F^{AO} = C^{-T} F^{MO} C^{-1}
        # With orthogonal m.o. orbitals we have that C^{T} \Sigma C = 1 so we have a simple expression
        # for C^{-T} and C^{-1}: C^{-T} = \Sigma C and C^{-1} = C^{T} \Sigma and we get the formula implemented beneath.
        # So this would be true if S is the overlap matrix of only A or Only B. It doesn't make any sense if S is the
        # total overlap matrix in AO (\Sigma above). But it's also true that orbitals of A are orthogonal
        # with respect to each other, and if they are constructed using a dimer centered basis set,
        # then CA^{T}SCA = 1 must still be true. 
        if i == 1:
            fock_ao_transform_timer = sapt_timer('MO --> AO, Fock + Omega')
        FAB_ao_prev = S.dot(CA_prev).dot(FAB_mo_prev).dot(CA_prev.T).dot(S)
        FBA_ao_prev = S.dot(CB_prev).dot(FBA_mo_prev).dot(CB_prev.T).dot(S)
        if i == 1:
            fock_ao_transform_timer.stop()

        #===================== DIIS in macro-itr ============================
        
        diis_timer = sapt_timer('DIIS for macro-iteration, Fock +Omega in AO')
        if diis_bool:
    
            # DIIS error build w/ HF analytic gradient 
            diis_eA = oe.contract('ij,jk,kl->il', FAB_ao_prev, DA_prev, S) - oe.contract('ij,jk,kl->il', S, DA_prev, FAB_ao_prev)
            diis_eA = A.dot(diis_eA).dot(A)

            diisA.add(state= FAB_ao_prev,
                    error= diis_eA)
            dRMSA = np.mean(diis_eA**2)**0.5

            diis_eB = oe.contract('ij,jk,kl->il', FBA_ao_prev, DB_prev, S) - oe.contract('ij,jk,kl->il', S, DB_prev, FBA_ao_prev)
            diis_eB = A.dot(diis_eB).dot(A)

            diisB.add(state= FBA_ao_prev,
                    error= diis_eB)
            dRMSB = np.mean(diis_eB**2)**0.5

            if i >=2:
                FAB_ao_prev = diisA.extrapolate()
                FBA_ao_prev = diisB.extrapolate()
        diis_timer.stop()

        #============================= Diagonalise the Fock matrix =======================
        #TODO: Add micro-iterations for these SCF cycles

        diag_A_timer = sapt_timer('Diagonalizing Modified Fock A')
        # Modified C coefficient matrix for Monomer A
        CA_i = diagonalise_fock(f= FAB_ao_prev,
                                orthogonalizer= A,
                                ) 
        diag_A_timer.stop()

        diag_B_timer = sapt_timer('Diagonalizing Modified Fock A')
        # Modified C coefficient matrix for Monomer B
        CB_i = diagonalise_fock(f= FBA_ao_prev,
                                orthogonalizer= A
                                ) 
        diag_B_timer.stop()

        if i == 1:
            energy_calc_timer = sapt_timer('NO-PB Energy Calculation')

        # Energy Calculation
        # ------------------itr i-1
        if i == 1:
            E_elst_timer = sapt_timer('Electrostatic Energy calc')    
        E_elst = get_elst(Ca=CA_occ_prev, 
                            Cb=CB_occ_prev,
                            I=I,
                            VA=VA, VB=VB,  
                            Enucint= Enucint)
        if i == 1:
            E_elst_timer.stop()
            E_exch_timer = sapt_timer('Exchange Energy calc')
        
        if not omega_elst_only:
            if non_s2:
                E_exch = get_Exch_sinf(
                                sapt= h_sapt,
                                ca= CA_prev,
                                cb=CB_prev
            )
            else:
                E_exch = get_Exch_sinf(
                                sapt= h_sapt,
                                ca= CA_prev,
                                cb=CB_prev
            )
        else:
            E_exch = 0  
        if i == 1:
            E_exch_timer.stop()
            

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

        Eprime_A = oe.contract('pq,pq->', FA_prev + HA, DA_prev) + EnucA
        Eprime_B = oe.contract('pq,pq->', FB_prev + HB, DB_prev) + EnucB
        dEA = (Eprime_A - rhfA)
        dEB = (Eprime_B - rhfB)

        if i == 1:
            E_def_timer.stop()

        # Energy Terms 
        # =================================
        E_int = dEA + dEB + E_elst + E_exch
        E_int_w_lm = E_int + E_lm
        dE_int = E_int - E_int_old
        dE_int_lm = E_int_w_lm - E_int_old_w_lm
        if i ==1:
            energy_calc_timer.stop()

        if to_write_orbitals:
            # Writing zeroth order orbitals  and each iteration    
            with open(no_pb_fname_zero, 'a') as f:
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
        if to_write_itr:
            #NOTE ================= Added for Bug-testing =========================================
            with open(no_pb_fname_matrices, 'a') as f:
                f.write(f"\n#Dimer :{dimer.name()}\n")
                f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
                f.write(f"\n#=============== itr:{i-1}")
                f.write(f"\n#def EA:{dEA}")
                f.write(f"\n#def EB:{dEB}")
                f.write(f"\n#E_elst:{E_elst}")
                f.write(f"\n#E_exch:{E_exch}")
                f.write(f"\n#E_int:{E_int}")

                f.write(f"\n#================= Fock (in MO)")
                f.write(f"\n#SCF EA:{Eprime_A}")
                f.write("\n#Fock(A),MO:\n")
                np.savetxt(f, FA_mo_prev, delimiter=',',
                            fmt='%f'
                            )
                
                f.write(f"\n#SCF EB:{Eprime_B}")
                f.write("\n#Fock(B),MO:\n")
                np.savetxt(f, FB_mo_prev, delimiter=',', 
                            fmt='%f'
                            )
                f.write(f"\n#SCF EA:{Eprime_A}")
                
                f.write(f"\n#================= Fock (in AO)")

                f.write("\n#Density(A),AO:\n")
                np.savetxt(f, DA_prev, delimiter=',',
                            fmt='%f'
                            )
                
                f.write("\n#Fock(A),AO:\n")
                np.savetxt(f, FA_prev, delimiter=',',
                            fmt='%f'
                            )
                
                f.write(f"\n#SCF EB:{Eprime_B}")
                f.write("\n#Density(B),AO:\n")
                np.savetxt(f, DB_prev, delimiter=',',
                            fmt='%f'
                            )
                f.write("\n#Fock(B),AO:\n")
                np.savetxt(f, FB_prev, delimiter=',', 
                            fmt='%f'
                            )
                
                # ================ Omega(A)
                f.write(f"\n#Dimer :{dimer.name()}\n")
                f.write(f"\n#Basis set:{psi4.core.get_option('SCF', 'basis')}")
                f.write(f"\n#itr:{i-1}")

                f.write("\n#=================== Omega (in MO)====================\n")
                f.write("\n#Omega-elst(A),MO:\n")
                np.savetxt(f, WA_elst_prev, delimiter=',', 
                            fmt='%f'
                            )
                f.write("\n#Omega-exch(A),MO:\n")
                np.savetxt(f, WA_exch_prev, delimiter=',', 
                            fmt='%f'
                            )
                if lm_bool:
                    f.write("\n#LM(A),MO:\n")
                    np.savetxt(f, lmA_mo_prev, delimiter=',', 
                            fmt='%f'
                            )
                f.write("\n#Omega-total(A),MO:\n")
                np.savetxt(f, WA_tot, delimiter=',', 
                            fmt='%f'
                            )
                
                f.write("\n#=================== Omega (in AO)====================\n")
                
                f.write("\n#Omega-total(A),AO:\n")
                np.savetxt(f, WA_ao_prev, delimiter=',', 
                            fmt='%f'
                            )
                
                # ================ Omega(B)
                f.write("\n#=================== Omega (in MO)====================\n")
                f.write("\n#Omega-elst(B),MO:\n")
                np.savetxt(f, WB_elst_prev, delimiter=',', 
                            fmt='%f'
                            )
                f.write("\n#Omega-exch(B),MO:\n")
                np.savetxt(f, WB_exch_prev, delimiter=',', 
                            fmt='%f'
                            )
                if lm_bool:
                    f.write("\n#LM(B),MO:\n")
                    np.savetxt(f, lmB_mo_prev, delimiter=',', 
                            fmt='%f'
                            )
                f.write("\n#Omega-total(B),MO:\n")
                np.savetxt(f, WB_tot, delimiter=',', 
                            fmt='%f'
                            )
                f.write("\n#=================== Omega (in AO)====================\n")
                
                f.write("\n#Omega-total(B),AO:\n")
                np.savetxt(f, WB_ao_prev, delimiter=',', 
                            fmt='%f'
                            )
                # ============= Modified Focks===============
                f.write(f"\n#========== Fock + Omega (in MO)============")
                f.write("\n#Fock(A) + Omega(B),MO:\n")
                np.savetxt(f, FAB_mo_prev, delimiter=',', 
                            fmt='%f'
                            )
                f.write("\n#Fock(B) + Omega(A),MO:\n")
                np.savetxt(f, FBA_mo_prev, delimiter=',', 
                            fmt='%f'
                            )
                
                f.write(f"\n#========== Fock + Omega (in AO)============")
                f.write("\n#Fock(A) + Omega(B),AO:\n")
                np.savetxt(f, FAB_ao_prev, delimiter=',', 
                            fmt='%f'
                            )
                f.write("\n#Fock(B) + Omega(A),AO:\n")
                np.savetxt(f, FBA_ao_prev, delimiter=',', 
                            fmt='%f'
                            )
                
                f.write(f"\n#========== Energy Values ============")

                f.write('\nn\tm \tdEA\t\t dEB\t\t E_elst\t\t E_exch\t\t E_lm\t\t E_int\t\t E_int+lm\t\t dE\t\t dE_lm' )
                f.write(f'\n{(i-1):3d}\t{(i-1):3d}  {dEA:.8e}  {dEB:.8e}  {E_elst:.8e}  {E_exch:.8e}  {E_lm:.8e}  {E_int:.8e}  {E_int_w_lm:.8e} {(dE_int):.8e} {(dE_int_lm):.8e}')
        #======================================================================================================================================
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
        
        # =========================================================
        # Zeroth itr 
        # (with unperturbed monomer orbitals)
        # =========================================================
        if i == 1:
            En_dict.update({'itr 0': itr_dict})

            # NOTE:Commented out for single point calculation
            # ===============================================
            # JSON Data Handing
            # write_itr_to_json(dimer_name= dimer.name(),
            #                   key= 'zeroth',
            #                 itr_dict= itr_dict)
            
        #============================= Check for convergence criteria
        # Converged itr 
        # (with converged monomer orbitals)
        # (i>1) condition is added to avoid convergence on zeroth itr
        # if (abs(E_int- E_int_old) < E_conv) and i>1:  
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

            conv_dict = copy.deepcopy(itr_dict)
            conv_dict.update({'itr': i-1,})
            En_dict.update({f'conv': conv_dict})
            
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
            # # JSON Data Handing
            # write_itr_to_json(dimer_name= dimer.name(),
            #                   key= 'conv',
            #                   itr_dict= conv_dict)
        #     return En_dict

        E_int_old = E_int
        E_int_old_w_lm = E_int_w_lm

        # Update for next iteration
        CA_prev = CA_i
        CB_prev = CB_i

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
            # NOTE: Does not return Convergence Error for NO-PB iteartions
            # Returns the itr_list at the end...
            # TODO: Add argument to handle this
            
            # raise ConvergenceError(eqn_description='NO-PB Interaction',
            #                        iteration=i,
            #                        additional_info="Maximum number of SCF cycles exceeded.")        
            return itr_list

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