
"""
Utilies and Wrapper Methods
related to LM Energy and Potential in SQ
"""

import numpy as np
import opt_einsum as oe
import copy

import psi4
from utils.helper_SAPT import helper_SAPT
from utils.sinfinity import sinfinity
from utils.delta_utils import (
                form_l_terms_s2, form_l_terms_s4, 
                form_lm_terms_s2, form_lm_terms_s4
                )

#================================================================================
#		                    LM Energy                                           #
#================================================================================

def get_delta_W(sapt:helper_SAPT, ca=None, cb=None):
    """
    Delta-W Energy Terms:
    Expressions for S_infinity
    """
    
    if ca is not None and cb is not None:
        sapt.set_orbitals(ca=ca, cb=cb)

    # ================ Calculation for S-infinity =================
    sinf = sinfinity(sapt=sapt)
    # )
    # RHF Spin summed..........(S-inf)
    delta_wA = (
        2*oe.contract("aArR, ra, RA",sapt.v('aarr'), sinf.E_ra, sinf.E_ra)
         -oe.contract("aArR, Ra, rA",sapt.v('aarr'), sinf.E_ra, sinf.E_ra)
    )

    # RHF Spin summed.........(S-inf)
    delta_wB = (
        2*oe.contract("bBsS, sb, SB",sapt.v('bbss'), sinf.F_sb, sinf.F_sb)
         -oe.contract("bBsS, Sb, sB",sapt.v('bbss'), sinf.F_sb, sinf.F_sb)
    )

    # print('Delta W/Monomer A: Non-S2',delta_wA)
    # print('Delta W/Monomer B: Non-S2',delta_wB)
    return delta_wA, delta_wB

def get_delta_F(sapt:helper_SAPT, 
                ca=None, 
                cb=None):
    """
    Delta-F Energy Terms:
    Expressions for S_infinity
    """
    if ca is not None and cb is not None:
        sapt.set_orbitals(ca=ca, cb=cb)
        
    CA = copy.deepcopy(sapt.C_A)
    CB = copy.deepcopy(sapt.C_B)

    CA_occ = CA[:,:sapt.ndocc_A]
    CB_occ = CB[:,:sapt.ndocc_B]
    
    DA = oe.contract('pi,qi->pq', CA_occ, CA_occ)
    DB = oe.contract('pi,qi->pq', CB_occ, CB_occ)
    sapt_I = np.asarray(sapt.mints.ao_eri())

    JA = oe.contract('pqrs,rs->pq', sapt_I, DA)
    KA = oe.contract('prqs,rs->pq', sapt_I, DA)
    TA = np.asarray(psi4.core.MintsHelper(sapt.wfnA.basisset()).ao_kinetic())
    VA = sapt.V_A
    HA = TA + VA
    FA = HA + 2*JA - KA

    JB = oe.contract('pqrs,rs->pq', sapt_I, DB)
    KB = oe.contract('prqs,rs->pq', sapt_I, DB)
    TB = np.asarray(psi4.core.MintsHelper(sapt.wfnB.basisset()).ao_kinetic())
    VB = sapt.V_B
    HB = TB + VB
    FB = HB + 2*JB - KB
    # print(sapt.rhfA)
    # print(sapt.rhfB)

    FA_mo = CA.T.dot(FA).dot(CA)
    FB_mo = CB.T.dot(FB).dot(CB)
    
    # S2 Approximation (monomer A)
    delta_FA_s2 = -2*oe.contract('ar, ba, rb', 
                            FA_mo[:sapt.ndocc_A,sapt.ndocc_A:], 
                            sapt.s('ba'), 
                            sapt.s('rb'))
    
    # S2 Approximation (monomer B)
    delta_FB_s2 = -2*oe.contract('bs, ab, sa', 
                            FB_mo[:sapt.ndocc_B,sapt.ndocc_B:], 
                            sapt.s('ab'), 
                            sapt.s('sa'))
    
    # print('Delta F/Monomer A: S2', delta_FA_s2)
    # print('Delta F/Monomer B: S2', delta_FB_s2)   
    
    # ==========================================================
    # ================ Calculation for S-infinity ==============
    # Non-S2 Approximation (monomer A)
    sinf = sinfinity(sapt=sapt)    
    delta_FA = -2*oe.contract('ar, ra', 
                        FA_mo[:sapt.ndocc_A,sapt.ndocc_A:], 
                        sinf.E_ra)
    
    # Non-S2 Approximation (monomer B)
    delta_FB = -2*oe.contract('bs, sb', 
                        FB_mo[:sapt.ndocc_B,sapt.ndocc_B:], 
                        sinf.F_sb)

    # print('Delta F/Monomer A: Non-S2', delta_FA)
    # print('Delta F/Monomer B: Non-S2', delta_FB)
    return delta_FA, delta_FB


#================================================================================
#		                    LM Potential                                        #
#                   Main Methods for call in SART utils 	        			#
#================================================================================

def form_lm_terms_w_sym(sapt:helper_SAPT,ca=None, cb=None, s_option='S2'):
    """
    Forms total LM contribution, (F+W) part
    """    
    if s_option not in ['S2', 'S4']:
        raise RuntimeError(f'Not Recognised options')
    
    if s_option == 'S2':
        del_lmA, del_lmB = form_lm_terms_s2(sapt= sapt,
                                         ca= ca,
                                         cb= cb,
                                         sym_all= True
                                         )
        return del_lmA, del_lmB
    
    elif s_option == 'S4':
        del_lmA_s2, del_lmB_s2 = form_lm_terms_s2(sapt= sapt,
                                         ca= ca,
                                         cb= cb,
                                         sym_all= True
                                         )
        del_lmA_s4, del_lmB_s4 = form_lm_terms_s4(sapt= sapt,
                                         ca= ca,
                                         cb= cb,
                                         sym_all= True
                                         )
        del_lmA = del_lmA_s2 + del_lmA_s4
        del_lmB = del_lmB_s2 + del_lmB_s4
        
        return del_lmA, del_lmB 

def form_l_terms_w_sym(sapt:helper_SAPT,ca=None, cb=None, s_option='S2'):
    """
    Forms Fock contribution(Landshoff part) of LM potential
    """
    # from utils.delta_utils_sinf import form_l_terms_s2, form_l_terms_sinf
    if s_option not in ['S2', 'S4']:
        raise RuntimeError(f'Not Recognised options')

    if s_option == 'S2':
        del_LA, del_LB = form_l_terms_s2(sapt= sapt,
                                         ca= ca,
                                         cb= cb,
                                         )
        return del_LA, del_LB
    
    elif s_option == 'S4':
        del_LA_s2, del_LB_s2 = form_l_terms_s2(sapt= sapt,
                                         ca= ca,
                                         cb= cb,
                                         )
        del_LA_s4, del_LB_s4 = form_l_terms_s4(sapt= sapt,
                                         ca= ca,
                                         cb= cb)  
        
        del_LA = del_LA_s2 + del_LA_s4
        del_LB = del_LB_s2 + del_LB_s4  
        
        return del_LA, del_LB
        
    elif s_option == 'Sinf':
        # del_LA, del_LB = form_l_terms_sinf(sapt= sapt,
        #                                  ca= ca,
        #                                  cb= cb,
        #                                  overlap_in_diag= 'Sinf'
        #                                  )
        # return del_LA, del_LB
        pass
    
def form_m_terms_w_sym(sapt:helper_SAPT,ca=None, cb=None, s_option='S2'):
    """
    Forms W contribution(Murell part) of LM potential
    """
    
    if s_option == 'S2':
        lmA_mo_s2, lmB_mo_s2 = form_lm_terms_s2(sapt=sapt,
                                                ca= ca,
                                                cb= cb,
                                                sym_all=True)

        lA_mo_s2, lB_mo_s2 = form_l_terms_s2(sapt= sapt,
                                                ca= ca,
                                                cb= cb
                                                )
        mA_mo = lmA_mo_s2 - lA_mo_s2
        mB_mo = lmB_mo_s2 - lB_mo_s2
        
        return mA_mo, mB_mo
    
    elif s_option == 'S4':
        lmA_mo_s2, lmB_mo_s2 = form_lm_terms_s2(sapt=sapt,
                                                ca= ca,
                                                cb= cb,
                                                sym_all=True)
        
        lmA_mo_s4, lmB_mo_s4 = form_lm_terms_s4(sapt=sapt,
                                                ca= ca,
                                                cb= cb,
                                                sym_all=True)
        
        lA_mo_s2, lB_mo_s2 = form_l_terms_s2(sapt=sapt,
                                                ca= ca,
                                                cb= cb,
                                        )
        
        lA_mo_s4, lB_mo_s4 = form_l_terms_s4(sapt=sapt,
                                                ca= ca,
                                                cb= cb,
                                                )

        mA_mo_s2 = lmA_mo_s2 - lA_mo_s2
        mB_mo_s2 = lmB_mo_s2 - lB_mo_s2
        
        mA_mo_s4 = lmA_mo_s4 - lA_mo_s4
        mB_mo_s4 = lmB_mo_s4 - lB_mo_s4
        
        mA_mo = mA_mo_s2 + mA_mo_s4
        mB_mo = mB_mo_s2 + mB_mo_s4
        
        return mA_mo, mB_mo
