
import numpy as np
import opt_einsum as oe
import copy
import pprint

import psi4
from utils.helper_SAPT import helper_SAPT
from utils.sinfinity import sinfinity

def get_omega_delta_F_s2(sapt:helper_SAPT,ca=None, cb=None):
    """
    Explicitly calculates the Fock contribution to the delta_F terms using the sapt.fock() method
    """
    if ca is not None and cb is not None:
        sapt.set_orbitals(ca=ca, cb=cb)

    # ================          Fock contribution, Omega(A)         ==============
    # OO Block
    termA_OO_FA_P = 0.5*( 
            -2 * oe.contract("ar,rp,Ba->pB", sapt.fock('A', 'ar'), sapt.s('rb'), sapt.s('ba')))    
    termA_OO_FB_P =0.5*( 
            -2 * oe.contract("Bs,ap,sa->pB", sapt.fock('B', 'bs'), sapt.s('ab'), sapt.s('sa')) 
            +2 * oe.contract("bp,Ba,ab->pB", sapt.fock('B', 'bb'), sapt.s('ba'), sapt.s('ab')))
    
    # OV Block
    termA_OV_FA_P = 0.5*( 
            -2 * oe.contract("ar,rS,Ba->BS", sapt.fock('A', 'ar'), sapt.s('rs'), sapt.s('ba')))    
    termA_OV_FB_P = 0.5*( 
            -2 * oe.contract("Bs,aS,sa->BS", sapt.fock('B', 'bs'), sapt.s('as'), sapt.s('sa')) 
            +2 * oe.contract("bS,Ba,ab->BS", sapt.fock('B', 'bs'), sapt.s('ba'), sapt.s('ab')))    
    termA_OV_P_FA = 0.5*( 
            -2 * oe.contract("ra,aS,Br->BS", sapt.fock('A', 'ra'), sapt.s('as'), sapt.s('br')))
    termA_OV_P_FB = 0.5*( 
            -2 * oe.contract("sS,as,Ba->BS", sapt.fock('B', 'ss'), sapt.s('as'), sapt.s('ba'))
            +2 * oe.contract("Bb,aS,ba->BS", sapt.fock('B', 'bb'), sapt.s('as'), sapt.s('ba')))

    # VO Block
    termA_VO_FA_P = 0.5*( 
            -2 * oe.contract("ar,rB,Sa->SB", sapt.fock('A', 'ar'), sapt.s('rb'), sapt.s('sa'))) 
    termA_VO_FB_P = 0.5*( 
            -2 * oe.contract("Ss,aB,sa->SB", sapt.fock('B', 'ss'), sapt.s('ab'), sapt.s('sa'))   
            +2 * oe.contract("bB,Sa,ab->SB", sapt.fock('B', 'bb'), sapt.s('sa'), sapt.s('ab')) )
    termA_VO_P_FA = 0.5*( 
            -2 * oe.contract("ra,aB,Sr->SB", sapt.fock('A', 'ra'), sapt.s('ab'), sapt.s('sr')))
    termA_VO_P_FB = 0.5*( 
            -2 * oe.contract("sB,as,Sa->SB", sapt.fock('B', 'sb'), sapt.s('as'), sapt.s('sa')) 
            +2 * oe.contract("Sb,aB,ba->SB", sapt.fock('B', 'sb'), sapt.s('ab'), sapt.s('ba'))) 

    # VV Block
    termA_VV_FA_P =0.5*( 
            -2 * oe.contract("ar,rq,Sa->qS", sapt.fock('A', 'ar'), sapt.s('rs'), sapt.s('sa')))    
    termA_VV_FB_P =0.5*( 
            -2 * oe.contract("Ss,aq,sa->qS", sapt.fock('B', 'ss'), sapt.s('as'), sapt.s('sa')) 
            +2 * oe.contract("bq,Sa,ab->qS", sapt.fock('B', 'bs'), sapt.s('sa'), sapt.s('ab'))) 
    
    # ================          Fock contribution, Omega(B)         ==============
    # OO Block
    termB_OO_FA_P =0.5*(
            -2 * oe.contract("Ar,bp,rb->pA", sapt.fock('A', 'ar'), sapt.s('ba'), sapt.s('rb')) 
            +2 * oe.contract("ap,ba,Ab->pA", sapt.fock('A', 'aa'), sapt.s('ba'), sapt.s('ab')))    
    termB_OO_FB_P =0.5*(
            -2 * oe.contract("bs,sp,Ab->pA", sapt.fock('B', 'bs'), sapt.s('sa'), sapt.s('ab')))
    
    # OV Block
    termB_OV_FA_P =0.5*( 
            -2 * oe.contract("Ar,bR,rb->AR", sapt.fock('A', 'ar'), sapt.s('br'), sapt.s('rb')) 
            +2 * oe.contract("aR,ba,Ab->AR", sapt.fock('A', 'ar'), sapt.s('ba'), sapt.s('ab')))    
    termB_OV_FB_P =0.5*(
            -2 * oe.contract("bs,sR,Ab->AR", sapt.fock('B', 'bs'), sapt.s('sr'), sapt.s('ab')))
    termB_OV_P_FA = 0.5*(-2 * oe.contract("rR,br,Ab->AR", sapt.fock('A', 'rr'), sapt.s('br'), sapt.s('ab'))
            +2 * oe.contract("Aa,bR,ab->AR", sapt.fock('A', 'aa'), sapt.s('br'), sapt.s('ab')))    
    termB_OV_P_FB = 0.5*(
            -2 * oe.contract("sb,bR,As->AR", sapt.fock('B', 'sb'), sapt.s('br'), sapt.s('as')))
    
    # VO Block
    termB_VO_FA_P = 0.5*(
            -2 * oe.contract("Rr,bA,rb->RA", sapt.fock('A', 'rr'), sapt.s('ba'), sapt.s('rb')) 
            +2 * oe.contract("aA,ba,Rb->RA", sapt.fock('A', 'aa'), sapt.s('ba'), sapt.s('rb')))
    termB_VO_FB_P = 0.5*(
            -2 * oe.contract("bs,sA,Rb->RA", sapt.fock('B', 'bs'), sapt.s('sa'), sapt.s('rb')))
    termB_VO_P_FA = 0.5*(
            -2 * oe.contract("rA,br,Rb->RA", sapt.fock('A', 'ra'), sapt.s('br'), sapt.s('rb')) 
            +2 * oe.contract("Ra,bA,ab->RA", sapt.fock('A', 'ra'), sapt.s('ba'), sapt.s('ab')))
    termB_VO_P_FB = 0.5*(
            -2 * oe.contract("sb,bA,Rs->RA", sapt.fock('B', 'sb'), sapt.s('ba'), sapt.s('rs')))
    
    # VV Block
    termB_VV_FA_P =0.5*(
            -2 * oe.contract("Rr,bq,rb->qR", sapt.fock('A', 'rr'), sapt.s('br'), sapt.s('rb')) 
            +2 * oe.contract("aq,ba,Rb->qR", sapt.fock('A', 'ar'), sapt.s('ba'), sapt.s('rb')))
    
    termB_VV_FB_P =0.5*(
            -2 * oe.contract("bs,sq,Rb->qR", sapt.fock('B', 'bs'), sapt.s('sr'), sapt.s('rb'))) 
    
    #====================================
    # Landshoff Blocks (Monomer A)				#
    #====================================
    LA_OO_HP_AB = (termA_OO_FA_P + termA_OO_FB_P) 
    LA_VV_HP_AB = (termA_VV_FA_P + termA_VV_FB_P)

    # Symmetric part for OO and VV blocks, 
    # equivalent to 1/2(PV +VP) scheme
    LA_OO_sym_HP = 0.5*(LA_OO_HP_AB + LA_OO_HP_AB.T) 
    LA_VV_sym_HP = 0.5*(LA_VV_HP_AB + LA_VV_HP_AB.T)

    # ========================================
    LA_OV_sym_HP = termA_OV_FA_P + termA_OV_FB_P 
    LA_OV_sym_PH = termA_OV_P_FA + termA_OV_P_FB 
    # Symmetric part of OV
    LA_OV_sym = 0.5*(LA_OV_sym_HP + LA_OV_sym_PH)

    LA_VO_sym_HP = termA_VO_FA_P + termA_VO_FB_P 
    LA_VO_sym_PH = termA_VO_P_FA + termA_VO_P_FB 
    # Symmetric part of VO
    LA_VO_sym = 0.5*(LA_VO_sym_HP + LA_VO_sym_PH)
    # print('LM-A(OV) ---> LM-A(VO).T', np.allclose(LA_OV_sym, LA_VO_sym.T))

    LA_block = np.block([[LA_OO_sym_HP, LA_OV_sym],
                        [LA_VO_sym, LA_VV_sym_HP]])
    
    print('termA_VV_FA_P')
    pprint(termA_VV_FA_P)
    print('termA_VV_FB_P')
    pprint(termA_VV_FB_P)
    # print('From LM utils')
    # print('termA_VO_FA_P')
    # pprint(termA_VO_FA_P)
    # print('termA_VO_FB_P')
    # pprint(termA_VO_FB_P)
    # print('termA_VO_P_FA')
    # pprint(termA_VO_P_FA)
    # print('termA_VO_P_FB')
    # pprint(termA_VO_P_FB)
    
    # print('L_omegaA, OO')
    # pprint(LA_OO_sym_HP)

    # print('L_omegaA, OV')
    # pprint(LA_OV_sym)

    # print('L_omegaA, VO')
    # pprint(LA_VO_sym)

    # print('L_omegaA, VV')
    # pprint(LA_VV_sym_HP)

    #===================================== 
    # LM Blocks (Monomer B)				 #
    #=====================================
    LB_OO_HP_AB = (termB_OO_FA_P + termB_OO_FB_P) 
    LB_VV_HP_AB = (termB_VV_FA_P + termB_VV_FB_P)

    # Symmetric part for OO and VV blocks
    LB_OO_sym_HP = 0.5*(LB_OO_HP_AB + LB_OO_HP_AB.T) 
    LB_VV_sym_HP = 0.5*(LB_VV_HP_AB + LB_VV_HP_AB.T)

    LB_OV_sym_HP = termB_OV_FA_P + termB_OV_FB_P 
    LB_OV_sym_PH = termB_OV_P_FA + termB_OV_P_FB 
    # Symmetric part of OV
    LB_OV_sym = 0.5*(LB_OV_sym_HP + LB_OV_sym_PH)

    LB_VO_sym_HP = termB_VO_FA_P + termB_VO_FB_P 
    LB_VO_sym_PH = termB_VO_P_FA + termB_VO_P_FB 
    # Symmetric part of VO
    # ===============================================
    # CHECK: LB_OV_sym == LB_VO_sym ?
    LB_VO_sym = 0.5*(LB_VO_sym_HP + LB_VO_sym_PH)
    # print('LM-B(OV) ---> LM-B(VO).T', np.allclose(LB_OV_sym, LB_VO_sym.T))

    LB_OV_sym_HP = termB_OV_FA_P + termB_OV_FB_P 

    # LB_block = np.block([[LB_OO_sym_HP, LB_OV_sym_HP],
    # 					[LB_OV_sym_HP.T, LB_VV_sym_HP]])

    LB_block = np.block([[LB_OO_sym_HP, LB_OV_sym],
                        [LB_VO_sym, LB_VV_sym_HP]])
    # print('LM-B ---> LM-B.T', np.allclose(LB_block, LB_block.T))

    return LA_block, LB_block

def form_l_terms_s2_alt(sapt:helper_SAPT, ca=None, cb=None):
    """
    TEST:
     Method to explicitly calculate the Fock contribution to the delta_F terms to verify(?)
    """
    if ca is not None and cb is not None:
        sapt.set_orbitals(ca=ca, cb=cb)   
    
    #====================================
    # L Blocks (Monomer A)		#
    #====================================
    termA_OO_FA_P = 0.5*( 
            -2 * oe.contract("ar,rp,Ba->pB", sapt.h_tot('A', 'ar'), sapt.s('rb'), sapt.s('ba')) 
            +oe.contract("rp,Ba,carc->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('aara')) 
            +oe.contract("rp,Ba,accr->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('aaar')) 
            -2 * oe.contract("rp,Ba,acrc->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('aara')) 
            -2 * oe.contract("rp,Ba,cacr->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('aaar')))
    termA_OO_FB_P = 0.5*( 
            -2 * oe.contract("Bs,ap,sa->pB", sapt.h_tot('B', 'bs'), sapt.s('ab'), sapt.s('sa')) 
            +oe.contract("ap,sa,bBsb->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('bbsb')) 
            +oe.contract("ap,sa,Bbbs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('bbbs')) 
            -2 * oe.contract("ap,sa,Bbsb->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('bbsb')) 
            -2 * oe.contract("ap,sa,bBbs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('bbbs')) 

            +2 * oe.contract("bp,Ba,ab->pB", sapt.h_tot('B', 'bb'), sapt.s('ba'), sapt.s('ab')) 
            +2 * oe.contract("Ba,ab,bcpc->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb')) 
            +2 * oe.contract("Ba,ab,cbcp->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb')) 
            -oe.contract("Ba,ab,cbpc->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb')) 
            -oe.contract("Ba,ab,bccp->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb')) )
    
    termA_OV_FA_P = 0.5*( 
            -2 * oe.contract("ar,rS,Ba->BS", sapt.h_tot('A', 'ar'), sapt.s('rs'), sapt.s('ba'))
            +oe.contract("rS,Ba,carc->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('aara')) 
            +oe.contract("rS,Ba,accr->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('aaar')) 
            -2 * oe.contract("rS,Ba,acrc->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('aara'))
            -2 * oe.contract("rS,Ba,cacr->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('aaar'))
            )
 
    
    termA_OV_FB_P = 0.5*(
            -2 * oe.contract("Bs,aS,sa->BS", sapt.h_tot('B', 'bs'), sapt.s('as'), sapt.s('sa')) 
            +oe.contract("aS,sa,bBsb->BS", sapt.s('as'), sapt.s('sa'), sapt.v('bbsb')) 
            +oe.contract("aS,sa,Bbbs->BS", sapt.s('as'), sapt.s('sa'), sapt.v('bbbs')) 
            -2 * oe.contract("aS,sa,Bbsb->BS", sapt.s('as'), sapt.s('sa'), sapt.v('bbsb'))  
            -2 * oe.contract("aS,sa,bBbs->BS", sapt.s('as'), sapt.s('sa'), sapt.v('bbbs'))

            +2 * oe.contract("bS,Ba,ab->BS", sapt.h_tot('B', 'bs'), sapt.s('ba'), sapt.s('ab'))
            +2 * oe.contract("Ba,ab,bcSc->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('bbsb')) 
            +2 * oe.contract("Ba,ab,cbcS->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbs'))  
            -oe.contract("Ba,ab,cbSc->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('bbsb')) 
            -oe.contract("Ba,ab,bccS->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbs')) 
            )
    
    termA_OV_P_FA =0.5*( 
            -2 * oe.contract("ra,aS,Br->BS", sapt.h_tot('A', 'ra'), sapt.s('as'), sapt.s('br'))
            +oe.contract("aS,Br,crac->BS", sapt.s('as'), sapt.s('br'), sapt.v('araa')) 
            +oe.contract("aS,Br,rcca->BS", sapt.s('as'), sapt.s('br'), sapt.v('raaa')) 
            -2 * oe.contract("aS,Br,rcac->BS", sapt.s('as'), sapt.s('br'), sapt.v('raaa'))
            -2 * oe.contract("aS,Br,crca->BS", sapt.s('as'), sapt.s('br'), sapt.v('araa')) 
            )
    
    termA_OV_P_FB = 0.5*(
            -2 * oe.contract("sS,as,Ba->BS", sapt.h_tot('B', 'ss'), sapt.s('as'), sapt.s('ba'))
            +oe.contract("as,Ba,bsSb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bssb')) 
            +oe.contract("as,Ba,sbbS->BS", sapt.s('as'), sapt.s('ba'), sapt.v('sbbs'))
            -2 * oe.contract("as,Ba,sbSb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('sbsb')) 
            -2 * oe.contract("as,Ba,bsbS->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bsbs')) 

            +2 * oe.contract("Bb,aS,ba->BS", sapt.h_tot('B', 'bb'), sapt.s('as'), sapt.s('ba'))
            +2 * oe.contract("aS,ba,Bcbc->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bbbb'))
            +2 * oe.contract("aS,ba,cBcb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bbbb')) 
            -oe.contract("aS,ba,cBbc->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bbbb')) 
            -oe.contract("aS,ba,Bccb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bbbb'))
        )	

    termA_VO_FA_P =0.5*( 
            -2 * oe.contract("ar,rB,Sa->SB", sapt.h_tot('A', 'ar'), sapt.s('rb'), sapt.s('sa')) 
            +oe.contract("rB,Sa,carc->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('aara'))
            +oe.contract("rB,Sa,accr->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('aaar')) 
            -2 * oe.contract("rB,Sa,acrc->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('aara')) 
            -2 * oe.contract("rB,Sa,cacr->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('aaar'))
            )
    
    termA_VO_FB_P =0.5*( 
            -2 * oe.contract("Ss,aB,sa->SB", sapt.h_tot('B', 'ss'), sapt.s('ab'), sapt.s('sa')) 
            +oe.contract("aB,sa,bSsb->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('bssb')) 				
            +oe.contract("aB,sa,Sbbs->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('sbbs'))  
            -2 * oe.contract("aB,sa,Sbsb->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('sbsb'))  
            -2 * oe.contract("aB,sa,bSbs->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('bsbs'))  

            +2 * oe.contract("bB,Sa,ab->SB", sapt.h_tot('B', 'bb'), sapt.s('sa'), sapt.s('ab')) 
            +2 * oe.contract("Sa,ab,bcBc->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbb'))  
            +2 * oe.contract("Sa,ab,cbcB->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbb'))  
            -oe.contract("Sa,ab,cbBc->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbb'))  
            -oe.contract("Sa,ab,bccB->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbb'))  
        )
    termA_VO_P_FA =0.5*( 
            -2 * oe.contract("ra,aB,Sr->SB", sapt.h_tot('A', 'ra'), sapt.s('ab'), sapt.s('sr'))
            +oe.contract("aB,Sr,crac->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('araa')) 
            +oe.contract("aB,Sr,rcca->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('raaa')) 
            -2 * oe.contract("aB,Sr,rcac->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('raaa'))  
            -2 * oe.contract("aB,Sr,crca->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('araa'))
            ) 
    
    termA_VO_P_FB =0.5*( 
            -2 * oe.contract("sB,as,Sa->SB", sapt.h_tot('B', 'sb'), sapt.s('as'), sapt.s('sa')) 
            +oe.contract("as,Sa,bsBb->SB", sapt.s('as'), sapt.s('sa'), sapt.v('bsbb'))   
            +oe.contract("as,Sa,sbbB->SB", sapt.s('as'), sapt.s('sa'), sapt.v('sbbb')) 
            -2 * oe.contract("as,Sa,sbBb->SB", sapt.s('as'), sapt.s('sa'), sapt.v('sbbb'))  
            -2 * oe.contract("as,Sa,bsbB->SB", sapt.s('as'), sapt.s('sa'), sapt.v('bsbb'))

            +2 * oe.contract("Sb,aB,ba->SB", sapt.h_tot('B', 'sb'), sapt.s('ab'), sapt.s('ba')) 
            +2 * oe.contract("aB,ba,Scbc->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('sbbb'))  			
            +2 * oe.contract("aB,ba,cScb->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('bsbb')) 				
            -oe.contract("aB,ba,cSbc->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('bsbb')) 					
            -oe.contract("aB,ba,Sccb->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('sbbb')) 
        )	
    
    termA_VV_FA_P =0.5*( 
            -2 * oe.contract("ar,rq,Sa->qS", sapt.h_tot('A', 'ar'), sapt.s('rs'), sapt.s('sa')) 
            +oe.contract("rq,Sa,carc->qS",  sapt.s('rs'), sapt.s('sa'), sapt.v('aara')) 
            +oe.contract("rq,Sa,accr->qS",  sapt.s('rs'), sapt.s('sa'), sapt.v('aaar')) 
            -2 * oe.contract("rq,Sa,acrc->qS",  sapt.s('rs'), sapt.s('sa'), sapt.v('aara')) 
            -2 * oe.contract("rq,Sa,cacr->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('aaar')))	
    
    
    termA_VV_FB_P =0.5*( 
            -2 * oe.contract("Ss,aq,sa->qS", sapt.h_tot('B', 'ss'), sapt.s('as'), sapt.s('sa')) 
            +oe.contract("aq,sa,bSsb->qS",  sapt.s('as'), sapt.s('sa'), sapt.v('bssb'))  
            +oe.contract("aq,sa,Sbbs->qS",  sapt.s('as'), sapt.s('sa'), sapt.v('sbbs')) 
            -2 * oe.contract("aq,sa,Sbsb->qS",  sapt.s('as'), sapt.s('sa'), sapt.v('sbsb')) 
            -2 * oe.contract("aq,sa,bSbs->qS", sapt.s('as'), sapt.s('sa'), sapt.v('bsbs'))  

            +2 * oe.contract("bq,Sa,ab->qS", sapt.h_tot('B', 'bs'), sapt.s('sa'), sapt.s('ab'))  
            +2 * oe.contract("Sa,ab,bcqc->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))  
            +2 * oe.contract("Sa,ab,cbcq->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))  
            -oe.contract("Sa,ab,cbqc->qS",  sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))  
            -oe.contract("Sa,ab,bccq->qS",  sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
    )
    # print('From Omega-exchange')
    # print('termA_VV_FA_P:')
    # pprint(termA_VV_FA_P)
    # print('termA_VV_FB_P:')
    # pprint(termA_VV_FB_P)
    
def form_l_terms_s2(sapt:helper_SAPT,ca=None, cb=None):        
        
        """
        Explicitly calculates the Fock contribution to the delta_F terms using the sapt.fock() method
        """
        print('L terms(S2) is added.')
        if ca is not None and cb is not None:
                sapt.set_orbitals(ca=ca, cb=cb)

        # ================          Fock contribution, Omega(A)         ==============
        # OO Block
        termA_OO_FA_P = 0.5*( 
                -2 * oe.contract("ar,rp,Ba->pB", sapt.fock('A', 'ar'), sapt.s('rb'), sapt.s('ba')))    
        termA_OO_FB_P =0.5*( 
                -2 * oe.contract("Bs,ap,sa->pB", sapt.fock('B', 'bs'), sapt.s('ab'), sapt.s('sa')) 
                +2 * oe.contract("bp,Ba,ab->pB", sapt.fock('B', 'bb'), sapt.s('ba'), sapt.s('ab')))
        
        # OV Block
        termA_OV_FA_P = 0.5*( 
                -2 * oe.contract("ar,rS,Ba->BS", sapt.fock('A', 'ar'), sapt.s('rs'), sapt.s('ba')))    
        termA_OV_FB_P = 0.5*( 
                -2 * oe.contract("Bs,aS,sa->BS", sapt.fock('B', 'bs'), sapt.s('as'), sapt.s('sa')) 
                +2 * oe.contract("bS,Ba,ab->BS", sapt.fock('B', 'bs'), sapt.s('ba'), sapt.s('ab')))    
        termA_OV_P_FA = 0.5*( 
                -2 * oe.contract("ra,aS,Br->BS", sapt.fock('A', 'ra'), sapt.s('as'), sapt.s('br')))
        termA_OV_P_FB = 0.5*( 
                -2 * oe.contract("sS,as,Ba->BS", sapt.fock('B', 'ss'), sapt.s('as'), sapt.s('ba'))
                +2 * oe.contract("Bb,aS,ba->BS", sapt.fock('B', 'bb'), sapt.s('as'), sapt.s('ba')))

        # VO Block
        termA_VO_FA_P = 0.5*( 
                -2 * oe.contract("ar,rB,Sa->SB", sapt.fock('A', 'ar'), sapt.s('rb'), sapt.s('sa'))) 
        termA_VO_FB_P = 0.5*( 
                -2 * oe.contract("Ss,aB,sa->SB", sapt.fock('B', 'ss'), sapt.s('ab'), sapt.s('sa'))   
                +2 * oe.contract("bB,Sa,ab->SB", sapt.fock('B', 'bb'), sapt.s('sa'), sapt.s('ab')) )
        termA_VO_P_FA = 0.5*( 
                -2 * oe.contract("ra,aB,Sr->SB", sapt.fock('A', 'ra'), sapt.s('ab'), sapt.s('sr')))
        termA_VO_P_FB = 0.5*( 
                -2 * oe.contract("sB,as,Sa->SB", sapt.fock('B', 'sb'), sapt.s('as'), sapt.s('sa')) 
                +2 * oe.contract("Sb,aB,ba->SB", sapt.fock('B', 'sb'), sapt.s('ab'), sapt.s('ba'))) 

        # VV Block
        termA_VV_FA_P =0.5*( 
                -2 * oe.contract("ar,rq,Sa->qS", sapt.fock('A', 'ar'), sapt.s('rs'), sapt.s('sa')))    
        termA_VV_FB_P =0.5*( 
                -2 * oe.contract("Ss,aq,sa->qS", sapt.fock('B', 'ss'), sapt.s('as'), sapt.s('sa')) 
                +2 * oe.contract("bq,Sa,ab->qS", sapt.fock('B', 'bs'), sapt.s('sa'), sapt.s('ab'))) 
        
        # ================          Fock contribution, Omega(B)         ==============
        # OO Block
        termB_OO_FA_P =0.5*(
                -2 * oe.contract("Ar,bp,rb->pA", sapt.fock('A', 'ar'), sapt.s('ba'), sapt.s('rb')) 
                +2 * oe.contract("ap,ba,Ab->pA", sapt.fock('A', 'aa'), sapt.s('ba'), sapt.s('ab')))    
        termB_OO_FB_P =0.5*(
                -2 * oe.contract("bs,sp,Ab->pA", sapt.fock('B', 'bs'), sapt.s('sa'), sapt.s('ab')))
        
        # OV Block
        termB_OV_FA_P =0.5*( 
                -2 * oe.contract("Ar,bR,rb->AR", sapt.fock('A', 'ar'), sapt.s('br'), sapt.s('rb')) 
                +2 * oe.contract("aR,ba,Ab->AR", sapt.fock('A', 'ar'), sapt.s('ba'), sapt.s('ab')))    
        termB_OV_FB_P =0.5*(
                -2 * oe.contract("bs,sR,Ab->AR", sapt.fock('B', 'bs'), sapt.s('sr'), sapt.s('ab')))
        termB_OV_P_FA = 0.5*(-2 * oe.contract("rR,br,Ab->AR", sapt.fock('A', 'rr'), sapt.s('br'), sapt.s('ab'))
                +2 * oe.contract("Aa,bR,ab->AR", sapt.fock('A', 'aa'), sapt.s('br'), sapt.s('ab')))    
        termB_OV_P_FB = 0.5*(
                -2 * oe.contract("sb,bR,As->AR", sapt.fock('B', 'sb'), sapt.s('br'), sapt.s('as')))
        
        # VO Block
        termB_VO_FA_P = 0.5*(
                -2 * oe.contract("Rr,bA,rb->RA", sapt.fock('A', 'rr'), sapt.s('ba'), sapt.s('rb')) 
                +2 * oe.contract("aA,ba,Rb->RA", sapt.fock('A', 'aa'), sapt.s('ba'), sapt.s('rb')))
        termB_VO_FB_P = 0.5*(
                -2 * oe.contract("bs,sA,Rb->RA", sapt.fock('B', 'bs'), sapt.s('sa'), sapt.s('rb')))
        termB_VO_P_FA = 0.5*(
                -2 * oe.contract("rA,br,Rb->RA", sapt.fock('A', 'ra'), sapt.s('br'), sapt.s('rb')) 
                +2 * oe.contract("Ra,bA,ab->RA", sapt.fock('A', 'ra'), sapt.s('ba'), sapt.s('ab')))
        termB_VO_P_FB = 0.5*(
                -2 * oe.contract("sb,bA,Rs->RA", sapt.fock('B', 'sb'), sapt.s('ba'), sapt.s('rs')))
        
        # VV Block
        termB_VV_FA_P =0.5*(
                -2 * oe.contract("Rr,bq,rb->qR", sapt.fock('A', 'rr'), sapt.s('br'), sapt.s('rb')) 
                +2 * oe.contract("aq,ba,Rb->qR", sapt.fock('A', 'ar'), sapt.s('ba'), sapt.s('rb')))
        
        termB_VV_FB_P =0.5*(
                -2 * oe.contract("bs,sq,Rb->qR", sapt.fock('B', 'bs'), sapt.s('sr'), sapt.s('rb'))) 
        
        #====================================
        # Landshoff Blocks (Monomer A)				#
        #====================================
        LA_OO_HP_AB = (termA_OO_FA_P + termA_OO_FB_P) 
        LA_VV_HP_AB = (termA_VV_FA_P + termA_VV_FB_P)

        # Symmetric part for OO and VV blocks, 
        # equivalent to 1/2(PV +VP) scheme
        LA_OO_sym_HP = 0.5*(LA_OO_HP_AB + LA_OO_HP_AB.T) 
        LA_VV_sym_HP = 0.5*(LA_VV_HP_AB + LA_VV_HP_AB.T)

        # ========================================
        LA_OV_sym_HP = termA_OV_FA_P + termA_OV_FB_P 
        LA_OV_sym_PH = termA_OV_P_FA + termA_OV_P_FB 
        # Symmetric part of OV
        LA_OV_sym = 0.5*(LA_OV_sym_HP + LA_OV_sym_PH)

        LA_VO_sym_HP = termA_VO_FA_P + termA_VO_FB_P 
        LA_VO_sym_PH = termA_VO_P_FA + termA_VO_P_FB 
        # Symmetric part of VO
        LA_VO_sym = 0.5*(LA_VO_sym_HP + LA_VO_sym_PH)
        # print('LM-A(OV) ---> LM-A(VO).T', np.allclose(LA_OV_sym, LA_VO_sym.T))

        LA_block = np.block([[LA_OO_sym_HP, LA_OV_sym],
                                [LA_VO_sym, LA_VV_sym_HP]])
        
        #===================================== 
        # LM Blocks (Monomer B)				 #
        #=====================================
        LB_OO_HP_AB = (termB_OO_FA_P + termB_OO_FB_P) 
        LB_VV_HP_AB = (termB_VV_FA_P + termB_VV_FB_P)

        # Symmetric part for OO and VV blocks
        LB_OO_sym_HP = 0.5*(LB_OO_HP_AB + LB_OO_HP_AB.T) 
        LB_VV_sym_HP = 0.5*(LB_VV_HP_AB + LB_VV_HP_AB.T)

        LB_OV_sym_HP = termB_OV_FA_P + termB_OV_FB_P 
        LB_OV_sym_PH = termB_OV_P_FA + termB_OV_P_FB 
        # Symmetric part of OV
        LB_OV_sym = 0.5*(LB_OV_sym_HP + LB_OV_sym_PH)

        LB_VO_sym_HP = termB_VO_FA_P + termB_VO_FB_P 
        LB_VO_sym_PH = termB_VO_P_FA + termB_VO_P_FB 
        # Symmetric part of VO
        # ===============================================
        # CHECK: LB_OV_sym == LB_VO_sym ?
        LB_VO_sym = 0.5*(LB_VO_sym_HP + LB_VO_sym_PH)
        # print('LM-B(OV) ---> LM-B(VO).T', np.allclose(LB_OV_sym, LB_VO_sym.T))

        LB_OV_sym_HP = termB_OV_FA_P + termB_OV_FB_P 

        # LB_block = np.block([[LB_OO_sym_HP, LB_OV_sym_HP],
        # 					[LB_OV_sym_HP.T, LB_VV_sym_HP]])

        LB_block = np.block([[LB_OO_sym_HP, LB_OV_sym],
                                [LB_VO_sym, LB_VV_sym_HP]])
        # print('LM-B ---> LM-B.T', np.allclose(LB_block, LB_block.T))

        return LA_block, LB_block
        
def form_l_terms_s4(sapt:helper_SAPT, ca=None, cb=None):
    
	"""
	Returns Landshoff part of LM potential in S4
	<{f,P}>
	"""
	print('L terms(S4) is added.')
	if ca is not None and cb is not None:
			sapt.set_orbitals(ca=ca, cb=cb)    
				
	fA_ar = sapt.fock('A', 'ar')
	fA_aa = sapt.fock('A', 'aa')
	fA_rr = sapt.fock('A', 'rr')
	
	fB_bb = sapt.fock('B', 'bb')
	fB_bs = sapt.fock('B', 'bs')
	fB_ss = sapt.fock('B', 'ss')
	
	# L potential due to monomer A
	#========================================
	termA_OO_FA_P = 0.5*(
			-2 * oe.contract("ar,rp,ba,Bc,cb->pB", fA_ar,sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('ab')) 
			-2 * oe.contract("ar,cp,Ba,bc,rb->pB", fA_ar,sapt.s('ab'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb')) )
	
	termA_OO_FB_P = 0.5*(
			+2 * oe.contract("bp,ca,Bd,ab,dc->pB", fB_bb, sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
			-2 * oe.contract("Bs,cp,sa,bc,ab->pB", fB_bs, sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'))
			-2 * oe.contract("bs,cp,Ba,sc,ab->pB", fB_bs, sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'))
	)
	
	termA_VV_FA_P = 0.5*(
			-2 * oe.contract("ar,rq,ba,Sc,cb->qS", fA_ar, sapt.s('rs'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'))
			-2 * oe.contract("ar,cq,Sa,bc,rb->qS", fA_ar, sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('rb'))
	)
	
	termA_VV_FB_P = 0.5*(
				+2 * oe.contract("bq,ca,Sd,ab,dc->qS", fB_bs, sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'))
				-2 * oe.contract("Ss,cq,sa,bc,ab->qS", fB_ss, sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'))
			-2 * oe.contract("bs,cq,Sa,sc,ab->qS", fB_bs, sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'))
	)
	
	termA_OV_FA_P = 0.5*(
			-2 * oe.contract("ar,rS,ba,Bc,cb->BS", fA_ar, sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'))
			-2 * oe.contract("ar,cS,Ba,bc,rb->BS", fA_ar, sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'))
	)
	termA_OV_FB_P = 0.5*(
				+2 * oe.contract("bS,ca,Bd,ab,dc->BS", fB_bs, sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
				-2 * oe.contract("Bs,cS,sa,bc,ab->BS", fB_bs, sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'))
			-2 * oe.contract("bs,cS,Ba,sc,ab->BS", fB_bs, sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'))
	)
	
	# L potential due to monomer B
	#========================================
	termB_OO_FA_P = 0.5*(
			+2 * oe.contract("ap,ba,cd,db,Ac->pA", fA_aa, sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
			-2 * oe.contract("Ar,cp,ba,rb,ac->pA", fA_ar, sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'))
			-2 * oe.contract("ar,cp,ba,Ab,rc->pA", fA_ar, sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
			
	)
	termB_OO_FB_P = 0.5*(
			-2 * oe.contract("bs,sp,ca,ab,Ac->pA", fB_bs, sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
			-2 * oe.contract("bs,cp,sa,Ab,ac->pA", fB_bs, sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'))                
	)
	termB_VV_FA_P = 0.5*(
			+2 * oe.contract("aq,ba,cd,db,Rc->qR", fA_ar, sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
			-2 * oe.contract("Rr,cq,ba,rb,ac->qR", fA_rr, sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'))
			-2 * oe.contract("ar,cq,ba,Rb,rc->qR", fA_ar, sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'))   
	)
	termB_VV_FB_P = 0.5*(
			-2 * oe.contract("bs,sq,ca,ab,Rc->qR", fB_bs, sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
			-2 * oe.contract("bs,cq,sa,Rb,ac->qR", fB_bs, sapt.s('br'), sapt.s('sa'), sapt.s('rb'), sapt.s('ab'))                
	)
	termB_OV_FA_P = 0.5*(
			+2 * oe.contract("aR,ba,cd,db,Ac->AR", fA_ar, sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
			-2 * oe.contract("Ar,cR,ba,rb,ac->AR", fA_ar, sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'))
			-2 * oe.contract("ar,cR,ba,Ab,rc->AR", fA_ar, sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
	)
	
	termB_OV_FB_P = 0.5*(
			-2 * oe.contract("bs,sR,ca,ab,Ac->AR", fB_bs, sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
			-2 * oe.contract("bs,cR,sa,Ab,ac->AR", fB_bs, sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'))
	)
	#====================================
	# L Blocks (Monomer A)				#
	#====================================
	lA_OO_FP_AB = (termA_OO_FA_P + termA_OO_FB_P) 
	lA_VV_FP_AB = (termA_VV_FA_P + termA_VV_FB_P)

	# Symmetric part for OO and VV blocks, 
	# equivalent to 1/2(PV +VP) scheme
	lA_OO_sym_FP = 0.5*(lA_OO_FP_AB + lA_OO_FP_AB.T) 
	lA_VV_sym_FP = 0.5*(lA_VV_FP_AB + lA_VV_FP_AB.T)
	lA_OV_sym_FP = termA_OV_FA_P + termA_OV_FB_P 

	lA_block = np.block([[lA_OO_sym_FP, lA_OV_sym_FP],
						[lA_OV_sym_FP.T, lA_VV_sym_FP]])
	
	#===================================== 
	# L Blocks (Monomer B)				 #
	#=====================================
	lB_OO_FP_AB = (termB_OO_FA_P + termB_OO_FB_P) 
	lB_VV_FP_AB = (termB_VV_FA_P + termB_VV_FB_P)

	# Symmetric part for OO and VV blocks, 
	# equivalent to 1/2(PV +VP) scheme
	lB_OO_sym_FP = 0.5*(lB_OO_FP_AB + lB_OO_FP_AB.T) 
	lB_VV_sym_FP = 0.5*(lB_VV_FP_AB + lB_VV_FP_AB.T)
	lB_OV_sym_FP = termB_OV_FA_P + termB_OV_FB_P 

	lB_block = np.block([[lB_OO_sym_FP, lB_OV_sym_FP],
						[lB_OV_sym_FP.T, lB_VV_sym_FP]])
	
	return lA_block, lB_block

def form_lm_terms_s2(sapt:helper_SAPT, ca=None, cb=None, sym_all=False):
	"""
	Calculates LM Terms in S2 approximation\\
	If sym_all == True, uses 1/2(HP +PH) symmetrization scheme,\\
	else, uses the scheme similar to omega-exchange,(<HP> for OV and <PH> for VO blocks)

	LM- Terms = 0.5*(<(HA + HB)P> + <P(HA + HB)>) 
	"""
	if ca is not None and cb is not None:
		sapt.set_orbitals(ca=ca, cb=cb)
	
	#===============================  Monomer A ========================================
	# OO blocks, HA*P Terms
	termA_OO_HA_P = 0.5*( 
                -2 * oe.contract("ar,rp,Ba->pB", sapt.h_tot('A', 'ar'), sapt.s('rb'), sapt.s('ba')) 
                +oe.contract("rp,Ba,carc->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('aara')) 
                +oe.contract("rp,Ba,accr->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('aaar')) 
                -2 * oe.contract("rp,Ba,acrc->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('aara')) 
                -2 * oe.contract("rp,Ba,cacr->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('aaar')))

	# OO blocks, HB*P Terms
	termA_OO_HB_P =0.5*( 
                -2 * oe.contract("Bs,ap,sa->pB", sapt.h_tot('B', 'bs'), sapt.s('ab'), sapt.s('sa')) 
                +2 * oe.contract("bp,Ba,ab->pB", sapt.h_tot('B', 'bb'), sapt.s('ba'), sapt.s('ab')) 
                +oe.contract("ap,sa,bBsb->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('bbsb')) 
                +oe.contract("ap,sa,Bbbs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('bbbs')) 
                +oe.contract("sa,ab,bBps->pB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs')) 
                +oe.contract("sa,ab,Bbsp->pB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
                +2 * oe.contract("Ba,ab,bcpc->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb')) 
                +2 * oe.contract("Ba,ab,cbcp->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb')) 
                -oe.contract("Ba,ab,cbpc->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb')) 
                -oe.contract("Ba,ab,bccp->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb')) 
                -2 * oe.contract("ap,sa,Bbsb->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('bbsb')) 
                -2 * oe.contract("ap,sa,bBbs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('bbbs')) 
                -2 * oe.contract("sa,ab,Bbps->pB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs')) 
                -2 * oe.contract("sa,ab,bBsp->pB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb')))
	
	# OV blocks, HA*P Terms
	termA_OV_HA_P =0.5*( 
                -2 * oe.contract("ar,rS,Ba->BS", sapt.h_tot('A', 'ar'), sapt.s('rs'), sapt.s('ba'))
                +oe.contract("rS,Ba,carc->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('aara')) 
                +oe.contract("rS,Ba,accr->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('aaar')) 
                -2 * oe.contract("rS,Ba,acrc->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('aara'))
                -2 * oe.contract("rS,Ba,cacr->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('aaar')))
	
	# OV blocks, HB*P Terms
	termA_OV_HB_P =0.5*( 
                -2 * oe.contract("Bs,aS,sa->BS", sapt.h_tot('B', 'bs'), sapt.s('as'), sapt.s('sa')) 
                +2 * oe.contract("bS,Ba,ab->BS", sapt.h_tot('B', 'bs'), sapt.s('ba'), sapt.s('ab'))
                +oe.contract("aS,sa,bBsb->BS", sapt.s('as'), sapt.s('sa'), sapt.v('bbsb')) 
                +oe.contract("aS,sa,Bbbs->BS", sapt.s('as'), sapt.s('sa'), sapt.v('bbbs')) 
                +oe.contract("sa,ab,bBSs->BS", sapt.s('sa'), sapt.s('ab'), sapt.v('bbss')) 
                +oe.contract("sa,ab,BbsS->BS", sapt.s('sa'), sapt.s('ab'), sapt.v('bbss')) 
                +2 * oe.contract("Ba,ab,bcSc->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('bbsb')) 
                +2 * oe.contract("Ba,ab,cbcS->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbs'))  
                -oe.contract("Ba,ab,cbSc->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('bbsb')) 
                -oe.contract("Ba,ab,bccS->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('bbbs'))  
                -2 * oe.contract("aS,sa,Bbsb->BS", sapt.s('as'), sapt.s('sa'), sapt.v('bbsb'))  
                -2 * oe.contract("aS,sa,bBbs->BS", sapt.s('as'), sapt.s('sa'), sapt.v('bbbs'))  
                -2 * oe.contract("sa,ab,BbSs->BS", sapt.s('sa'), sapt.s('ab'), sapt.v('bbss')) 
                -2 * oe.contract("sa,ab,bBsS->BS", sapt.s('sa'), sapt.s('ab'), sapt.v('bbss')) )
	
	termA_VV_HA_P =0.5*( 
                -2 * oe.contract("ar,rq,Sa->qS", sapt.h_tot('A', 'ar'), sapt.s('rs'), sapt.s('sa')) 
                +oe.contract("rq,Sa,carc->qS",  sapt.s('rs'), sapt.s('sa'), sapt.v('aara')) 
                +oe.contract("rq,Sa,accr->qS",  sapt.s('rs'), sapt.s('sa'), sapt.v('aaar')) 
                -2 * oe.contract("rq,Sa,acrc->qS",  sapt.s('rs'), sapt.s('sa'), sapt.v('aara')) 
                -2 * oe.contract("rq,Sa,cacr->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('aaar')))

	termA_VV_HB_P =0.5*( 
                -2 * oe.contract("Ss,aq,sa->qS", sapt.h_tot('B', 'ss'), sapt.s('as'), sapt.s('sa')) 
                +2 * oe.contract("bq,Sa,ab->qS", sapt.h_tot('B', 'bs'), sapt.s('sa'), sapt.s('ab'))  
                +oe.contract("aq,sa,bSsb->qS",  sapt.s('as'), sapt.s('sa'), sapt.v('bssb'))  
                +oe.contract("aq,sa,Sbbs->qS",  sapt.s('as'), sapt.s('sa'), sapt.v('sbbs')) 
                +oe.contract("sa,ab,bSqs->qS",  sapt.s('sa'), sapt.s('ab'), sapt.v('bsss')) 
                +oe.contract("sa,ab,Sbsq->qS",  sapt.s('sa'), sapt.s('ab'), sapt.v('sbss')) 
                +2 * oe.contract("Sa,ab,bcqc->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))  
                +2 * oe.contract("Sa,ab,cbcq->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))  
                -oe.contract("Sa,ab,cbqc->qS",  sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))  
                -oe.contract("Sa,ab,bccq->qS",  sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))  
                -2 * oe.contract("aq,sa,Sbsb->qS",  sapt.s('as'), sapt.s('sa'), sapt.v('sbsb')) 
                -2 * oe.contract("aq,sa,bSbs->qS", sapt.s('as'), sapt.s('sa'), sapt.v('bsbs'))  
                -2 * oe.contract("sa,ab,Sbqs->qS",  sapt.s('sa'), sapt.s('ab'), sapt.v('sbss')) 
                -2 * oe.contract("sa,ab,bSsq->qS",  sapt.s('sa'), sapt.s('ab'), sapt.v('bsss')))
	
	#===============================  Monomer B ========================================
	termB_OO_HA_P =0.5*(
                -2 * oe.contract("Ar,bp,rb->pA", sapt.h_tot('A', 'ar'), sapt.s('ba'), sapt.s('rb')) 
                +2 * oe.contract("ap,ba,Ab->pA", sapt.h_tot('A', 'aa'), sapt.s('ba'), sapt.s('ab'))
                +oe.contract("bp,rb,aAra->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('aara'))
                +oe.contract("bp,rb,Aaar->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('aaar')) 
                +oe.contract("ba,rb,aApr->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('aaar'))
                +oe.contract("ba,rb,Aarp->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('aara')) 
                +2 * oe.contract("ba,Ab,acpc->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('aaaa'))
                +2 * oe.contract("ba,Ab,cacp->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('aaaa'))
                -oe.contract("ba,Ab,capc->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('aaaa'))
                -oe.contract("ba,Ab,accp->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('aaaa'))
                -2 * oe.contract("bp,rb,Aara->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('aara'))
                -2 * oe.contract("bp,rb,aAar->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('aaar'))
                -2 * oe.contract("ba,rb,Aapr->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('aaar')) 
                -2 * oe.contract("ba,rb,aArp->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('aara')))
	
	termB_OO_HB_P =0.5*(
                -2 * oe.contract("bs,sp,Ab->pA", sapt.h_tot('B', 'bs'), sapt.s('sa'), sapt.s('ab'))
                +oe.contract("sp,Ab,cbsc->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
                +oe.contract("sp,Ab,bccs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
                -2 * oe.contract("sp,Ab,bcsc->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
                -2 * oe.contract("sp,Ab,cbcs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs')))
	
	termB_OV_HA_P =0.5*( 
                -2 * oe.contract("Ar,bR,rb->AR", sapt.h_tot('A', 'ar'), sapt.s('br'), sapt.s('rb')) 
                +2 * oe.contract("aR,ba,Ab->AR", sapt.h_tot('A', 'ar'), sapt.s('ba'), sapt.s('ab')) 
                +oe.contract("bR,rb,aAra->AR", sapt.s('br'), sapt.s('rb'), sapt.v('aara')) 
                +oe.contract("bR,rb,Aaar->AR", sapt.s('br'), sapt.s('rb'), sapt.v('aaar'))  
                +oe.contract("ba,rb,aARr->AR", sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
                +oe.contract("ba,rb,AarR->AR", sapt.s('ba'), sapt.s('rb'), sapt.v('aarr')) 
                +2 * oe.contract("ba,Ab,acRc->AR", sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
                +2 * oe.contract("ba,Ab,cacR->AR", sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
                -oe.contract("ba,Ab,caRc->AR", sapt.s('ba'), sapt.s('ab'), sapt.v('aara')) 
                -oe.contract("ba,Ab,accR->AR", sapt.s('ba'), sapt.s('ab'), sapt.v('aaar')) 
                -2 * oe.contract("bR,rb,Aara->AR", sapt.s('br'), sapt.s('rb'), sapt.v('aara')) 
                -2 * oe.contract("bR,rb,aAar->AR", sapt.s('br'), sapt.s('rb'), sapt.v('aaar')) 
                -2 * oe.contract("ba,rb,AaRr->AR", sapt.s('ba'), sapt.s('rb'), sapt.v('aarr')) 
                -2 * oe.contract("ba,rb,aArR->AR", sapt.s('ba'), sapt.s('rb'), sapt.v('aarr')))
	
	termB_OV_HB_P =0.5*(
                -2 * oe.contract("bs,sR,Ab->AR", sapt.h_tot('B', 'bs'), sapt.s('sr'), sapt.s('ab'))
                +oe.contract("sR,Ab,cbsc->AR", sapt.s('sr'), sapt.s('ab'), sapt.v('bbsb'))
                +oe.contract("sR,Ab,bccs->AR", sapt.s('sr'), sapt.s('ab'), sapt.v('bbbs')) 
                -2 * oe.contract("sR,Ab,bcsc->AR", sapt.s('sr'), sapt.s('ab'), sapt.v('bbsb')) 
                -2 * oe.contract("sR,Ab,cbcs->AR", sapt.s('sr'), sapt.s('ab'), sapt.v('bbbs')))
	
	termB_VV_HA_P =0.5*(
                -2 * oe.contract("Rr,bq,rb->qR", sapt.h_tot('A', 'rr'), sapt.s('br'), sapt.s('rb')) 
                +2 * oe.contract("aq,ba,Rb->qR", sapt.h_tot('A', 'ar'), sapt.s('ba'), sapt.s('rb')) 
                +oe.contract("bq,rb,aRra->qR", sapt.s('br'), sapt.s('rb'), sapt.v('arra')) 
                +oe.contract("bq,rb,Raar->qR", sapt.s('br'), sapt.s('rb'), sapt.v('raar')) 
                +oe.contract("ba,rb,aRqr->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('arrr')) 
                +oe.contract("ba,rb,Rarq->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('rarr')) 
                +2 * oe.contract("ba,Rb,acqc->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('aara')) 
                +2 * oe.contract("ba,Rb,cacq->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('aaar')) 
                -oe.contract("ba,Rb,caqc->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('aara')) 
                -oe.contract("ba,Rb,accq->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('aaar')) 
                -2 * oe.contract("bq,rb,Rara->qR", sapt.s('br'), sapt.s('rb'), sapt.v('rara'))
                -2 * oe.contract("bq,rb,aRar->qR", sapt.s('br'), sapt.s('rb'), sapt.v('arar')) 
                -2 * oe.contract("ba,rb,Raqr->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('rarr'))
                -2 * oe.contract("ba,rb,aRrq->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('arrr')))
	
	termB_VV_HB_P =0.5*(
                -2 * oe.contract("bs,sq,Rb->qR", sapt.h_tot('B', 'bs'), sapt.s('sr'), sapt.s('rb')) 
                +oe.contract("sq,Rb,cbsc->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('bbsb'))
                +oe.contract("sq,Rb,bccs->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('bbbs')) 
                -2 * oe.contract("sq,Rb,bcsc->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('bbsb'))
                -2 * oe.contract("sq,Rb,cbcs->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('bbbs')))
	
	if not sym_all:
		#====================================
		# LM Blocks (Monomer A)				#
		#====================================
		lmA_OO_HP_AB = (termA_OO_HA_P + termA_OO_HB_P) 
		lmA_VV_HP_AB = (termA_VV_HA_P + termA_VV_HB_P)

		# Symmetric part for OO and VV blocks, 
		# equivalent to 1/2(PV +VP) scheme
		lmA_OO_sym_HP = 0.5*(lmA_OO_HP_AB + lmA_OO_HP_AB.T) 
		lmA_VV_sym_HP = 0.5*(lmA_VV_HP_AB + lmA_VV_HP_AB.T)
		lmA_OV_sym_HP = termA_OV_HA_P + termA_OV_HB_P 

		lmA_block = np.block([[lmA_OO_sym_HP, lmA_OV_sym_HP],
							[lmA_OV_sym_HP.T, lmA_VV_sym_HP]])
		
		#===================================== 
		# LM Blocks (Monomer B)				 #
		#=====================================
		lmB_OO_HP_AB = (termB_OO_HA_P + termB_OO_HB_P) 
		lmB_VV_HP_AB = (termB_VV_HA_P + termB_VV_HB_P)

		# Symmetric part for OO and VV blocks, 
		# equivalent to 1/2(PV +VP) scheme
		lmB_OO_sym_HP = 0.5*(lmB_OO_HP_AB + lmB_OO_HP_AB.T) 
		lmB_VV_sym_HP = 0.5*(lmB_VV_HP_AB + lmB_VV_HP_AB.T)
		lmB_OV_sym_HP = termB_OV_HA_P + termB_OV_HB_P 

		lmB_block = np.block([[lmB_OO_sym_HP, lmB_OV_sym_HP],
							[lmB_OV_sym_HP.T, lmB_VV_sym_HP]])
		
		return lmA_block, lmB_block
	else:
		#=======================================
		# Use 1/2(HP +PH) symmetrization scheme
		#======================================

		termA_OV_P_HA =0.5*( 
			-2 * oe.contract("ra,aS,Br->BS", sapt.h_tot('A', 'ra'), sapt.s('as'), sapt.s('br'))
			+oe.contract("aS,Br,crac->BS", sapt.s('as'), sapt.s('br'), sapt.v('araa')) 
			+oe.contract("aS,Br,rcca->BS", sapt.s('as'), sapt.s('br'), sapt.v('raaa')) 
			-2 * oe.contract("aS,Br,rcac->BS", sapt.s('as'), sapt.s('br'), sapt.v('raaa'))
			-2 * oe.contract("aS,Br,crca->BS", sapt.s('as'), sapt.s('br'), sapt.v('araa')) )

		termA_OV_P_HB =0.5*( 
			-2 * oe.contract("sS,as,Ba->BS", sapt.h_tot('B', 'ss'), sapt.s('as'), sapt.s('ba'))
			+2 * oe.contract("Bb,aS,ba->BS", sapt.h_tot('B', 'bb'), sapt.s('as'), sapt.s('ba'))
			+oe.contract("as,Ba,bsSb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bssb')) 
			+oe.contract("as,Ba,sbbS->BS", sapt.s('as'), sapt.s('ba'), sapt.v('sbbs'))
			+oe.contract("as,ba,sBSb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('sbsb')) 
			+oe.contract("as,ba,BsbS->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bsbs')) 
			+2 * oe.contract("aS,ba,Bcbc->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bbbb'))
			+2 * oe.contract("aS,ba,cBcb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bbbb')) 
			-oe.contract("aS,ba,cBbc->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bbbb')) 
			-oe.contract("aS,ba,Bccb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bbbb'))
			-2 * oe.contract("as,Ba,sbSb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('sbsb')) 
			-2 * oe.contract("as,Ba,bsbS->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bsbs'))  
			-2 * oe.contract("as,ba,BsSb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('bssb')) 
			-2 * oe.contract("as,ba,sBbS->BS", sapt.s('as'), sapt.s('ba'), sapt.v('sbbs')))
		
		termA_VO_HA_P =0.5*( 
			-2 * oe.contract("ar,rB,Sa->SB", sapt.h_tot('A', 'ar'), sapt.s('rb'), sapt.s('sa')) 
			+oe.contract("rB,Sa,carc->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('aara'))
			+oe.contract("rB,Sa,accr->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('aaar')) 
			-2 * oe.contract("rB,Sa,acrc->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('aara')) 
			-2 * oe.contract("rB,Sa,cacr->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('aaar')))
		
		termA_VO_HB_P =0.5*( 
			-2 * oe.contract("Ss,aB,sa->SB", sapt.h_tot('B', 'ss'), sapt.s('ab'), sapt.s('sa'))   
			+2 * oe.contract("bB,Sa,ab->SB", sapt.h_tot('B', 'bb'), sapt.s('sa'), sapt.s('ab')) 
			+oe.contract("aB,sa,bSsb->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('bssb')) 
			+oe.contract("aB,sa,Sbbs->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('sbbs')) 
			+oe.contract("sa,ab,bSBs->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bsbs'))  
			+oe.contract("sa,ab,SbsB->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('sbsb'))  
			+2 * oe.contract("Sa,ab,bcBc->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbb'))  
			+2 * oe.contract("Sa,ab,cbcB->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbb'))  
			-oe.contract("Sa,ab,cbBc->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbb'))  
			-oe.contract("Sa,ab,bccB->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bbbb'))  
			-2 * oe.contract("aB,sa,Sbsb->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('sbsb'))  
			-2 * oe.contract("aB,sa,bSbs->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('bsbs'))  
			-2 * oe.contract("sa,ab,SbBs->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('sbbs'))  
			-2 * oe.contract("sa,ab,bSsB->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('bssb')) )
		
		termA_VO_P_HA =0.5*( 
			-2 * oe.contract("ra,aB,Sr->SB", sapt.h_tot('A', 'ra'), sapt.s('ab'), sapt.s('sr'))
			+oe.contract("aB,Sr,crac->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('araa')) 
			+oe.contract("aB,Sr,rcca->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('raaa')) 
			-2 * oe.contract("aB,Sr,rcac->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('raaa'))  
			-2 * oe.contract("aB,Sr,crca->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('araa'))) 

		termA_VO_P_HB =0.5*( 
			-2 * oe.contract("sB,as,Sa->SB", sapt.h_tot('B', 'sb'), sapt.s('as'), sapt.s('sa')) 
			+2 * oe.contract("Sb,aB,ba->SB", sapt.h_tot('B', 'sb'), sapt.s('ab'), sapt.s('ba')) 
			+oe.contract("as,Sa,bsBb->SB", sapt.s('as'), sapt.s('sa'), sapt.v('bsbb'))   
			+oe.contract("as,Sa,sbbB->SB", sapt.s('as'), sapt.s('sa'), sapt.v('sbbb')) 
			+oe.contract("as,ba,sSBb->SB", sapt.s('as'), sapt.s('ba'), sapt.v('ssbb'))  
			+oe.contract("as,ba,SsbB->SB", sapt.s('as'), sapt.s('ba'), sapt.v('ssbb'))  
			+2 * oe.contract("aB,ba,Scbc->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('sbbb'))  
			+2 * oe.contract("aB,ba,cScb->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('bsbb')) 
			-oe.contract("aB,ba,cSbc->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('bsbb')) 
			-oe.contract("aB,ba,Sccb->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('sbbb')) 
			-2 * oe.contract("as,Sa,sbBb->SB", sapt.s('as'), sapt.s('sa'), sapt.v('sbbb'))  
			-2 * oe.contract("as,Sa,bsbB->SB", sapt.s('as'), sapt.s('sa'), sapt.v('bsbb')) 
			-2 * oe.contract("as,ba,SsBb->SB", sapt.s('as'), sapt.s('ba'), sapt.v('ssbb'))  
			-2 * oe.contract("as,ba,sSbB->SB", sapt.s('as'), sapt.s('ba'), sapt.v('ssbb')))
		
		termB_OV_P_HA =0.5*(
                        -2 * oe.contract("rR,br,Ab->AR", sapt.h_tot('A', 'rr'), sapt.s('br'), sapt.s('ab'))
                        +2 * oe.contract("Aa,bR,ab->AR", sapt.h_tot('A', 'aa'), sapt.s('br'), sapt.s('ab'))
                        +oe.contract("br,Ab,arRa->AR", sapt.s('br'), sapt.s('ab'), sapt.v('arra')) 
                        +oe.contract("br,Ab,raaR->AR", sapt.s('br'), sapt.s('ab'), sapt.v('raar')) 
                        +oe.contract("br,ab,rARa->AR", sapt.s('br'), sapt.s('ab'), sapt.v('rara')) 
                        +oe.contract("br,ab,AraR->AR", sapt.s('br'), sapt.s('ab'), sapt.v('arar')) 
                        +2 * oe.contract("bR,ab,Acac->AR", sapt.s('br'), sapt.s('ab'), sapt.v('aaaa')) 
                        +2 * oe.contract("bR,ab,cAca->AR", sapt.s('br'), sapt.s('ab'), sapt.v('aaaa'))
                        -oe.contract("bR,ab,cAac->AR", sapt.s('br'), sapt.s('ab'), sapt.v('aaaa'))
                        -oe.contract("bR,ab,Acca->AR", sapt.s('br'), sapt.s('ab'), sapt.v('aaaa'))
                        -2 * oe.contract("br,Ab,raRa->AR",sapt.s('br'), sapt.s('ab'), sapt.v('rara')) 
                        -2 * oe.contract("br,Ab,araR->AR", sapt.s('br'), sapt.s('ab'), sapt.v('arar')) 
                        -2 * oe.contract("br,ab,ArRa->AR", sapt.s('br'), sapt.s('ab'), sapt.v('arra')) 
                        -2 * oe.contract("br,ab,rAaR->AR", sapt.s('br'), sapt.s('ab'), sapt.v('raar')))
		
		termB_OV_P_HB =0.5*(
                        -2 * oe.contract("sb,bR,As->AR", sapt.h_tot('B', 'sb'), sapt.s('br'), sapt.s('as'))
                        +oe.contract("bR,As,csbc->AR", sapt.s('br'), sapt.s('as'), sapt.v('bsbb')) 
                        +oe.contract("bR,As,sccb->AR", sapt.s('br'), sapt.s('as'), sapt.v('sbbb'))
                        -2 * oe.contract("bR,As,scbc->AR", sapt.s('br'), sapt.s('as'), sapt.v('sbbb')) 
                        -2 * oe.contract("bR,As,cscb->AR", sapt.s('br'), sapt.s('as'), sapt.v('bsbb')))
                                
		termB_VO_HA_P =0.5*(
                        -2 * oe.contract("Rr,bA,rb->RA", sapt.h_tot('A', 'rr'), sapt.s('ba'), sapt.s('rb')) 
                        +2 * oe.contract("aA,ba,Rb->RA", sapt.h_tot('A', 'aa'), sapt.s('ba'), sapt.s('rb')) 
                        +oe.contract("bA,rb,aRra->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('arra')) 
                        +oe.contract("bA,rb,Raar->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('raar')) 
                        +oe.contract("ba,rb,aRAr->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('arar')) 
                        +oe.contract("ba,rb,RarA->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('rara')) 
                        +2 * oe.contract("ba,Rb,acAc->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('aaaa'))
                        +2 * oe.contract("ba,Rb,cacA->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('aaaa'))
                        -oe.contract("ba,Rb,caAc->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('aaaa'))
                        -oe.contract("ba,Rb,accA->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('aaaa'))
                        -2 * oe.contract("bA,rb,Rara->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('rara')) 
                        -2 * oe.contract("bA,rb,aRar->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('arar'))
                        -2 * oe.contract("ba,rb,RaAr->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('raar')) 
                        -2 * oe.contract("ba,rb,aRrA->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('arra')))
		
		termB_VO_HB_P =0.5*(
                        -2 * oe.contract("bs,sA,Rb->RA", sapt.h_tot('B', 'bs'), sapt.s('sa'), sapt.s('rb'))
                        +oe.contract("sA,Rb,cbsc->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('bbsb')) 
                        +oe.contract("sA,Rb,bccs->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('bbbs')) 
                        -2 * oe.contract("sA,Rb,bcsc->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('bbsb'))
                        -2 * oe.contract("sA,Rb,cbcs->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('bbbs')))
		
		termB_VO_P_HA =0.5*(
                        -2 * oe.contract("rA,br,Rb->RA", sapt.h_tot('A', 'ra'), sapt.s('br'), sapt.s('rb')) 
                        +2 * oe.contract("Ra,bA,ab->RA", sapt.h_tot('A', 'ra'), sapt.s('ba'), sapt.s('ab')) 
                        +oe.contract("br,Rb,arAa->RA", sapt.s('br'), sapt.s('rb'), sapt.v('araa')) 
                        +oe.contract("br,Rb,raaA->RA", sapt.s('br'), sapt.s('rb'), sapt.v('raaa')) 
                        +oe.contract("br,ab,rRAa->RA", sapt.s('br'), sapt.s('ab'), sapt.v('rraa')) 
                        +oe.contract("br,ab,RraA->RA", sapt.s('br'), sapt.s('ab'), sapt.v('rraa')) 
                        +2 * oe.contract("bA,ab,Rcac->RA", sapt.s('ba'), sapt.s('ab'), sapt.v('raaa')) 
                        +2 * oe.contract("bA,ab,cRca->RA", sapt.s('ba'), sapt.s('ab'), sapt.v('araa')) 
                        -oe.contract("bA,ab,cRac->RA", sapt.s('ba'), sapt.s('ab'), sapt.v('araa')) 
                        -oe.contract("bA,ab,Rcca->RA", sapt.s('ba'), sapt.s('ab'), sapt.v('raaa')) 
                        -2 * oe.contract("br,Rb,raAa->RA", sapt.s('br'), sapt.s('rb'), sapt.v('raaa')) 
                        -2 * oe.contract("br,Rb,araA->RA", sapt.s('br'), sapt.s('rb'), sapt.v('araa')) 
                        -2 * oe.contract("br,ab,RrAa->RA", sapt.s('br'), sapt.s('ab'), sapt.v('rraa')) 
                        -2 * oe.contract("br,ab,rRaA->RA", sapt.s('br'), sapt.s('ab'), sapt.v('rraa')))
		
		termB_VO_P_HB =0.5*(
                        -2 * oe.contract("sb,bA,Rs->RA", sapt.h_tot('B', 'sb'), sapt.s('ba'), sapt.s('rs'))
                        +oe.contract("bA,Rs,csbc->RA", sapt.s('ba'), sapt.s('rs'), sapt.v('bsbb'))
                        +oe.contract("bA,Rs,sccb->RA", sapt.s('ba'), sapt.s('rs'), sapt.v('sbbb')) 
                        -2 * oe.contract("bA,Rs,scbc->RA", sapt.s('ba'), sapt.s('rs'), sapt.v('sbbb'))
                        -2 * oe.contract("bA,Rs,cscb->RA", sapt.s('ba'), sapt.s('rs'), sapt.v('bsbb')))
		
		#====================================
		# LM Blocks (Monomer A)				#
		#====================================
		lmA_OO_HP_AB = (termA_OO_HA_P + termA_OO_HB_P) 
		lmA_VV_HP_AB = (termA_VV_HA_P + termA_VV_HB_P)

		# Symmetric part for OO and VV blocks, 
		# equivalent to 1/2(PV +VP) scheme
		lmA_OO_sym_HP = 0.5*(lmA_OO_HP_AB + lmA_OO_HP_AB.T) 
		lmA_VV_sym_HP = 0.5*(lmA_VV_HP_AB + lmA_VV_HP_AB.T)

		# ========================================
		lmA_OV_sym_HP = termA_OV_HA_P + termA_OV_HB_P 
		lmA_OV_sym_PH = termA_OV_P_HA + termA_OV_P_HB 
		# Symmetric part of OV
		lmA_OV_sym = 0.5*(lmA_OV_sym_HP + lmA_OV_sym_PH)

		lmA_VO_sym_HP = termA_VO_HA_P + termA_VO_HB_P 
		lmA_VO_sym_PH = termA_VO_P_HA + termA_VO_P_HB 
		# Symmetric part of VO
		lmA_VO_sym = 0.5*(lmA_VO_sym_HP + lmA_VO_sym_PH)
		# print('LM-A(OV) ---> LM-A(VO).T', np.allclose(lmA_OV_sym, lmA_VO_sym.T))

		lmA_block = np.block([[lmA_OO_sym_HP, lmA_OV_sym],
							[lmA_VO_sym, lmA_VV_sym_HP]])
		# print('LM-A ---> LM-A.T', np.allclose(lmA_block, lmA_block.T))
		
		#===================================== 
		# LM Blocks (Monomer B)				 #
		#=====================================
		lmB_OO_HP_AB = (termB_OO_HA_P + termB_OO_HB_P) 
		lmB_VV_HP_AB = (termB_VV_HA_P + termB_VV_HB_P)

		# Symmetric part for OO and VV blocks
		lmB_OO_sym_HP = 0.5*(lmB_OO_HP_AB + lmB_OO_HP_AB.T) 
		lmB_VV_sym_HP = 0.5*(lmB_VV_HP_AB + lmB_VV_HP_AB.T)

		lmB_OV_sym_HP = termB_OV_HA_P + termB_OV_HB_P 
		lmB_OV_sym_PH = termB_OV_P_HA + termB_OV_P_HB 
		# Symmetric part of OV
		lmB_OV_sym = 0.5*(lmB_OV_sym_HP + lmB_OV_sym_PH)

		lmB_VO_sym_HP = termB_VO_HA_P + termB_VO_HB_P 
		lmB_VO_sym_PH = termB_VO_P_HA + termB_VO_P_HB 
		# Symmetric part of VO
		# ===============================================
		# CHECK: lmB_OV_sym == lmB_VO_sym ?
		lmB_VO_sym = 0.5*(lmB_VO_sym_HP + lmB_VO_sym_PH)
		# print('LM-B(OV) ---> LM-B(VO).T', np.allclose(lmB_OV_sym, lmB_VO_sym.T))

		lmB_OV_sym_HP = termB_OV_HA_P + termB_OV_HB_P 

		# lmB_block = np.block([[lmB_OO_sym_HP, lmB_OV_sym_HP],
		# 					[lmB_OV_sym_HP.T, lmB_VV_sym_HP]])

		lmB_block = np.block([[lmB_OO_sym_HP, lmB_OV_sym],
							[lmB_VO_sym, lmB_VV_sym_HP]])
		# print('LM-B ---> LM-B.T', np.allclose(lmB_block, lmB_block.T))

		return lmA_block, lmB_block
      
def form_lm_terms_s4(sapt:helper_SAPT, ca=None, cb=None, sym_all=False): 
    print('Calculating LM terms in S4 approximation')
        
    """
            Calculates LM Terms in S2 approximation\\
            If sym_all == True, uses 1/2(HP +PH) symmetrization scheme,\\
            else, uses the scheme similar to omega-exchange,(<HP> for OV and <PH> for VO blocks)

            LM- Terms = 0.5*(<(HA + HB)P> + <P(HA + HB)>) 
            
    """

    if ca is not None and cb is not None:
            sapt.set_orbitals(ca=ca, cb=cb)

    #==================================== OO Block ====================================
    termA_OO_HA_P = 0.5*(
            oe.contract("rp,Ba,bc,ab,dcrd->pB", sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
            +oe.contract("rp,Ba,bc,ab,ceer->pB", sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
            +oe.contract("ap,ba,Bc,rb,dcrd->pB", sapt.s('ab'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb'), sapt.v('aara'))
            +oe.contract("ap,ba,Bc,rb,ceer->pB", sapt.s('ab'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb'), sapt.v('aaar'))
            +2 * oe.contract("cp,ba,Bd,rb,adrc->pB", sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
            +2 * oe.contract("cp,ba,Bd,rb,dacr->pB", sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
            -oe.contract("cp,ba,Bd,rb,darc->pB", sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
            -oe.contract("cp,ba,Bd,rb,adcr->pB", sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
            -2 * oe.contract("ar,rp,ba,Bc,cb->pB", sapt.h_tot('A', 'ar'),sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('ab')) 
            -2 * oe.contract("ar,cp,Ba,bc,rb->pB", sapt.h_tot('A', 'ar'),sapt.s('ab'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb')) 
            -2 * oe.contract("rp,Ba,bc,ab,cere->pB", sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
            -2 * oe.contract("rp,Ba,bc,ab,dcdr->pB", sapt.s('rb'), sapt.s('ba'),sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
            -2 * oe.contract("ap,ba,Bc,rb,cere->pB", sapt.s('ab'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb'), sapt.v('aara'))
            -2 * oe.contract("ap,ba,Bc,rb,dcdr->pB", sapt.s('ab'), sapt.s('ba'),sapt.s('ba'), sapt.s('rb'), sapt.v('aaar'))
            )
        
    termA_OO_HB_P = 0.5*(
        oe.contract("cp,Ba,sc,ab,ebse->pB", sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("cp,Ba,sc,ab,bees->pB", sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
        +oe.contract("cp,sa,bc,ab,eBse->pB", sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("cp,sa,bc,ab,Bees->pB", sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbbs'))
        +oe.contract("sa,Bc,ab,ce,beps->pB", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        +oe.contract("sa,Bc,cb,ae,besp->pB", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("ba,sc,cb,ae,eBps->pB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        +oe.contract("ba,sc,cb,ae,Besp->pB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        +2 * oe.contract("bp,ca,Bd,ab,dc->pB", sapt.h_tot('B', 'bb'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
        +2 * oe.contract("cp,sa,dc,ab,bBsd->pB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbss'))
        +2 * oe.contract("cp,da,sc,ab,Bbsd->pB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbss'))
        +2 * oe.contract("ba,Bc,cb,ae,egpg->pB", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbb'))
        +2 * oe.contract("ba,Bc,cb,ae,fefp->pB", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbb'))
        -oe.contract("cp,sa,dc,ab,Bbsd->pB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbss'))
        -oe.contract("cp,da,sc,ab,bBsd->pB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbss'))
        -oe.contract("ba,Bc,cb,ae,fepf->pB", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbb'))
        -oe.contract("ba,Bc,cb,ae,eggp->pB", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbb'))
        -2 * oe.contract("Bs,cp,sa,bc,ab->pB", sapt.h_tot('B', 'bs'), sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'))
        -2 * oe.contract("bs,cp,Ba,sc,ab->pB", sapt.h_tot('B', 'bs'), sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'))
        -2 * oe.contract("cp,Ba,sc,ab,bese->pB", sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("cp,Ba,sc,ab,ebes->pB", sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("cp,sa,bc,ab,Bese->pB", sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("cp,sa,bc,ab,eBes->pB", sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("sa,Bc,ab,ce,besp->pB", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("sa,Bc,cb,ae,beps->pB", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("ba,sc,cb,ae,Beps->pB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("ba,sc,cb,ae,eBsp->pB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        )
            
    #==================================== VV Block ====================================
    termA_VV_HA_P = 0.5*(
        oe.contract("rq,Sa,bc,ab,dcrd->qS", sapt.s('rs'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
        +oe.contract("rq,Sa,bc,ab,ceer->qS", sapt.s('rs'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
        +oe.contract("aq,ba,Sc,rb,dcrd->qS", sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aara'))
        +oe.contract("aq,ba,Sc,rb,ceer->qS", sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aaar'))
        +2 * oe.contract("cq,ba,Sd,rb,adrc->qS", sapt.s('rs'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aarr'))
        +2 * oe.contract("cq,ba,Sd,rb,dacr->qS", sapt.s('rs'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aarr'))
        -oe.contract("cq,ba,Sd,rb,darc->qS", sapt.s('rs'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aarr'))
        -oe.contract("cq,ba,Sd,rb,adcr->qS", sapt.s('rs'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aarr'))
        -2 * oe.contract("ar,rq,ba,Sc,cb->qS", sapt.h_tot('A', 'ar'), sapt.s('rs'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'))
        -2 * oe.contract("ar,cq,Sa,bc,rb->qS", sapt.h_tot('A', 'ar'), sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('rb'))
        -2 * oe.contract("rq,Sa,bc,ab,cere->qS", sapt.s('rs'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
        -2 * oe.contract("rq,Sa,bc,ab,dcdr->qS", sapt.s('rs'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
        -2 * oe.contract("aq,ba,Sc,rb,cere->qS", sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aara'))
        -2 * oe.contract("aq,ba,Sc,rb,dcdr->qS", sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aaar'))
                    
    )

    termA_VV_HB_P = 0.5*(
            
        oe.contract("cq,Sa,sc,ab,ebse->qS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("cq,Sa,sc,ab,bees->qS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
        +oe.contract("cq,sa,bc,ab,eSse->qS", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bssb'))
        +oe.contract("cq,sa,bc,ab,Sees->qS", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbs'))
        +oe.contract("sa,Sc,ab,ce,beqs->qS", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        +oe.contract("sa,Sc,cb,ae,besq->qS", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        +oe.contract("ba,sc,cb,ae,eSqs->qS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bsss'))
        +oe.contract("ba,sc,cb,ae,Sesq->qS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('sbss'))
        +2 * oe.contract("bq,ca,Sd,ab,dc->qS", sapt.h_tot('B', 'bs'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'))
        +2 * oe.contract("cq,sa,dc,ab,bSsd->qS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bsss'))
        +2 * oe.contract("cq,da,sc,ab,Sbsd->qS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('sbss'))
        +2 * oe.contract("ba,Sc,cb,ae,egqg->qS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        +2 * oe.contract("ba,Sc,cb,ae,fefq->qS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        -oe.contract("cq,sa,dc,ab,Sbsd->qS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('sbss'))
        -oe.contract("cq,da,sc,ab,bSsd->qS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bsss'))
        -oe.contract("ba,Sc,cb,ae,feqf->qS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        -oe.contract("ba,Sc,cb,ae,eggq->qS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("Ss,cq,sa,bc,ab->qS", sapt.h_tot('B', 'ss'), sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'))
        -2 * oe.contract("bs,cq,Sa,sc,ab->qS", sapt.h_tot('B', 'bs'), sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'))
        -2 * oe.contract("cq,Sa,sc,ab,bese->qS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("cq,Sa,sc,ab,ebes->qS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("cq,sa,bc,ab,Sese->qS", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbsb'))
        -2 * oe.contract("cq,sa,bc,ab,eSes->qS", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbs'))
        -2 * oe.contract("sa,Sc,ab,ce,besq->qS", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        -2 * oe.contract("sa,Sc,cb,ae,beqs->qS", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        -2 * oe.contract("ba,sc,cb,ae,Seqs->qS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('sbss'))
        -2 * oe.contract("ba,sc,cb,ae,eSsq->qS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bsss'))
    )

    # ============ OV Block
    termA_OV_HA_P = 0.5*(
            +oe.contract("rS,Ba,bc,ab,dcrd->BS", sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
            +oe.contract("rS,Ba,bc,ab,ceer->BS", sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
            +oe.contract("aS,ba,Bc,rb,dcrd->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.v('aara'))
            +oe.contract("aS,ba,Bc,rb,ceer->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.v('aaar'))
            +2 * oe.contract("cS,ba,Bd,rb,adrc->BS", sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
            +2 * oe.contract("cS,ba,Bd,rb,dacr->BS", sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
            -oe.contract("cS,ba,Bd,rb,darc->BS", sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
            -oe.contract("cS,ba,Bd,rb,adcr->BS", sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.v('aarr'))
            -2 * oe.contract("ar,rS,ba,Bc,cb->BS", sapt.h_tot('A', 'ar'), sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'))
            -2 * oe.contract("ar,cS,Ba,bc,rb->BS", sapt.h_tot('A', 'ar'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'))
            -2 * oe.contract("rS,Ba,bc,ab,cere->BS", sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
            -2 * oe.contract("rS,Ba,bc,ab,dcdr->BS", sapt.s('rs'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
            -2 * oe.contract("aS,ba,Bc,rb,cere->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.v('aara'))
            -2 * oe.contract("aS,ba,Bc,rb,dcdr->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.v('aaar'))
        )    

    termA_OV_HB_P = 0.5*(
            +oe.contract("cS,Ba,sc,ab,ebse->BS", sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
            +oe.contract("cS,Ba,sc,ab,bees->BS", sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
            +oe.contract("cS,sa,bc,ab,eBse->BS", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbsb'))
            +oe.contract("cS,sa,bc,ab,Bees->BS", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbbs'))
            +oe.contract("sa,Bc,ab,ce,beSs->BS", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
            +oe.contract("sa,Bc,cb,ae,besS->BS", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
            +oe.contract("ba,sc,cb,ae,eBSs->BS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
            +oe.contract("ba,sc,cb,ae,BesS->BS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
            +2 * oe.contract("bS,ca,Bd,ab,dc->BS", sapt.h_tot('B', 'bs'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
            +2 * oe.contract("cS,sa,dc,ab,bBsd->BS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbss'))
            +2 * oe.contract("cS,da,sc,ab,Bbsd->BS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbss'))
            +2 * oe.contract("ba,Bc,cb,ae,egSg->BS", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
            +2 * oe.contract("ba,Bc,cb,ae,fefS->BS", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
            -oe.contract("cS,sa,dc,ab,Bbsd->BS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbss'))
            -oe.contract("cS,da,sc,ab,bBsd->BS", sapt.s('as'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbss'))
            -oe.contract("ba,Bc,cb,ae,feSf->BS", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
            -oe.contract("ba,Bc,cb,ae,eggS->BS", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
            -2 * oe.contract("Bs,cS,sa,bc,ab->BS", sapt.h_tot('B', 'bs'), sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'))
            -2 * oe.contract("bs,cS,Ba,sc,ab->BS", sapt.h_tot('B', 'bs'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'))
            -2 * oe.contract("cS,Ba,sc,ab,bese->BS", sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
            -2 * oe.contract("cS,Ba,sc,ab,ebes->BS", sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
            -2 * oe.contract("cS,sa,bc,ab,Bese->BS", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbsb'))
            -2 * oe.contract("cS,sa,bc,ab,eBes->BS", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbbs'))
            -2 * oe.contract("sa,Bc,ab,ce,besS->BS", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
            -2 * oe.contract("sa,Bc,cb,ae,beSs->BS", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
            -2 * oe.contract("ba,sc,cb,ae,BeSs->BS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
            -2 * oe.contract("ba,sc,cb,ae,eBsS->BS", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        )    

    #=========================== Omega(B)
    # ==========================================================================================================================
    # OO Blocks
    termB_OO_HA_P = 0.5*(
        +oe.contract("cp,ba,Ab,rc,eare->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aara'))
        +oe.contract("cp,ba,Ab,rc,aeer->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaar'))
        +oe.contract("cp,ba,rb,ac,eAre->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aara'))
        +oe.contract("cp,ba,rb,ac,Aeer->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aaar'))
        +oe.contract("ba,cd,rb,Ac,adpr->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aaar'))
        +oe.contract("ca,bd,rb,Ac,adrp->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aara'))
        +oe.contract("ca,bd,ab,rc,dApr->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaar'))
        +oe.contract("ca,bd,ab,rc,Adrp->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aara'))
        +2 * oe.contract("ap,ba,cd,db,Ac->pA", sapt.h_tot('A', 'aa'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
        +2 * oe.contract("cp,ba,rb,dc,aArd->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        +2 * oe.contract("cp,ba,db,rc,Aard->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        +2 * oe.contract("ca,bd,ab,Ac,dgpg->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaaa'))
        +2 * oe.contract("ca,bd,ab,Ac,fdfp->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaaa'))
        -oe.contract("cp,ba,rb,dc,Aard->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        -oe.contract("cp,ba,db,rc,aArd->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        -oe.contract("ca,bd,ab,Ac,fdpf->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaaa'))
        -oe.contract("ca,bd,ab,Ac,dggp->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaaa'))
        -2 * oe.contract("Ar,cp,ba,rb,ac->pA", sapt.h_tot('A', 'ar'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'))
        -2 * oe.contract("ar,cp,ba,Ab,rc->pA", sapt.h_tot('A', 'ar'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
        -2 * oe.contract("cp,ba,Ab,rc,aere->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aara'))
        -2 * oe.contract("cp,ba,Ab,rc,eaer->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaar'))
        -2 * oe.contract("cp,ba,rb,ac,Aere->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aara'))
        -2 * oe.contract("cp,ba,rb,ac,eAer->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aaar'))
        -2 * oe.contract("ba,cd,rb,Ac,adrp->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aara'))
        -2 * oe.contract("ca,bd,rb,Ac,adpr->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aaar'))
        -2 * oe.contract("ca,bd,ab,rc,Adpr->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaar'))
        -2 * oe.contract("ca,bd,ab,rc,dArp->pA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aara'))
    )

    termB_OO_HB_P = 0.5*(
        +oe.contract("sp,ba,Ab,ac,dcsd->pA", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("sp,ba,Ab,ac,cees->pA", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        +oe.contract("bp,sa,ab,Ac,dcsd->pA", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("bp,sa,ab,Ac,cees->pA", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        +2 * oe.contract("cp,sa,ab,Ad,bdsc->pA", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        +2 * oe.contract("cp,sa,ab,Ad,dbcs->pA", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        -oe.contract("cp,sa,ab,Ad,dbsc->pA", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        -oe.contract("cp,sa,ab,Ad,bdcs->pA", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        -2 * oe.contract("bs,sp,ca,ab,Ac->pA", sapt.h_tot('B', 'bs'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
        -2 * oe.contract("bs,cp,sa,Ab,ac->pA", sapt.h_tot('B', 'bs'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'))
        -2 * oe.contract("sp,ba,Ab,ac,cese->pA", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("sp,ba,Ab,ac,dcds->pA", sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("bp,sa,ab,Ac,cese->pA", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("bp,sa,ab,Ac,dcds->pA", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        )

    # VV Block
    termB_VV_HA_P = 0.5*(
        +oe.contract("cq,ba,Rb,rc,eare->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aara'))
        +oe.contract("cq,ba,Rb,rc,aeer->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aaar'))
        +oe.contract("cq,ba,rb,ac,eRre->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('arra'))
        +oe.contract("cq,ba,rb,ac,Reer->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('raar'))
        +oe.contract("ba,cd,rb,Rc,adqr->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        +oe.contract("ca,bd,rb,Rc,adrq->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        +oe.contract("ca,bd,ab,rc,dRqr->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('arrr'))
        +oe.contract("ca,bd,ab,rc,Rdrq->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('rarr'))
        +2 * oe.contract("aq,ba,cd,db,Rc->qR", sapt.h_tot('A', 'ar'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
        +2 * oe.contract("cq,ba,rb,dc,aRrd->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('arrr'))
        +2 * oe.contract("cq,ba,db,rc,Rard->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('rarr'))
        +2 * oe.contract("ca,bd,ab,Rc,dgqg->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aara'))
        +2 * oe.contract("ca,bd,ab,Rc,fdfq->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaar'))
        -oe.contract("cq,ba,rb,dc,Rard->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('rarr'))
        -oe.contract("cq,ba,db,rc,aRrd->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('arrr'))
        -oe.contract("ca,bd,ab,Rc,fdqf->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aara'))
        -oe.contract("ca,bd,ab,Rc,dggq->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaar'))
        -2 * oe.contract("Rr,cq,ba,rb,ac->qR", sapt.h_tot('A', 'rr'), sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'))
        -2 * oe.contract("ar,cq,ba,Rb,rc->qR", sapt.h_tot('A', 'ar'), sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'))
        -2 * oe.contract("cq,ba,Rb,rc,aere->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aara'))
        -2 * oe.contract("cq,ba,Rb,rc,eaer->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aaar'))
        -2 * oe.contract("cq,ba,rb,ac,Rere->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('rara'))
        -2 * oe.contract("cq,ba,rb,ac,eRer->qR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('arar'))
        -2 * oe.contract("ba,cd,rb,Rc,adrq->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        -2 * oe.contract("ca,bd,rb,Rc,adqr->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        -2 * oe.contract("ca,bd,ab,rc,Rdqr->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('rarr'))
        -2 * oe.contract("ca,bd,ab,rc,dRrq->qR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('arrr'))
    )

    termB_VV_HB_P = 0.5*(
        +oe.contract("sq,ba,Rb,ac,dcsd->qR", sapt.s('sr'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("sq,ba,Rb,ac,cees->qR", sapt.s('sr'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('bbbs'))
        +oe.contract("bq,sa,ab,Rc,dcsd->qR", sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbsb'))
        +oe.contract("bq,sa,ab,Rc,cees->qR", sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbbs'))
        +2 * oe.contract("cq,sa,ab,Rd,bdsc->qR", sapt.s('sr'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbss'))
        +2 * oe.contract("cq,sa,ab,Rd,dbcs->qR", sapt.s('sr'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbss'))
        -oe.contract("cq,sa,ab,Rd,dbsc->qR", sapt.s('sr'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbss'))
        -oe.contract("cq,sa,ab,Rd,bdcs->qR", sapt.s('sr'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbss'))
        -2 * oe.contract("bs,sq,ca,ab,Rc->qR", sapt.h_tot('B', 'bs'), sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
        -2 * oe.contract("bs,cq,sa,Rb,ac->qR", sapt.h_tot('B', 'bs'), sapt.s('br'), sapt.s('sa'), sapt.s('rb'), sapt.s('ab'))
        -2 * oe.contract("sq,ba,Rb,ac,cese->qR", sapt.s('sr'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("sq,ba,Rb,ac,dcds->qR", sapt.s('sr'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("bq,sa,ab,Rc,cese->qR", sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbsb'))
        -2 * oe.contract("bq,sa,ab,Rc,dcds->qR", sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbbs'))
    )

    # OV Block
    termB_OV_HA_P = 0.5*(
        +oe.contract("cR,ba,Ab,rc,eare->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aara'))
        +oe.contract("cR,ba,Ab,rc,aeer->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaar'))
        +oe.contract("cR,ba,rb,ac,eAre->AR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aara'))
        +oe.contract("cR,ba,rb,ac,Aeer->AR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aaar'))
        +oe.contract("ba,cd,rb,Ac,adRr->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aarr'))
        +oe.contract("ca,bd,rb,Ac,adrR->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aarr'))
        +oe.contract("ca,bd,ab,rc,dARr->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aarr'))
        +oe.contract("ca,bd,ab,rc,AdrR->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aarr'))
        +2 * oe.contract("aR,ba,cd,db,Ac->AR", sapt.h_tot('A', 'ar'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
        +2 * oe.contract("cR,ba,rb,dc,aArd->AR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        +2 * oe.contract("cR,ba,db,rc,Aard->AR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        +2 * oe.contract("ca,bd,ab,Ac,dgRg->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aara'))
        +2 * oe.contract("ca,bd,ab,Ac,fdfR->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaar'))
        -oe.contract("cR,ba,rb,dc,Aard->AR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        -oe.contract("cR,ba,db,rc,aArd->AR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aarr'))
        -oe.contract("ca,bd,ab,Ac,fdRf->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aara'))
        -oe.contract("ca,bd,ab,Ac,dggR->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaar'))
        -2 * oe.contract("Ar,cR,ba,rb,ac->AR", sapt.h_tot('A', 'ar'), sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'))
        -2 * oe.contract("ar,cR,ba,Ab,rc->AR", sapt.h_tot('A', 'ar'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
        -2 * oe.contract("cR,ba,Ab,rc,aere->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aara'))
        -2 * oe.contract("cR,ba,Ab,rc,eaer->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaar'))
        -2 * oe.contract("cR,ba,rb,ac,Aere->AR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aara'))
        -2 * oe.contract("cR,ba,rb,ac,eAer->AR", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aaar'))
        -2 * oe.contract("ba,cd,rb,Ac,adrR->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aarr'))
        -2 * oe.contract("ca,bd,rb,Ac,adRr->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('aarr'))
        -2 * oe.contract("ca,bd,ab,rc,AdRr->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aarr'))
        -2 * oe.contract("ca,bd,ab,rc,dArR->AR", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aarr'))
    )

    termB_OV_HB_P = 0.5*(
        +oe.contract("sR,ba,Ab,ac,dcsd->AR", sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("sR,ba,Ab,ac,cees->AR", sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        +oe.contract("bR,sa,ab,Ac,dcsd->AR", sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        +oe.contract("bR,sa,ab,Ac,cees->AR", sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        +2 * oe.contract("cR,sa,ab,Ad,bdsc->AR", sapt.s('sr'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        +2 * oe.contract("cR,sa,ab,Ad,dbcs->AR", sapt.s('sr'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        -oe.contract("cR,sa,ab,Ad,dbsc->AR", sapt.s('sr'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        -oe.contract("cR,sa,ab,Ad,bdcs->AR", sapt.s('sr'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbss'))
        -2 * oe.contract("bs,sR,ca,ab,Ac->AR", sapt.h_tot('B', 'bs'), sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
        -2 * oe.contract("bs,cR,sa,Ab,ac->AR", sapt.h_tot('B', 'bs'), sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'))
        -2 * oe.contract("sR,ba,Ab,ac,cese->AR", sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("sR,ba,Ab,ac,dcds->AR", sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
        -2 * oe.contract("bR,sa,ab,Ac,cese->AR", sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
        -2 * oe.contract("bR,sa,ab,Ac,dcds->AR", sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
    )

    
    if not sym_all:
        
        
        #====================================
        # LM Blocks (Monomer A)				#
        #====================================
        lmA_OO_HP_AB = (termA_OO_HA_P + termA_OO_HB_P) 
        lmA_VV_HP_AB = (termA_VV_HA_P + termA_VV_HB_P)

        # Symmetric part for OO and VV blocks, 
        # equivalent to 1/2(PV +VP) scheme
        lmA_OO_sym_HP = 0.5*(lmA_OO_HP_AB + lmA_OO_HP_AB.T) 
        lmA_VV_sym_HP = 0.5*(lmA_VV_HP_AB + lmA_VV_HP_AB.T)
        lmA_OV_sym_HP = termA_OV_HA_P + termA_OV_HB_P 

        lmA_block = np.block([[lmA_OO_sym_HP, lmA_OV_sym_HP],
                            [lmA_OV_sym_HP.T, lmA_VV_sym_HP]])
        
        #===================================== 
        # LM Blocks (Monomer B)				 #
        #=====================================
        lmB_OO_HP_AB = (termB_OO_HA_P + termB_OO_HB_P) 
        lmB_VV_HP_AB = (termB_VV_HA_P + termB_VV_HB_P)

        # Symmetric part for OO and VV blocks, 
        # equivalent to 1/2(PV +VP) scheme
        lmB_OO_sym_HP = 0.5*(lmB_OO_HP_AB + lmB_OO_HP_AB.T) 
        lmB_VV_sym_HP = 0.5*(lmB_VV_HP_AB + lmB_VV_HP_AB.T)
        lmB_OV_sym_HP = termB_OV_HA_P + termB_OV_HB_P 

        lmB_block = np.block([[lmB_OO_sym_HP, lmB_OV_sym_HP],
                            [lmB_OV_sym_HP.T, lmB_VV_sym_HP]])
        return lmA_block, lmB_block       
                  
        

    else:
        termA_OV_P_HA = 0.5*(
                +oe.contract("aS,Br,ba,cb,drcd->BS", sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.v('araa'))
                +oe.contract("aS,Br,ba,cb,rddc->BS", sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.v('raaa'))
                +oe.contract("cS,br,Ba,ab,drcd->BS", sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.v('araa'))
                +oe.contract("cS,br,Ba,ab,rddc->BS", sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.v('raaa'))
                +2 * oe.contract("cS,br,Bd,ab,rdac->BS", sapt.s('as'), sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.v('rraa'))
                +2 * oe.contract("cS,br,Bd,ab,drca->BS", sapt.s('as'), sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.v('rraa'))
                -oe.contract("cS,br,Bd,ab,drac->BS", sapt.s('as'), sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.v('rraa'))
                -oe.contract("cS,br,Bd,ab,rdca->BS", sapt.s('as'), sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.v('rraa'))
                -2 * oe.contract("ra,aS,br,Bc,cb->BS", sapt.h_tot('A', 'ra'), sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'))
                -2 * oe.contract("ra,cS,Br,bc,ab->BS", sapt.h_tot('A', 'ra'), sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'))
                -2 * oe.contract("aS,Br,ba,cb,rdcd->BS", sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.v('raaa'))
                -2 * oe.contract("aS,Br,ba,cb,drdc->BS", sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.v('araa'))
                -2 * oe.contract("cS,br,Ba,ab,rdcd->BS", sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.v('raaa'))
                -2 * oe.contract("cS,br,Ba,ab,drdc->BS", sapt.s('as'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.v('araa'))
            )
        
        termA_OV_P_HB = 0.5*(
            +oe.contract("aS,cs,ba,Bc,esbe->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('bsbb'))
            +oe.contract("aS,cs,ba,Bc,seeb->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('sbbb'))
            +oe.contract("cS,as,ba,dc,Bsbd->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('bsbb'))
            +oe.contract("cS,as,da,bc,sBbd->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('sbbb'))
            +oe.contract("cs,Ba,bc,ab,esSe->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bssb'))
            +oe.contract("cs,Ba,bc,ab,seeS->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbs'))
            +oe.contract("cs,da,bc,ab,sBSd->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbsb'))
            +oe.contract("cs,da,bc,ab,BsdS->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbs'))
            +2 * oe.contract("Bb,cS,ba,dc,ad->BS", sapt.h_tot('B', 'bb'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'))
            +2 * oe.contract("cS,da,bc,ab,Bfdf->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb'))
            +2 * oe.contract("cS,da,bc,ab,fBfd->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb'))
            +2 * oe.contract("as,cd,ba,Bc,sdbS->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('ssbs'))
            +2 * oe.contract("cs,ad,ba,Bc,sdSb->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('sssb'))
            -oe.contract("cS,da,bc,ab,fBdf->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb'))
            -oe.contract("cS,da,bc,ab,Bffd->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bbbb'))
            -oe.contract("as,cd,ba,Bc,sdSb->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('sssb'))
            -oe.contract("cs,ad,ba,Bc,sdbS->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('ssbs'))
            -2 * oe.contract("sS,cs,Ba,bc,ab->BS", sapt.h_tot('B', 'ss'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'))
            -2 * oe.contract("sb,aS,cs,ba,Bc->BS", sapt.h_tot('B', 'sb'), sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'))
            -2 * oe.contract("aS,cs,ba,Bc,sebe->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('sbbb'))
            -2 * oe.contract("aS,cs,ba,Bc,eseb->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('bsbb'))
            -2 * oe.contract("cS,as,ba,dc,sBbd->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('sbbb'))
            -2 * oe.contract("cS,as,da,bc,Bsbd->BS", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('bsbb'))
            -2 * oe.contract("cs,Ba,bc,ab,seSe->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbsb'))
            -2 * oe.contract("cs,Ba,bc,ab,eseS->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbs'))
            -2 * oe.contract("cs,da,bc,ab,BsSd->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bssb'))
            -2 * oe.contract("cs,da,bc,ab,sBdS->BS", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbs'))
        )

        # ============ VO Block
        termA_VO_HA_P = 0.5*(
                +oe.contract("rB,Sa,bc,ab,dcrd->SB", sapt.s('rb'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
                +oe.contract("rB,Sa,bc,ab,ceer->SB", sapt.s('rb'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
                +oe.contract("aB,ba,Sc,rb,dcrd->SB", sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aara'))
                +oe.contract("aB,ba,Sc,rb,ceer->SB", sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aaar'))
                +2 * oe.contract("cB,ba,Sd,rb,adrc->SB", sapt.s('rb'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aarr'))
                +2 * oe.contract("cB,ba,Sd,rb,dacr->SB", sapt.s('rb'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aarr'))
                -oe.contract("cB,ba,Sd,rb,darc->SB", sapt.s('rb'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aarr'))
                -oe.contract("cB,ba,Sd,rb,adcr->SB", sapt.s('rb'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aarr'))
                -2 * oe.contract("ar,rB,ba,Sc,cb->SB", sapt.h_tot('A','ar'), sapt.s('rb'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'))
                -2 * oe.contract("ar,cB,Sa,bc,rb->SB", sapt.h_tot('A','ar'), sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('rb'))
                -2 * oe.contract("rB,Sa,bc,ab,cere->SB", sapt.s('rb'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('aara'))
                -2 * oe.contract("rB,Sa,bc,ab,dcdr->SB", sapt.s('rb'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('aaar'))
                -2 * oe.contract("aB,ba,Sc,rb,cere->SB", sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aara'))
                -2 * oe.contract("aB,ba,Sc,rb,dcdr->SB", sapt.s('ab'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.v('aaar'))
                )

        termA_VO_HB_P = 0.5*(
            +oe.contract("cB,Sa,sc,ab,ebse->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
            +oe.contract("cB,Sa,sc,ab,bees->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
            +oe.contract("cB,sa,bc,ab,eSse->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bssb'))
            +oe.contract("cB,sa,bc,ab,Sees->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbs'))
            +oe.contract("sa,Sc,ab,ce,beBs->SB", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
            +oe.contract("sa,Sc,cb,ae,besB->SB", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
            +oe.contract("ba,sc,cb,ae,eSBs->SB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bsbs'))
            +oe.contract("ba,sc,cb,ae,SesB->SB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('sbsb'))
            +2 * oe.contract("bB,ca,Sd,ab,dc->SB", sapt.h_tot('B','bb'), sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'))
            +2 * oe.contract("cB,sa,dc,ab,bSsd->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bsss'))
            +2 * oe.contract("cB,da,sc,ab,Sbsd->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('sbss'))
            +2 * oe.contract("ba,Sc,cb,ae,egBg->SB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbb'))
            +2 * oe.contract("ba,Sc,cb,ae,fefB->SB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbb'))
            -oe.contract("cB,sa,dc,ab,Sbsd->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('sbss'))
            -oe.contract("cB,da,sc,ab,bSsd->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bsss'))
            -oe.contract("ba,Sc,cb,ae,feBf->SB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbb'))
            -oe.contract("ba,Sc,cb,ae,eggB->SB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbb'))
            -2 * oe.contract("Ss,cB,sa,bc,ab->SB", sapt.h_tot('B','ss'), sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'))
            -2 * oe.contract("bs,cB,Sa,sc,ab->SB", sapt.h_tot('B','bs'), sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'))
            -2 * oe.contract("cB,Sa,sc,ab,bese->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbsb'))
            -2 * oe.contract("cB,Sa,sc,ab,ebes->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.v('bbbs'))
            -2 * oe.contract("cB,sa,bc,ab,Sese->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbsb'))
            -2 * oe.contract("cB,sa,bc,ab,eSes->SB", sapt.s('ab'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbs'))
            -2 * oe.contract("sa,Sc,ab,ce,besB->SB", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbsb'))
            -2 * oe.contract("sa,Sc,cb,ae,beBs->SB", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bbbs'))
            -2 * oe.contract("ba,sc,cb,ae,SeBs->SB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('sbbs'))
            -2 * oe.contract("ba,sc,cb,ae,eSsB->SB", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('ab'), sapt.v('bssb'))
        )

        termA_VO_P_HA = 0.5*(
            +oe.contract("aB,Sr,ba,cb,drcd->SB", sapt.s('ab'), sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.v('araa'))
            +oe.contract("aB,Sr,ba,cb,rddc->SB", sapt.s('ab'), sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.v('raaa'))
            +oe.contract("cB,br,Sa,ab,drcd->SB", sapt.s('ab'), sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.v('araa'))
            +oe.contract("cB,br,Sa,ab,rddc->SB", sapt.s('ab'), sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.v('raaa'))
            +2 * oe.contract("cB,br,Sd,ab,rdac->SB", sapt.s('ab'), sapt.s('br'), sapt.s('sr'), sapt.s('ab'), sapt.v('rraa'))
            +2 * oe.contract("cB,br,Sd,ab,drca->SB", sapt.s('ab'), sapt.s('br'), sapt.s('sr'), sapt.s('ab'), sapt.v('rraa'))
            -oe.contract("cB,br,Sd,ab,drac->SB", sapt.s('ab'), sapt.s('br'), sapt.s('sr'), sapt.s('ab'), sapt.v('rraa'))
            -oe.contract("cB,br,Sd,ab,rdca->SB", sapt.s('ab'), sapt.s('br'), sapt.s('sr'), sapt.s('ab'), sapt.v('rraa'))
            -2 * oe.contract("ra,aB,br,Sc,cb->SB", sapt.h_tot('A','ra'), sapt.s('ab'), sapt.s('br'), sapt.s('sa'), sapt.s('ab'))
            -2 * oe.contract("ra,cB,Sr,bc,ab->SB", sapt.h_tot('A','ra'), sapt.s('ab'), sapt.s('sr'), sapt.s('ba'), sapt.s('ab'))
            -2 * oe.contract("aB,Sr,ba,cb,rdcd->SB", sapt.s('ab'), sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.v('raaa'))
            -2 * oe.contract("aB,Sr,ba,cb,drdc->SB", sapt.s('ab'), sapt.s('sr'), sapt.s('ba'), sapt.s('ab'), sapt.v('araa'))
            -2 * oe.contract("cB,br,Sa,ab,rdcd->SB", sapt.s('ab'), sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.v('raaa'))
            -2 * oe.contract("cB,br,Sa,ab,drdc->SB", sapt.s('ab'), sapt.s('br'), sapt.s('sa'), sapt.s('ab'), sapt.v('araa'))
        )

        termA_VO_P_HB = 0.5*(
                +oe.contract("aB,cs,ba,Sc,esbe->SB", sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.v('bsbb'))
                +oe.contract("aB,cs,ba,Sc,seeb->SB", sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.v('sbbb'))
                +oe.contract("cB,as,ba,dc,Ssbd->SB", sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('ssbb'))
                +oe.contract("cB,as,da,bc,sSbd->SB", sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('ssbb'))
                +oe.contract("cs,Sa,bc,ab,esBe->SB", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
                +oe.contract("cs,Sa,bc,ab,seeB->SB", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
                +oe.contract("cs,da,bc,ab,sSBd->SB", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('ssbb'))
                +oe.contract("cs,da,bc,ab,SsdB->SB", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('ssbb'))
                +2 * oe.contract("Sb,cB,ba,dc,ad->SB", sapt.h_tot('B','sb'), sapt.s('ab'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'))
                +2 * oe.contract("cB,da,bc,ab,Sfdf->SB", sapt.s('ab'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
                +2 * oe.contract("cB,da,bc,ab,fSfd->SB", sapt.s('ab'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
                +2 * oe.contract("as,cd,ba,Sc,sdbB->SB", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.v('ssbb'))
                +2 * oe.contract("cs,ad,ba,Sc,sdBb->SB", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.v('ssbb'))
                -oe.contract("cB,da,bc,ab,fSdf->SB", sapt.s('ab'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
                -oe.contract("cB,da,bc,ab,Sffd->SB", sapt.s('ab'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
                -oe.contract("as,cd,ba,Sc,sdBb->SB", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.v('ssbb'))
                -oe.contract("cs,ad,ba,Sc,sdbB->SB", sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.v('ssbb'))
                -2 * oe.contract("sB,cs,Sa,bc,ab->SB", sapt.h_tot('B','sb'), sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'))
                -2 * oe.contract("sb,aB,cs,ba,Sc->SB", sapt.h_tot('B','sb'), sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'))
                -2 * oe.contract("aB,cs,ba,Sc,sebe->SB", sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.v('sbbb'))
                -2 * oe.contract("aB,cs,ba,Sc,eseb->SB", sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('sa'), sapt.v('bsbb'))
                -2 * oe.contract("cB,as,ba,dc,sSbd->SB", sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('ssbb'))
                -2 * oe.contract("cB,as,da,bc,Ssbd->SB", sapt.s('ab'), sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.v('ssbb'))
                -2 * oe.contract("cs,Sa,bc,ab,seBe->SB", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
                -2 * oe.contract("cs,Sa,bc,ab,eseB->SB", sapt.s('as'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
                -2 * oe.contract("cs,da,bc,ab,SsBd->SB", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('ssbb'))
                -2 * oe.contract("cs,da,bc,ab,sSdB->SB", sapt.s('as'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.v('ssbb'))
            )
        #================== Omega(B)

        termB_OV_P_HA = 0.5*(
                +oe.contract("bR,cr,ab,Ac,erae->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('araa'))
                +oe.contract("bR,cr,ab,Ac,reea->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('raaa'))
                +oe.contract("cR,br,ab,dc,Arad->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('araa'))
                +oe.contract("cR,br,db,ac,rAad->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('raaa'))
                +oe.contract("cr,ba,Ab,ac,erRe->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('arra'))
                +oe.contract("cr,ba,Ab,ac,reeR->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('raar'))
                +oe.contract("cr,ba,db,ac,rARd->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('rara'))
                +oe.contract("cr,ba,db,ac,ArdR->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('arar'))
                +2 * oe.contract("Aa,cR,bd,ab,dc->AR", sapt.h_tot('A', 'aa'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
                +2 * oe.contract("cR,ba,db,ac,Afdf->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaaa'))
                +2 * oe.contract("cR,ba,db,ac,fAfd->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaaa'))
                +2 * oe.contract("br,cd,ab,Ac,rdaR->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('rrar'))
                +2 * oe.contract("cr,bd,ab,Ac,rdRa->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('rrra'))
                -oe.contract("cR,ba,db,ac,fAdf->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaaa'))
                -oe.contract("cR,ba,db,ac,Affd->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('aaaa'))
                -oe.contract("br,cd,ab,Ac,rdRa->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('rrra'))
                -oe.contract("cr,bd,ab,Ac,rdaR->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('rrar'))
                -2 * oe.contract("rR,cr,ba,Ab,ac->AR", sapt.h_tot('A', 'rr'), sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
                -2 * oe.contract("ra,bR,cr,ab,Ac->AR", sapt.h_tot('A', 'ra'), sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'))
                -2 * oe.contract("bR,cr,ab,Ac,reae->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('raaa'))
                -2 * oe.contract("bR,cr,ab,Ac,erea->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('araa'))
                -2 * oe.contract("cR,br,ab,dc,rAad->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('raaa'))
                -2 * oe.contract("cR,br,db,ac,Arad->AR", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('araa'))
                -2 * oe.contract("cr,ba,Ab,ac,reRe->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('rara'))
                -2 * oe.contract("cr,ba,Ab,ac,ereR->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('arar'))
                -2 * oe.contract("cr,ba,db,ac,ArRd->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('arra'))
                -2 * oe.contract("cr,ba,db,ac,rAdR->AR", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('raar'))
            )

        termB_OV_P_HB = 0.5*(
                +oe.contract("bR,As,ca,ab,dscd->AR", sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
                +oe.contract("bR,As,ca,ab,sddc->AR", sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
                +oe.contract("cR,as,ba,Ab,dscd->AR", sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
                +oe.contract("cR,as,ba,Ab,sddc->AR", sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
                +2 * oe.contract("cR,as,Ad,ba,sdbc->AR", sapt.s('br'), sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.v('ssbb'))
                +2 * oe.contract("cR,as,Ad,ba,dscb->AR", sapt.s('br'), sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.v('ssbb'))
                -oe.contract("cR,as,Ad,ba,dsbc->AR", sapt.s('br'), sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.v('ssbb'))
                -oe.contract("cR,as,Ad,ba,sdcb->AR", sapt.s('br'), sapt.s('as'), sapt.s('as'), sapt.s('ba'), sapt.v('ssbb'))
                -2 * oe.contract("sb,bR,as,ca,Ac->AR", sapt.h_tot('B', 'sb'), sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'))
                -2 * oe.contract("sb,cR,As,ba,ac->AR", sapt.h_tot('B', 'sb'), sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'))
                -2 * oe.contract("bR,As,ca,ab,sdcd->AR", sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
                -2 * oe.contract("bR,As,ca,ab,dsdc->AR", sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
                -2 * oe.contract("cR,as,ba,Ab,sdcd->AR", sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
                -2 * oe.contract("cR,as,ba,Ab,dsdc->AR", sapt.s('br'), sapt.s('as'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
            )
        # VO Blocks
        termB_VO_HA_P = 0.5*(
            +oe.contract("cA,ba,Rb,rc,eare->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aara'))
            +oe.contract("cA,ba,Rb,rc,aeer->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aaar'))
            +oe.contract("cA,ba,rb,ac,eRre->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('arra'))
            +oe.contract("cA,ba,rb,ac,Reer->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('raar'))
            +oe.contract("ba,cd,rb,Rc,adAr->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aaar'))
            +oe.contract("ca,bd,rb,Rc,adrA->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aara'))
            +oe.contract("ca,bd,ab,rc,dRAr->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('arar'))
            +oe.contract("ca,bd,ab,rc,RdrA->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('rara'))
            +2 * oe.contract("aA,ba,cd,db,Rc->RA", sapt.h_tot('A', 'aa'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
            +2 * oe.contract("cA,ba,rb,dc,aRrd->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('arrr'))
            +2 * oe.contract("cA,ba,db,rc,Rard->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('rarr'))
            +2 * oe.contract("ca,bd,ab,Rc,dgAg->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaaa'))
            +2 * oe.contract("ca,bd,ab,Rc,fdfA->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaaa'))
            -oe.contract("cA,ba,rb,dc,Rard->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('rarr'))
            -oe.contract("cA,ba,db,rc,aRrd->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('arrr'))
            -oe.contract("ca,bd,ab,Rc,fdAf->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaaa'))
            -oe.contract("ca,bd,ab,Rc,dggA->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('aaaa'))
            -2 * oe.contract("Rr,cA,ba,rb,ac->RA", sapt.h_tot('A', 'rr'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'))
            -2 * oe.contract("ar,cA,ba,Rb,rc->RA", sapt.h_tot('A', 'ar'), sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'))
            -2 * oe.contract("cA,ba,Rb,rc,aere->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aara'))
            -2 * oe.contract("cA,ba,Rb,rc,eaer->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aaar'))
            -2 * oe.contract("cA,ba,rb,ac,Rere->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('rara'))
            -2 * oe.contract("cA,ba,rb,ac,eRer->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('arar'))
            -2 * oe.contract("ba,cd,rb,Rc,adrA->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aara'))
            -2 * oe.contract("ca,bd,rb,Rc,adAr->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('rb'), sapt.s('rb'), sapt.v('aaar'))
            -2 * oe.contract("ca,bd,ab,rc,RdAr->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('raar'))
            -2 * oe.contract("ca,bd,ab,rc,dRrA->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'), sapt.v('arra'))
        )

        termB_VO_P_HA = 0.5*(
            +oe.contract("bA,cr,ab,Rc,erae->RA", sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'), sapt.v('araa'))
            +oe.contract("bA,cr,ab,Rc,reea->RA", sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'), sapt.v('raaa'))
            +oe.contract("cA,br,ab,dc,Rrad->RA", sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('rraa'))
            +oe.contract("cA,br,db,ac,rRad->RA", sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('rraa'))
            +oe.contract("cr,ba,Rb,ac,erAe->RA", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('araa'))
            +oe.contract("cr,ba,Rb,ac,reeA->RA", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('raaa'))
            +oe.contract("cr,ba,db,ac,rRAd->RA", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('rraa'))
            +oe.contract("cr,ba,db,ac,RrdA->RA", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('rraa'))
            +2 * oe.contract("Ra,cA,bd,ab,dc->RA", sapt.h_tot('A', 'ra'), sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'))
            +2 * oe.contract("cA,ba,db,ac,Rfdf->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('raaa'))
            +2 * oe.contract("cA,ba,db,ac,fRfd->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('araa'))
            +2 * oe.contract("br,cd,ab,Rc,rdaA->RA", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'), sapt.v('rraa'))
            +2 * oe.contract("cr,bd,ab,Rc,rdAa->RA", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'), sapt.v('rraa'))
            -oe.contract("cA,ba,db,ac,fRdf->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('araa'))
            -oe.contract("cA,ba,db,ac,Rffd->RA", sapt.s('ba'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('raaa'))
            -oe.contract("br,cd,ab,Rc,rdAa->RA", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'), sapt.v('rraa'))
            -oe.contract("cr,bd,ab,Rc,rdaA->RA", sapt.s('br'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'), sapt.v('rraa'))
            -2 * oe.contract("rA,cr,ba,Rb,ac->RA", sapt.h_tot('A', 'ra'), sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'))
            -2 * oe.contract("ra,bA,cr,ab,Rc->RA", sapt.h_tot('A', 'ra'), sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'))
            -2 * oe.contract("bA,cr,ab,Rc,reae->RA", sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'), sapt.v('raaa'))
            -2 * oe.contract("bA,cr,ab,Rc,erea->RA", sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('rb'), sapt.v('araa'))
            -2 * oe.contract("cA,br,ab,dc,rRad->RA", sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('rraa'))
            -2 * oe.contract("cA,br,db,ac,Rrad->RA", sapt.s('ba'), sapt.s('br'), sapt.s('ab'), sapt.s('ab'), sapt.v('rraa'))
            -2 * oe.contract("cr,ba,Rb,ac,reAe->RA", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('raaa'))
            -2 * oe.contract("cr,ba,Rb,ac,ereA->RA", sapt.s('br'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('araa'))
            -2 * oe.contract("cr,ba,db,ac,RrAd->RA", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('rraa'))
            -2 * oe.contract("cr,ba,db,ac,rRdA->RA", sapt.s('br'), sapt.s('ba'), sapt.s('ab'), sapt.s('ab'), sapt.v('rraa'))
        )

        termB_VO_HB_P = 0.5*(
            +oe.contract("sA,ba,Rb,ac,dcsd->RA", sapt.s('sa'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('bbsb'))
            +oe.contract("sA,ba,Rb,ac,cees->RA", sapt.s('sa'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('bbbs'))
            +oe.contract("bA,sa,ab,Rc,dcsd->RA", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbsb'))
            +oe.contract("bA,sa,ab,Rc,cees->RA", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbbs'))
            +2 * oe.contract("cA,sa,ab,Rd,bdsc->RA", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbss'))
            +2 * oe.contract("cA,sa,ab,Rd,dbcs->RA", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbss'))
            -oe.contract("cA,sa,ab,Rd,dbsc->RA", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbss'))
            -oe.contract("cA,sa,ab,Rd,bdcs->RA", sapt.s('sa'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbss'))
            -2 * oe.contract("bs,sA,ca,ab,Rc->RA", sapt.h_tot('B', 'bs'), sapt.s('sa'), sapt.s('ba'), sapt.s('ab'), sapt.s('rb'))
            -2 * oe.contract("bs,cA,sa,Rb,ac->RA", sapt.h_tot('B', 'bs'), sapt.s('ba'), sapt.s('sa'), sapt.s('rb'), sapt.s('ab'))
            -2 * oe.contract("sA,ba,Rb,ac,cese->RA", sapt.s('sa'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('bbsb'))
            -2 * oe.contract("sA,ba,Rb,ac,dcds->RA", sapt.s('sa'), sapt.s('ba'), sapt.s('rb'), sapt.s('ab'), sapt.v('bbbs'))
            -2 * oe.contract("bA,sa,ab,Rc,cese->RA", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbsb'))
            -2 * oe.contract("bA,sa,ab,Rc,dcds->RA", sapt.s('ba'), sapt.s('sa'), sapt.s('ab'), sapt.s('rb'), sapt.v('bbbs'))
        )

        termB_VO_P_HB = 0.5*(
            +oe.contract("bA,Rs,ca,ab,dscd->RA", sapt.s('ba'), sapt.s('rs'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
            +oe.contract("bA,Rs,ca,ab,sddc->RA", sapt.s('ba'), sapt.s('rs'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
            +oe.contract("cA,as,ba,Rb,dscd->RA", sapt.s('ba'), sapt.s('as'), sapt.s('ba'), sapt.s('rb'), sapt.v('bsbb'))
            +oe.contract("cA,as,ba,Rb,sddc->RA", sapt.s('ba'), sapt.s('as'), sapt.s('ba'), sapt.s('rb'), sapt.v('sbbb'))
            +2 * oe.contract("cA,as,Rd,ba,sdbc->RA", sapt.s('ba'), sapt.s('as'), sapt.s('rs'), sapt.s('ba'), sapt.v('ssbb'))
            +2 * oe.contract("cA,as,Rd,ba,dscb->RA", sapt.s('ba'), sapt.s('as'), sapt.s('rs'), sapt.s('ba'), sapt.v('ssbb'))
            -oe.contract("cA,as,Rd,ba,dsbc->RA", sapt.s('ba'), sapt.s('as'), sapt.s('rs'), sapt.s('ba'), sapt.v('ssbb'))
            -oe.contract("cA,as,Rd,ba,sdcb->RA", sapt.s('ba'), sapt.s('as'), sapt.s('rs'), sapt.s('ba'), sapt.v('ssbb'))
            -2 * oe.contract("sb,bA,as,ca,Rc->RA", sapt.h_tot('B', 'sb'), sapt.s('ba'), sapt.s('as'), sapt.s('ba'), sapt.s('rb'))
            -2 * oe.contract("sb,cA,Rs,ba,ac->RA", sapt.h_tot('B', 'sb'), sapt.s('ba'), sapt.s('rs'), sapt.s('ba'), sapt.s('ab'))
            -2 * oe.contract("bA,Rs,ca,ab,sdcd->RA", sapt.s('ba'), sapt.s('rs'), sapt.s('ba'), sapt.s('ab'), sapt.v('sbbb'))
            -2 * oe.contract("bA,Rs,ca,ab,dsdc->RA", sapt.s('ba'), sapt.s('rs'), sapt.s('ba'), sapt.s('ab'), sapt.v('bsbb'))
            -2 * oe.contract("cA,as,ba,Rb,sdcd->RA", sapt.s('ba'), sapt.s('as'), sapt.s('ba'), sapt.s('rb'), sapt.v('sbbb'))
            -2 * oe.contract("cA,as,ba,Rb,dsdc->RA", sapt.s('ba'), sapt.s('as'), sapt.s('ba'), sapt.s('rb'), sapt.v('bsbb'))
        )

        #====================================
        # L Blocks (Monomer A)				#
        #====================================
        lmA_OO_HP_AB = (termA_OO_HA_P + termA_OO_HB_P) 
        lmA_VV_HP_AB = (termA_VV_HA_P + termA_VV_HB_P)

        # Symmetric part for OO and VV blocks, 
        # equivalent to 1/2(PV +VP) scheme
        lmA_OO_sym_HP = 0.5*(lmA_OO_HP_AB + lmA_OO_HP_AB.T) 
        lmA_VV_sym_HP = 0.5*(lmA_VV_HP_AB + lmA_VV_HP_AB.T)

        # ========================================
        lmA_OV_sym_HP = termA_OV_HA_P + termA_OV_HB_P 
        lmA_OV_sym_PH = termA_OV_P_HA + termA_OV_P_HB 
        # Symmetric part of OV
        lmA_OV_sym = 0.5*(lmA_OV_sym_HP + lmA_OV_sym_PH)

        lmA_VO_sym_HP = termA_VO_HA_P + termA_VO_HB_P 
        lmA_VO_sym_PH = termA_VO_P_HA + termA_VO_P_HB 
        # Symmetric part of VO
        lmA_VO_sym = 0.5*(lmA_VO_sym_HP + lmA_VO_sym_PH)
        # print('LM-A(OV) ---> LM-A(VO).T', np.allclose(lmA_OV_sym, lmA_VO_sym.T))

        lmA_block = np.block([[lmA_OO_sym_HP, lmA_OV_sym],
                            [lmA_VO_sym, lmA_VV_sym_HP]])
        # print('LM-A ---> LM-A.T', np.allclose(lmA_block, lmA_block.T))
        
        #===================================== 
        # L Blocks (Monomer B)				 #
        #=====================================
        lmB_OO_HP_AB = (termB_OO_HA_P + termB_OO_HB_P) 
        lmB_VV_HP_AB = (termB_VV_HA_P + termB_VV_HB_P)

        # Symmetric part for OO and VV blocks
        lmB_OO_sym_HP = 0.5*(lmB_OO_HP_AB + lmB_OO_HP_AB.T) 
        lmB_VV_sym_HP = 0.5*(lmB_VV_HP_AB + lmB_VV_HP_AB.T)

        lmB_OV_sym_HP = termB_OV_HA_P + termB_OV_HB_P 
        lmB_OV_sym_PH = termB_OV_P_HA + termB_OV_P_HB 
        # Symmetric part of OV
        lmB_OV_sym = 0.5*(lmB_OV_sym_HP + lmB_OV_sym_PH)

        lmB_VO_sym_HP = termB_VO_HA_P + termB_VO_HB_P 
        lmB_VO_sym_PH = termB_VO_P_HA + termB_VO_P_HB 
        # Symmetric part of VO
        # ===============================================
        # CHECK: lmB_OV_sym == lmB_VO_sym ?
        lmB_VO_sym = 0.5*(lmB_VO_sym_HP + lmB_VO_sym_PH)
        # print('LM-B(OV) ---> LM-B(VO).T', np.allclose(lmB_OV_sym, lmB_VO_sym.T))

        lmB_OV_sym_HP = termB_OV_HA_P + termB_OV_HB_P 

        # lmB_block = np.block([[lmB_OO_sym_HP, lmB_OV_sym_HP],
        # 					[lmB_OV_sym_HP.T, lmB_VV_sym_HP]])

        lmB_block = np.block([[lmB_OO_sym_HP, lmB_OV_sym],
                            [lmB_VO_sym, lmB_VV_sym_HP]])
        # print('LM-B ---> LM-B.T', np.allclose(lmB_block, lmB_block.T))

        return lmA_block, lmB_block      

        

# ============================ Test ============================
if __name__ == '__main__':
    import psi4
    from pprint import pprint
    he_li_str_10 = """
        He 0.0 0.0 0.0
        --
        1 1
        Li 0.0 0.0 3.75
                            
        units bohr
                            
        symmetry c1
        no_reorient
        no_com
        """
    dimer = psi4.geometry(he_li_str_10)
    psi4.set_options({'basis': 'sto-3g', 
                      'scf_type': 'direct', 
                      'e_convergence': 1e-07})
    sapt = helper_SAPT(dimer)

    sapt_CA = sapt.wfnA.Ca()
    sapt_CB = sapt.wfnB.Ca()
  