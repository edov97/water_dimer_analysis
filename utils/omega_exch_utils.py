"""
Utilities for calculating Omega exchange operator and energy

**** Modified to test Exch Omega Blocks(w/ OO-VV blocks included)
**** Modified to use LM Terms
**** Use this in case of full Omega_exchange options

**** NOTES ****
Symmetrization schemes for all the blocks are not consistent here, partially S2
***************
"""

from time import time
import psi4
import numpy as np
import opt_einsum as oe

from utils.helper_SAPT import helper_SAPT
from utils.sinfinity import sinfinity

def form_omega_exchange_s2_sym(sapt:helper_SAPT, ca=None, cb=None, ov_vo:bool=False):
	
	""""
	Full Exchange potential evaluated under S2 approximation \\
	Uniform Symmetrization schemes for all the blocks of Omegas, 
	NOTE:1/2(PV +VP) \\
	if ov_vo bool True, returns only OV-VO block
	"""
	
	if ca is not None and cb is not None:
		sapt.set_orbitals(ca=ca, cb=cb) 

	#============================ 
	# 	OmegaB(exchange) in MO	#
	#============================   
	 
	# OO block
	omegaB_exch_OO_VP = 0.5*(
        -4 * oe.contract("sA,pb,abas->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
        -4 * oe.contract("bA,rb,pcrc->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')) 
        -4 * oe.contract("sa,ab,pbAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))  
        -2 * oe.contract("sA,pb,bs->pA", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs','A')) 
        -2 * oe.contract("sA,rb,pbrs->pA", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
        -2 * oe.contract("bA,rb,pr->pA", sapt.s('ba'), sapt.s('rb'), sapt.potential('ar','B')) 
        +2 * oe.contract("sa,pb,abAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas')) 
        +2 * oe.contract("ba,pb,aA->pA", sapt.s('ba'), sapt.s('ab'), sapt.potential('aa','B'))  
        +4 * oe.contract("ba,pb,acAc->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('abab')))
	
	# VV Block
	omegaB_exch_VV_VP =0.5*(
        -4 * oe.contract("sR,qb,abas->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('abas'))
        -4 * oe.contract("bR,rb,qcrc->qR", sapt.s('br'), sapt.s('rb'), sapt.v('rbrb')) 
        -4 * oe.contract("sa,ab,qbRs->qR", sapt.s('sa'), sapt.s('ab'), sapt.v('rbrs')) 
        -2 * oe.contract("sR,qb,bs->qR",  sapt.s('sr'), sapt.s('rb'), sapt.potential('bs','A')) 
        -2 * oe.contract("sR,rb,qbrs->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('rbrs')) 
        -2 * oe.contract("bR,rb,qr->qR",  sapt.s('br'), sapt.s('rb'), sapt.potential('rr','B'))
        +2 * oe.contract("sa,qb,abRs->qR",sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
        +2 * oe.contract("ba,qb,aR->qR",  sapt.s('ba'), sapt.s('rb'), sapt.potential('ar','B'))
        +4 * oe.contract("ba,qb,acRc->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')))
	
	# OmegaB_exch OV(VP) block
	omegaB_exch_OV_VP = 0.5*(
        -4 * oe.contract("sR,Ab,abas->AR", sapt.s('sr'), sapt.s('ab'), sapt.v('abas'))
        -4 * oe.contract("bR,rb,Acrc->AR", sapt.s('br'), sapt.s('rb'), sapt.v('abrb'))
        -4 * oe.contract("sa,ab,AbRs->AR", sapt.s('sa'), sapt.s('ab'), sapt.v('abrs'))   
        -2 * oe.contract("sR,Ab,bs->AR", sapt.s('sr'), sapt.s('ab'), sapt.potential('bs','A'))
        -2 * oe.contract("sR,rb,Abrs->AR", sapt.s('sr'), sapt.s('rb'), sapt.v('abrs'))  
        -2 * oe.contract("bR,rb,Ar->AR", sapt.s('br'), sapt.s('rb'), sapt.potential('ar','B')) 
        +2 * oe.contract("sa,Ab,abRs->AR", sapt.s('sa'), sapt.s('ab'), sapt.v('abrs'))  
        +2 * oe.contract("ba,Ab,aR->AR", sapt.s('ba'), sapt.s('ab'), sapt.potential('ar','B')) 
        +4 * oe.contract("ba,Ab,acRc->AR", sapt.s('ba'), sapt.s('ab'), sapt.v('abrb')))
	
	# OmegaB_exch OV(PV) block
	omegaB_exch_OV_PV = 0.5*(
        -4 * oe.contract("bR,As,asab->AR", sapt.s('br'), sapt.s('as'), sapt.v('asab')) 
        -4 * oe.contract("br,Ab,rcRc->AR",  sapt.s('br'), sapt.s('ab'), sapt.v('rbrb'))
        -4 * oe.contract("as,ba,AsRb->AR",  sapt.s('as'), sapt.s('ba'), sapt.v('asrb'))
        -2 * oe.contract("bR,As,sb->AR", sapt.s('br'), sapt.s('as'), sapt.potential('sb','A')) 
        -2 * oe.contract("br,As,rsRb->AR",  sapt.s('br'), sapt.s('as'), sapt.v('rsrb')) 
        -2 * oe.contract("br,Ab,rR->AR", sapt.s('br'), sapt.s('ab'), sapt.potential('rr','B'))
        +2 * oe.contract("bR,as,Asab->AR", sapt.s('br'), sapt.s('as'), sapt.v('asab')) 
        +2 * oe.contract("bR,ab,Aa->AR", sapt.s('br'), sapt.s('ab'), sapt.potential('aa','B')) 
        +4 * oe.contract("bR,ab,Acac->AR", sapt.s('br'), sapt.s('ab'), sapt.v('abab')) )
	
	# OmegaB_exch VO(VP) block
	omegaB_exch_VO_VP = 0.5*(
        -4 * oe.contract("sA,Rb,abas->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('abas'))
        -4 * oe.contract("bA,rb,Rcrc->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('rbrb')) 
        -4 * oe.contract("sa,ab,RbAs->RA", sapt.s('sa'), sapt.s('ab'), sapt.v('rbas')) 
        -2 * oe.contract("sA,Rb,bs->RA", sapt.s('sa'), sapt.s('rb'), sapt.potential('bs','A'))
        -2 * oe.contract("sA,rb,Rbrs->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('rbrs'))  
        -2 * oe.contract("bA,rb,Rr->RA", sapt.s('ba'), sapt.s('rb'), sapt.potential('rr','B'))
        +2 * oe.contract("sa,Rb,abAs->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('abas')) 
        +2 * oe.contract("ba,Rb,aA->RA", sapt.s('ba'), sapt.s('rb'), sapt.potential('aa','B'))
        +4 * oe.contract("ba,Rb,acAc->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('abab')))
	
	# OmegaB_exch VO(PV) block
	omegaB_exch_VO_PV = 0.5*(
        -4 * oe.contract("bA,Rs,asab->RA", sapt.s('ba'), sapt.s('rs'), sapt.v('asab'))
        -4 * oe.contract("br,Rb,rcAc->RA", sapt.s('br'), sapt.s('rb'), sapt.v('rbab'))
        -4 * oe.contract("as,ba,RsAb->RA", sapt.s('as'), sapt.s('ba'), sapt.v('rsab'))
        -2 * oe.contract("bA,Rs,sb->RA", sapt.s('ba'), sapt.s('rs'), sapt.potential('sb','A')) 
        -2 * oe.contract("br,Rs,rsAb->RA", sapt.s('br'), sapt.s('rs'), sapt.v('rsab')) 
        -2 * oe.contract("br,Rb,rA->RA", sapt.s('br'), sapt.s('rb'), sapt.potential('ra','B')) 
        +2 * oe.contract("bA,as,Rsab->RA", sapt.s('ba'), sapt.s('as'), sapt.v('rsab')) 
        +2 * oe.contract("bA,ab,Ra->RA", sapt.s('ba'), sapt.s('ab'), sapt.potential('ra','B'))
        +4 * oe.contract("bA,ab,Rcac->RA", sapt.s('ba'), sapt.s('ab'), sapt.v('rbab')))
    
	#============================ 
	# 	OmegaA(exchange) in MO	#
	#============================   
	# OO Block
	omegaA_exch_OO_VP =0.5*(
        -4 * oe.contract("rp,Ba,abrb->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('abrb')) 
        -4 * oe.contract("ap,sa,cBcs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('abas'))
        -4 * oe.contract("ba,rb,aBrp->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
        -2 * oe.contract("rp,Ba,ar->pB", sapt.s('rb'), sapt.s('ba'), sapt.potential('ar','B')) 
        -2 * oe.contract("rp,sa,aBrs->pB",sapt.s('rb'), sapt.s('sa'), sapt.v('abrs')) 
        -2 * oe.contract("ap,sa,Bs->pB", sapt.s('ab'), sapt.s('sa'), sapt.potential('bs','A'))
        +2 * oe.contract("Ba,rb,abrp->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')) 
        +2 * oe.contract("Ba,ab,bp->pB", sapt.s('ba'), sapt.s('ab'), sapt.potential('bb','A')) 
        +4 * oe.contract("Ba,ab,cbcp->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('abab')))
	
	# VV Block 
	omegaA_exch_VV_VP =0.5*(
        -4 * oe.contract("rq,Sa,abrb->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('abrb'))
        -4 * oe.contract("aq,sa,cScs->qS", sapt.s('as'), sapt.s('sa'), sapt.v('asas')) 
        -4 * oe.contract("ba,rb,aSrq->qS", sapt.s('ba'), sapt.s('rb'), sapt.v('asrs'))
        -2 * oe.contract("rq,Sa,ar->qS", sapt.s('rs'), sapt.s('sa'), sapt.potential('ar','B')) 
        -2 * oe.contract("rq,sa,aSrs->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('asrs'))
        -2 * oe.contract("aq,sa,Ss->qS", sapt.s('as'), sapt.s('sa'), sapt.potential('ss','A')) 
        +2 * oe.contract("Sa,rb,abrq->qS", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
        +2 * oe.contract("Sa,ab,bq->qS", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs','A')) 
        +4 * oe.contract("Sa,ab,cbcq->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('abas')))
	
	# omegaA exchange(OV) block
	omegaA_exch_OV_VP =0.5*(
        -4 * oe.contract("rS,Ba,abrb->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('abrb')) 
        -4 * oe.contract("aS,sa,cBcs->BS", sapt.s('as'), sapt.s('sa'), sapt.v('abas')) 
        -4 * oe.contract("ba,rb,aBrS->BS", sapt.s('ba'), sapt.s('rb'), sapt.v('abrs')) 
        -2 * oe.contract("rS,Ba,ar->BS", sapt.s('rs'), sapt.s('ba'), sapt.potential('ar', 'B')) 
        -2 * oe.contract("rS,sa,aBrs->BS", sapt.s('rs'), sapt.s('sa'), sapt.v('abrs'))  
        -2 * oe.contract("aS,sa,Bs->BS", sapt.s('as'), sapt.s('sa'), sapt.potential('bs', 'A')) 
        +2 * oe.contract("Ba,rb,abrS->BS", sapt.s('ba'), sapt.s('rb'), sapt.v('abrs')) 
        +2 * oe.contract("Ba,ab,bS->BS", sapt.s('ba'), sapt.s('ab'), sapt.potential('bs', 'A')) 
        +4 * oe.contract("Ba,ab,cbcS->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('abas')))
	
	omegaA_exch_OV_PV =0.5*(
        -4 * oe.contract("aS,Br,rbab->BS", sapt.s('as'), sapt.s('br'), sapt.v('rbab'))
        -4 * oe.contract("br,ab,rBaS->BS", sapt.s('br'), sapt.s('ab'), sapt.v('rbas')) 
        -4 * oe.contract("as,Ba,cscS->BS", sapt.s('as'), sapt.s('ba'), sapt.v('asas')) 
        -2 * oe.contract("aS,Br,ra->BS", sapt.s('as'), sapt.s('br'), sapt.potential('ra', 'B'))  
        -2 * oe.contract("Br,as,rsaS->BS", sapt.s('br'), sapt.s('as'), sapt.v('rsas')) 
        -2 * oe.contract("as,Ba,sS->BS", sapt.s('as'), sapt.s('ba'), sapt.potential('ss', 'A')) 
        +2 * oe.contract("aS,br,rBab->BS", sapt.s('as'), sapt.s('br'), sapt.v('rbab'))
        +2 * oe.contract("aS,ba,Bb->BS", sapt.s('as'), sapt.s('ba'), sapt.potential('bb', 'A')) 
        +4 * oe.contract("aS,ba,cBcb->BS", sapt.s('as'), sapt.s('ba'), sapt.v('abab')))
	
	omegaA_exch_VO_VP =0.5*(
        -4 * oe.contract("rB,Sa,abrb->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('abrb')) 
        -4 * oe.contract("aB,sa,cScs->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('asas')) 
        -4 * oe.contract("ba,rb,aSrB->SB", sapt.s('ba'), sapt.s('rb'), sapt.v('asrb')) 
        -2 * oe.contract("rB,Sa,ar->SB", sapt.s('rb'), sapt.s('sa'), sapt.potential('ar', 'B')) 
        -2 * oe.contract("rB,sa,aSrs->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('asrs'))
        -2 * oe.contract("aB,sa,Ss->SB", sapt.s('ab'), sapt.s('sa'), sapt.potential('ss', 'A')) 
        +2 * oe.contract("Sa,rb,abrB->SB", sapt.s('sa'), sapt.s('rb'), sapt.v('abrb')) 
        +2 * oe.contract("Sa,ab,bB->SB", sapt.s('sa'), sapt.s('ab'), sapt.potential('bb', 'A')) 
        +4 * oe.contract("Sa,ab,cbcB->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('abab')))
	
	omegaA_exch_VO_PV =0.5*(
        -4 * oe.contract("aB,Sr,rbab->SB", sapt.s('ab'), sapt.s('sr'), sapt.v('rbab')) 
        -4 * oe.contract("br,ab,rSaB->SB", sapt.s('br'), sapt.s('ab'), sapt.v('rsab'))
        -4 * oe.contract("as,Sa,cscB->SB", sapt.s('as'), sapt.s('sa'), sapt.v('asab'))
        -2 * oe.contract("aB,Sr,ra->SB", sapt.s('ab'), sapt.s('sr'), sapt.potential('ra', 'B')) 
        -2 * oe.contract("Sr,as,rsaB->SB", sapt.s('sr'), sapt.s('as'), sapt.v('rsab')) 
        -2 * oe.contract("as,Sa,sB->SB", sapt.s('as'), sapt.s('sa'), sapt.potential('sb', 'A')) 
        +2 * oe.contract("aB,br,rSab->SB", sapt.s('ab'), sapt.s('br'), sapt.v('rsab')) 
        +2 * oe.contract("aB,ba,Sb->SB", sapt.s('ab'), sapt.s('ba'), sapt.potential('sb', 'A')) 
        +4 * oe.contract("aB,ba,cScb->SB", sapt.s('ab'), sapt.s('ba'), sapt.v('asab')))       

	#  Omegas symmetrized
	# 1/2(VP +PV)
	omegaB_exch_OO_sym = 0.5*(omegaB_exch_OO_VP + omegaB_exch_OO_VP.T)
	omegaB_exch_OV_sym = 0.5*(omegaB_exch_OV_VP + omegaB_exch_OV_PV)    
	omegaB_exch_VO_sym = 0.5*(omegaB_exch_VO_VP + omegaB_exch_VO_PV)
	omegaB_exch_VV_sym = 0.5*(omegaB_exch_VV_VP + omegaB_exch_VV_VP.T)

	omegaA_exch_OO_sym = 0.5*(omegaA_exch_OO_VP + omegaA_exch_OO_VP.T)
	omegaA_exch_OV_sym = 0.5*(omegaA_exch_OV_VP + omegaA_exch_OV_PV)
	omegaA_exch_VO_sym = 0.5*(omegaA_exch_VO_VP + omegaA_exch_VO_PV)
	omegaA_exch_VV_sym = 0.5*(omegaA_exch_VV_VP + omegaA_exch_VV_VP.T)

	# omegaB_exch_OO_sym = (omegaB_exch_OO_VP + omegaB_exch_OO_VP.T)
	# omegaB_exch_OV_sym = (omegaB_exch_OV_VP + omegaB_exch_OV_PV)    
	# omegaB_exch_VO_sym = (omegaB_exch_VO_VP + omegaB_exch_VO_PV)
	# omegaB_exch_VV_sym = (omegaB_exch_VV_VP + omegaB_exch_VV_VP.T)

	# omegaA_exch_OO_sym = (omegaA_exch_OO_VP + omegaA_exch_OO_VP.T)
	# omegaA_exch_OV_sym = (omegaA_exch_OV_VP + omegaA_exch_OV_PV)
	# omegaA_exch_VO_sym = (omegaA_exch_VO_VP + omegaA_exch_VO_PV)
	# omegaA_exch_VV_sym = (omegaA_exch_VV_VP + omegaA_exch_VV_VP.T)

	omegaB_exch_sym = np.block([[omegaB_exch_OO_sym, omegaB_exch_OV_sym],
                               [omegaB_exch_VO_sym, omegaB_exch_VV_sym]])

	omegaA_exch_sym = np.block([[omegaA_exch_OO_sym, omegaA_exch_OV_sym],
                               [omegaA_exch_VO_sym, omegaA_exch_VV_sym]])
	
	if ov_vo:
		# print('Getting OV-VO only')
		### OmegaB_exch in MO
		nvirtA = sapt.nmo - sapt.ndocc_A
		omegaB_exch_aa = np.zeros((sapt.ndocc_A, sapt.ndocc_A))
		omegaB_exch_rr = np.zeros((nvirtA, nvirtA))
		omegaB_exch_MO= np.block([[omegaB_exch_aa, omegaB_exch_OV_sym],
								[omegaB_exch_VO_sym, omegaB_exch_rr]])
		omegaB_exch_sym = omegaB_exch_MO
		# print(omegaB_exch_sym)

		### OmegaA_exch in MO
		nvirtB = sapt.nmo - sapt.ndocc_B
		omegaA_exch_bb = np.zeros((sapt.ndocc_B, sapt.ndocc_B))
		omegaA_exch_ss = np.zeros((nvirtB, nvirtB))
		omegaA_exch_MO = np.block([[omegaA_exch_bb, omegaA_exch_OV_sym],
							 	[omegaA_exch_VO_sym, omegaA_exch_ss]])
		omegaA_exch_sym = omegaA_exch_MO

	return omegaA_exch_sym, omegaB_exch_sym 

def form_omega_exchange_sinf_sym(sapt:helper_SAPT, ca=None, cb=None, ov_vo:bool=False, s4_bool:bool=False):
	if ca is not None and cb is not None:
		sapt.set_orbitals(ca=ca, cb=cb)

	sinf = sinfinity(sapt)

	# Constructing symmeytric blocks of OV-VO, Non-S2
	omegaA_exch_bs = sinf.omega_exchA_bs
	omegaA_exch_sb = sinf.omega_exchA_sb
	omegaB_exch_ar = sinf.omega_exchB_ar
	omegaB_exch_ra = sinf.omega_exchB_ra
	
	if not ov_vo:
		#  S2 Blocks
		(omegaA_exch_OO_VP_s2, 
   		omegaA_exch_VV_VP_s2, 
		omegaB_exch_OO_VP_s2, 
		omegaB_exch_VV_VP_s2) = form_omega_exchange_off_diag_s2(sapt, ca, cb)

		omegaA_exch_OO_VP = omegaA_exch_OO_VP_s2
		omegaA_exch_VV_VP = omegaA_exch_VV_VP_s2

		omegaB_exch_OO_VP = omegaB_exch_OO_VP_s2
		omegaB_exch_VV_VP = omegaB_exch_VV_VP_s2

		#  S4 Blocks
		if s4_bool:
			print('Adding S4 into the omega-exchange terms')
			(omegaA_exch_OO_VP_s4, 
			omegaA_exch_VV_VP_s4, 
			omegaB_exch_OO_VP_s4, 
			omegaB_exch_VV_VP_s4) = form_omega_exchange_off_diag_s4(sapt, ca, cb)

			omegaA_exch_OO_VP += omegaA_exch_OO_VP_s4
			omegaA_exch_VV_VP += omegaA_exch_VV_VP_s4

			omegaB_exch_OO_VP += omegaB_exch_OO_VP_s4
			omegaB_exch_VV_VP += omegaB_exch_VV_VP_s4

		# # OO Block
		# omegaA_exch_OO_VP =0.5*(
		# 	-4 * oe.contract("rp,Ba,abrb->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('abrb')) 
		# 	-4 * oe.contract("ap,sa,cBcs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('abas'))
		# 	-4 * oe.contract("ba,rb,aBrp->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
		# 	-2 * oe.contract("rp,Ba,ar->pB", sapt.s('rb'), sapt.s('ba'), sapt.potential('ar','B')) 
		# 	-2 * oe.contract("rp,sa,aBrs->pB",sapt.s('rb'), sapt.s('sa'), sapt.v('abrs')) 
		# 	-2 * oe.contract("ap,sa,Bs->pB", sapt.s('ab'), sapt.s('sa'), sapt.potential('bs','A'))
		# 	+2 * oe.contract("Ba,rb,abrp->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')) 
		# 	+2 * oe.contract("Ba,ab,bp->pB", sapt.s('ba'), sapt.s('ab'), sapt.potential('bb','A')) 
		# 	+4 * oe.contract("Ba,ab,cbcp->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('abab')))
		
		# # VV Block 
		# omegaA_exch_VV_VP =0.5*(
		# 	-4 * oe.contract("rq,Sa,abrb->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('abrb'))
		# 	-4 * oe.contract("aq,sa,cScs->qS", sapt.s('as'), sapt.s('sa'), sapt.v('asas')) 
		# 	-4 * oe.contract("ba,rb,aSrq->qS", sapt.s('ba'), sapt.s('rb'), sapt.v('asrs'))
		# 	-2 * oe.contract("rq,Sa,ar->qS", sapt.s('rs'), sapt.s('sa'), sapt.potential('ar','B')) 
		# 	-2 * oe.contract("rq,sa,aSrs->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('asrs'))
		# 	-2 * oe.contract("aq,sa,Ss->qS", sapt.s('as'), sapt.s('sa'), sapt.potential('ss','A')) 
		# 	+2 * oe.contract("Sa,rb,abrq->qS", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
		# 	+2 * oe.contract("Sa,ab,bq->qS", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs','A')) 
		# 	+4 * oe.contract("Sa,ab,cbcq->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('abas')))

		# # OO block
		# omegaB_exch_OO_VP = 0.5*(
		# 	-4 * oe.contract("sA,pb,abas->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
		# 	-4 * oe.contract("bA,rb,pcrc->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')) 
		# 	-4 * oe.contract("sa,ab,pbAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))  
		# 	-2 * oe.contract("sA,pb,bs->pA", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs','A')) 
		# 	-2 * oe.contract("sA,rb,pbrs->pA", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
		# 	-2 * oe.contract("bA,rb,pr->pA", sapt.s('ba'), sapt.s('rb'), sapt.potential('ar','B')) 
		# 	+2 * oe.contract("sa,pb,abAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas')) 
		# 	+2 * oe.contract("ba,pb,aA->pA", sapt.s('ba'), sapt.s('ab'), sapt.potential('aa','B'))  
		# 	+4 * oe.contract("ba,pb,acAc->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('abab')))
		
		# # VV Block
		# omegaB_exch_VV_VP =0.5*(
		# 	-4 * oe.contract("sR,qb,abas->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('abas'))
		# 	-4 * oe.contract("bR,rb,qcrc->qR", sapt.s('br'), sapt.s('rb'), sapt.v('rbrb')) 
		# 	-4 * oe.contract("sa,ab,qbRs->qR", sapt.s('sa'), sapt.s('ab'), sapt.v('rbrs')) 
		# 	-2 * oe.contract("sR,qb,bs->qR",  sapt.s('sr'), sapt.s('rb'), sapt.potential('bs','A')) 
		# 	-2 * oe.contract("sR,rb,qbrs->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('rbrs')) 
		# 	-2 * oe.contract("bR,rb,qr->qR",  sapt.s('br'), sapt.s('rb'), sapt.potential('rr','B'))
		# 	+2 * oe.contract("sa,qb,abRs->qR",sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
		# 	+2 * oe.contract("ba,qb,aR->qR",  sapt.s('ba'), sapt.s('rb'), sapt.potential('ar','B'))
		# 	+4 * oe.contract("ba,qb,acRc->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')))

		# Symmetrization with 0.5*(VP +PV) scheme
		# ==========================================	
		omegaA_exch_OO_sym = 0.5*(omegaA_exch_OO_VP + omegaA_exch_OO_VP.T)
		omegaA_exch_OV_sym = 0.5*(omegaA_exch_bs + omegaA_exch_sb.T)
		omegaA_exch_VO_sym = omegaA_exch_OV_sym.T
		omegaA_exch_VV_sym = 0.5*(omegaA_exch_VV_VP + omegaA_exch_VV_VP.T)

		omegaB_exch_OO_sym = 0.5*(omegaB_exch_OO_VP + omegaB_exch_OO_VP.T)
		omegaB_exch_OV_sym = 0.5*(omegaB_exch_ar + omegaB_exch_ra.T)
		omegaB_exch_VO_sym = omegaB_exch_OV_sym.T
		omegaB_exch_VV_sym = 0.5*(omegaB_exch_VV_VP + omegaB_exch_VV_VP.T)

	else:
		# Only OV-VO blocks are non-zero, Non-S2
		### OmegaA_exch in MO
		nvirtB = sapt.nmo - sapt.ndocc_B
		omegaA_exch_OO_sym = np.zeros((sapt.ndocc_B, sapt.ndocc_B))
		omegaA_exch_VV_sym = np.zeros((nvirtB, nvirtB))
		omegaA_exch_OV_sym = 0.5*(omegaA_exch_bs + omegaA_exch_sb.T)
		omegaA_exch_VO_sym = omegaA_exch_OV_sym.T

		### OmegaB_exch in MO
		nvirtA = sapt.nmo - sapt.ndocc_A
		omegaB_exch_OO_sym = np.zeros((sapt.ndocc_A, sapt.ndocc_A))
		omegaB_exch_VV_sym = np.zeros((nvirtA, nvirtA))
		omegaB_exch_OV_sym = 0.5*(omegaB_exch_ar + omegaB_exch_ra.T)
		omegaB_exch_VO_sym = omegaB_exch_OV_sym.T


	omegaB_exch_sym = np.block([[omegaB_exch_OO_sym, omegaB_exch_OV_sym],
                               [omegaB_exch_VO_sym, omegaB_exch_VV_sym]])

	omegaA_exch_sym = np.block([[omegaA_exch_OO_sym, omegaA_exch_OV_sym],
                               [omegaA_exch_VO_sym, omegaA_exch_VV_sym]])

	return omegaA_exch_sym, omegaB_exch_sym

def form_omega_exchange_s2(sapt:helper_SAPT, ca=None, cb=None):
	"""
	Only Non-zero OV-VO blocks\\
	Exchange under S2 approximation\\
	Calculates the Omega with helper_sapt and returns Omega exchange(OV-VO block) formed by the symmetric blocks
	- (OV-VO, S2) non-zero blocks
	- (OO-VV) zero blocks
	"""

	if ca is not None and cb is not None:
		sapt.set_orbitals(ca=ca, cb=cb)

	# Symmetric Block of OmegaB_exchange
	# Calculation of omegaB exchange
	omegaB_exch_ar = (-2 * oe.contract("sR,Ab,abas->AR", sapt.s('sr'), sapt.s('ab'), sapt.v('abas'))
				- 2 * oe.contract("bR,rb,Acrc->AR", sapt.s('br'), sapt.s('rb'), sapt.v('abrb'))
				- 2 * oe.contract("sa,ab,AbRs->AR", sapt.s('sa'), sapt.s('ab'), sapt.v('abrs'))
				- oe.contract("sR,Ab,bs->AR",sapt.s('sr'), sapt.s('ab'), sapt.potential('bs','A'))
				- oe.contract("sR,rb,Abrs->AR",sapt.s('sr'), sapt.s('rb'), sapt.v('abrs'))
				- oe.contract("bR,rb,Ar->AR",sapt.s('br'), sapt.s('rb'), sapt.potential('ar','B'))
				+ oe.contract("sa,Ab,abRs->AR", sapt.s('sa'), sapt.s('ab'), sapt.v('abrs'))
				+ oe.contract("ba,Ab,aR->AR",sapt.s('ba'), sapt.s('ab'), sapt.potential('ar','B'))
				+ 2*oe.contract("ba,Ab,acRc->AR", sapt.s('ba'), sapt.s('ab'), sapt.v('abrb')))

	# Symmetric Block of OmegaA_exchange
	# Calculation of omegaA exchange
	omegaA_exch_bs = (-2 * oe.contract("rS,Ba,abrb->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('abrb'))
				- 2 * oe.contract("aS,sa,cBcs->BS", sapt.s('as'), sapt.s('sa'), sapt.v('abas'))
				- 2 * oe.contract("ba,rb,aBrS->BS", sapt.s('ba'), sapt.s('rb'), sapt.v('abrs'))
				- oe.contract("rS,Ba,ar->BS", sapt.s('rs'), sapt.s('ba'), sapt.potential('ar','B'))
				- oe.contract("rS,sa,aBrs->BS", sapt.s('rs'), sapt.s('sa'), sapt.v('abrs'))
				- oe.contract("aS,sa,Bs->BS", sapt.s('as'), sapt.s('sa'), sapt.potential('bs','A'))
				+ oe.contract("Ba,rb,abrS->BS", sapt.s('ba'), sapt.s('rb'), sapt.v('abrs'))
				+ oe.contract("Ba,ab,bS->BS",sapt.s('ba'), sapt.s('ab'), sapt.potential('bs','A'))
				+ 2*oe.contract("Ba,ab,cbcS->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('abas')))

	### OmegaB_exch in MO
	nvirtA = sapt.nmo - sapt.ndocc_A
	omegaB_exch_aa = np.zeros((sapt.ndocc_A, sapt.ndocc_A))
	omegaB_exch_rr = np.zeros((nvirtA, nvirtA))
	omegaB_exch_MO = np.block([[omegaB_exch_aa, omegaB_exch_ar],[omegaB_exch_ar.T, omegaB_exch_rr]])

	### OmegaA_exch in MO
	nvirtB = sapt.nmo - sapt.ndocc_B
	omegaA_exch_bb = np.zeros((sapt.ndocc_B, sapt.ndocc_B))
	omegaA_exch_ss = np.zeros((nvirtB, nvirtB))
	omegaA_exch_MO = np.block([[omegaA_exch_bb, omegaA_exch_bs],[omegaA_exch_bs.T, omegaA_exch_ss]])
	return omegaA_exch_MO, omegaB_exch_MO

def form_omega_exchange_s2_total(sapt:helper_SAPT, ca=None, cb=None, sym_blocks:bool=True):
	
	""""
	==== TEST to add OO-VV blocks ============\\
	Exchange under S2 approximation \\
	Calculates the Omega with helper_sapt and returns Omega exchange(all blocks) \\
	NOTE: OO-VV blocks with general definition(1/2(OO/VV + OO/VV.T)) in S2 \\
	OV-VO blocks formed by the symmetric blocks, taking transpose 

	- If sym_blocks is True(default), returns the symmetrized blocks.
	- If sym_blocks is False, only for TEST
	"""
	
	if ca is not None and cb is not None:
		sapt.set_orbitals(ca=ca, cb=cb) 

	# OmegaB(exchange) in MO
	# ===========================    
	# OO block
	omegaB_exch_aa = 0.5*(
		-4 * oe.contract("sA,pb,abas->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
		-4 * oe.contract("bA,rb,pcrc->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
		-4 * oe.contract("sa,ab,pbAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
		-2 * oe.contract("sA,pb,bs->pA", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs', 'A'))
		-2 * oe.contract("sA,rb,pbrs->pA", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs'))
		-2 * oe.contract("bA,rb,pr->pA", sapt.s('ba'), sapt.s('rb'), sapt.potential('ar', 'B'))
		+2 * oe.contract("sa,pb,abAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
		+2 * oe.contract("ba,pb,aA->pA", sapt.s('ba'), sapt.s('ab'), sapt.potential('aa', 'B'))
		+4 * oe.contract("ba,pb,acAc->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('abab'))
	)
	# VV Block
	omegaB_exch_rr = 0.5*(
		-4 * oe.contract("sR,qb,abas->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('abas'))
		-4 * oe.contract("bR,rb,qcrc->qR", sapt.s('br'), sapt.s('rb'), sapt.v('rbrb'))
		-4 * oe.contract("sa,ab,qbRs->qR", sapt.s('sa'), sapt.s('ab'), sapt.v('rbrs'))
		-2 * oe.contract("sR,qb,bs->qR", sapt.s('sr'), sapt.s('rb'), sapt.potential('bs', 'A'))
		-2 * oe.contract("sR,rb,qbrs->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('rbrs'))
		-2 * oe.contract("bR,rb,qr->qR", sapt.s('br'), sapt.s('rb'), sapt.potential('rr', 'B'))
		+2 * oe.contract("sa,qb,abRs->qR", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs'))
		+2 * oe.contract("ba,qb,aR->qR", sapt.s('ba'), sapt.s('rb'), sapt.potential('ar', 'B'))
		+4 * oe.contract("ba,qb,acRc->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
	)
	# OmegaB_exch(OV) block
	# Validated against SAPT ExchInd = CPHF_B(ra)*omegaB_exch(ar)
	omegaB_exch_ar = (-2 * oe.contract("sR,Ab,abas->AR", sapt.s('sr'), sapt.s('ab'), sapt.v('abas'))
				- 2 * oe.contract("bR,rb,Acrc->AR", sapt.s('br'), sapt.s('rb'), sapt.v('abrb'))
				- 2 * oe.contract("sa,ab,AbRs->AR", sapt.s('sa'), sapt.s('ab'), sapt.v('abrs'))
				- oe.contract("sR,Ab,bs->AR",sapt.s('sr'), sapt.s('ab'), sapt.potential('bs','A'))
				- oe.contract("sR,rb,Abrs->AR",sapt.s('sr'), sapt.s('rb'), sapt.v('abrs'))
				- oe.contract("bR,rb,Ar->AR",sapt.s('br'), sapt.s('rb'), sapt.potential('ar','B'))
				+ oe.contract("sa,Ab,abRs->AR", sapt.s('sa'), sapt.s('ab'), sapt.v('abrs'))
				+ oe.contract("ba,Ab,aR->AR",sapt.s('ba'), sapt.s('ab'), sapt.potential('ar','B'))
				+ 2*oe.contract("ba,Ab,acRc->AR", sapt.s('ba'), sapt.s('ab'), sapt.v('abrb')))
	
	# VO block
	omegaB_exch_ra = 0.5*(-4 * oe.contract("sA,Rb,abas->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('abas'))
						-4 * oe.contract("bA,rb,Rcrc->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('rbrb'))
						-4 * oe.contract("sa,ab,RbAs->RA", sapt.s('sa'), sapt.s('ab'), sapt.v('rbas'))
						-2 * oe.contract("sA,Rb,bs->RA", sapt.s('sa'), sapt.s('rb'), sapt.potential('bs', 'A'))
						-2 * oe.contract("sA,rb,Rbrs->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('rbrs'))
						-2 * oe.contract("bA,rb,Rr->RA",  sapt.s('ba'), sapt.s('rb'), sapt.potential('rr', 'B'))
						+2 * oe.contract("sa,Rb,abAs->RA", sapt.s('sa'), sapt.s('rb'), sapt.v('abas'))
						+2 * oe.contract("ba,Rb,aA->RA", sapt.s('ba'), sapt.s('rb'), sapt.potential('aa', 'B'))
						+4 * oe.contract("ba,Rb,acAc->RA", sapt.s('ba'), sapt.s('rb'), sapt.v('abab')))
	
	# OmegaA(Exchange) in MO
	# =============================
	# OO Block
	omegaA_exch_bb = 0.5*(
		  -4 * oe.contract("rB,pa,abrb->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('abrb'))
		-4 * oe.contract("aB,sa,cpcs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('abas'))
		-4 * oe.contract("ba,rb,aprB->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
		-2 * oe.contract("rB,pa,ar->pB", sapt.s('rb'), sapt.s('ba'), sapt.potential('ar','B'))
		-2 * oe.contract("rB,sa,aprs->pB", sapt.s('rb'), sapt.s('sa'), sapt.v('abrs'))
		-2 * oe.contract("aB,sa,ps->pB", sapt.s('ab'), sapt.s('sa'), sapt.potential('bs','A'))
		+2 * oe.contract("pa,rb,abrB->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')) 
		+2 * oe.contract("pa,ab,bB->pB", sapt.s('ba'), sapt.s('ab'), sapt.potential('bb','A'))
		+4 * oe.contract("pa,ab,cbcB->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('abab'))
	)   
	# VV Block 
	omegaA_exch_ss = 0.5*(
		-4 * oe.contract("rS,qa,abrb->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('abrb'))
		-4 * oe.contract("aS,sa,cqcs->qS", sapt.s('as'), sapt.s('sa'), sapt.v('asas'))
		-4 * oe.contract("ba,rb,aqrS->qS", sapt.s('ba'), sapt.s('rb'), sapt.v('asrs'))
		-2 * oe.contract("rS,qa,ar->qS", sapt.s('rs'), sapt.s('sa'), sapt.potential('ar','B'))
		-2 * oe.contract("rS,sa,aqrs->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('asrs'))
		-2 * oe.contract("aS,sa,qs->qS", sapt.s('as'), sapt.s('sa'), sapt.potential('ss','A'))
		+2 * oe.contract("qa,rb,abrS->qS", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs'))
		+2 * oe.contract("qa,ab,bS->qS", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs','A'))
		+4 * oe.contract("qa,ab,cbcS->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
	)       
	# omegaA exchange(OV) block
	# Validated against SAPT ExchInd = CPHF_A(sb)*omegaB_exch(bs)
	omegaA_exch_bs = (-2 * oe.contract("rS,Ba,abrb->BS", sapt.s('rs'), sapt.s('ba'), sapt.v('abrb'))
				- 2 * oe.contract("aS,sa,cBcs->BS", sapt.s('as'), sapt.s('sa'), sapt.v('abas'))
				- 2 * oe.contract("ba,rb,aBrS->BS", sapt.s('ba'), sapt.s('rb'), sapt.v('abrs'))
				- oe.contract("rS,Ba,ar->BS", sapt.s('rs'), sapt.s('ba'), sapt.potential('ar','B'))
				- oe.contract("rS,sa,aBrs->BS", sapt.s('rs'), sapt.s('sa'), sapt.v('abrs'))
				- oe.contract("aS,sa,Bs->BS", sapt.s('as'), sapt.s('sa'), sapt.potential('bs','A'))
				+ oe.contract("Ba,rb,abrS->BS", sapt.s('ba'), sapt.s('rb'), sapt.v('abrs'))
				+ oe.contract("Ba,ab,bS->BS",sapt.s('ba'), sapt.s('ab'), sapt.potential('bs','A'))
				+ 2*oe.contract("Ba,ab,cbcS->BS", sapt.s('ba'), sapt.s('ab'), sapt.v('abas')))
	# VO Block
	omegaA_exch_sb = 0.5*(
				-4 * oe.contract("rB,Sa,abrb->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('abrb'))
				-4 * oe.contract("aB,sa,cScs->SB", sapt.s('ab'), sapt.s('sa'), sapt.v('asas'))
				-4 * oe.contract("ba,rb,aSrB->SB", sapt.s('ba'), sapt.s('rb'), sapt.v('asrb'))
				-2 * oe.contract("rB,Sa,ar->SB", sapt.s('rb'), sapt.s('sa'), sapt.potential('ar','B'))
				-2 * oe.contract("rB,sa,aSrs->SB", sapt.s('rb'), sapt.s('sa'), sapt.v('asrs'))
				-2 * oe.contract("aB,sa,Ss->SB", sapt.s('ab'), sapt.s('sa'), sapt.potential('ss','A'))
				+2 * oe.contract("Sa,rb,abrB->SB", sapt.s('sa'), sapt.s('rb'), sapt.v('abrb'))
				+2 * oe.contract("Sa,ab,bB->SB", sapt.s('sa'), sapt.s('ab'), sapt.potential('bb','A'))
				+4 * oe.contract("Sa,ab,cbcB->SB", sapt.s('sa'), sapt.s('ab'), sapt.v('abab'))
	)   

	#  Omegas evaluated
	### OmegaB_exch in MO
	omegaB_exch_MO_without_sym = np.block([[omegaB_exch_aa, omegaB_exch_ar],[omegaB_exch_ra, omegaB_exch_rr]])
	### OmegaA_exch in MO
	omegaA_exch_MO_without_sym = np.block([[omegaA_exch_bb, omegaA_exch_bs],[omegaA_exch_sb, omegaA_exch_ss]])

	# Omega OO-VV symmetrized
	# OV(validated with ExchInd) replacing the VO blocks
	omegaA_exch_bb_sym = 0.5*(omegaA_exch_bb + omegaA_exch_bb.T)
	omegaA_exch_ss_sym = 0.5*(omegaA_exch_ss + omegaA_exch_ss.T)

	omegaB_exch_aa_sym = 0.5*(omegaB_exch_aa + omegaB_exch_aa.T)
	omegaB_exch_rr_sym = 0.5*(omegaB_exch_rr + omegaB_exch_rr.T)

	### OmegaA_exch in MO
	omegaB_exch_MO_sym = np.block([[omegaB_exch_aa_sym, omegaB_exch_ar],[omegaB_exch_ar.T, omegaB_exch_rr_sym]])
	omegaA_exch_MO_sym = np.block([[omegaA_exch_bb_sym, omegaA_exch_bs],[omegaA_exch_bs.T, omegaA_exch_ss_sym]])

	if sym_blocks:
		omegaA_exch_MO = omegaA_exch_MO_sym
		omegaB_exch_MO = omegaB_exch_MO_sym
	else:
		omegaA_exch_MO = omegaA_exch_MO_without_sym
		omegaB_exch_MO = omegaB_exch_MO_without_sym  
	return omegaA_exch_MO, omegaB_exch_MO

def form_omega_exchange_sinf(sapt:helper_SAPT, ca=None, cb=None):
	   
	"""
	Only Non-zero OV-VO blocks\\
	Exchange under Non-S2 approximation\\
	Calculates the Omega with helper_sapt and returns Omega exchange(OV-VO block) formed by the symmetric blocks
	- (OV-VO, Non-S2) non-zero blocks
	- (OO-VV) zero blocks
	"""

	if ca is not None and cb is not None:
		sapt.set_orbitals(ca=ca, cb=cb)

	sinf = sinfinity(sapt)
	omegaA_exch_bs = sinf.omega_exchA_bs
	omegaB_exch_ar = sinf.omega_exchB_ar

	### OmegaB_exch in MO
	nvirtA = sapt.nmo - sapt.ndocc_A
	omegaB_exch_aa = np.zeros((sapt.ndocc_A, sapt.ndocc_A))
	omegaB_exch_rr = np.zeros((nvirtA, nvirtA))
	omegaB_exch_MO = np.block([[omegaB_exch_aa, omegaB_exch_ar],[omegaB_exch_ar.T, omegaB_exch_rr]])

	### OmegaA_exch in MO
	nvirtB = sapt.nmo - sapt.ndocc_B
	omegaA_exch_bb = np.zeros((sapt.ndocc_B, sapt.ndocc_B))
	omegaA_exch_ss = np.zeros((nvirtB, nvirtB))
	omegaA_exch_MO = np.block([[omegaA_exch_bb, omegaA_exch_bs],[omegaA_exch_bs.T, omegaA_exch_ss]])
	return omegaA_exch_MO, omegaB_exch_MO

def form_omega_exchange_sinf_total(sapt:helper_SAPT, ca=None, cb=None):
	""""
	==== TEST to add OO-VV blocks ============\\
	Exchange with partially Non-S2(OV-VO) Approximation \\
	Calculates the Omega with helper_sapt and returns Omega exchange(all blocks) \\
	
	returns:
	  - Omega exchange(OV-VO block) formed by the symmetric blocks
	  - Omega exchange(OO-VV block) in S2 approximation
	NOTE
	- OO-VV blocks with general definition(1/2(OO/VV + OO/VV.T)) in S2 \\
	- OV-VO blocks formed by the symmetric blocks, taking transpose in Non-S2
	"""

	if ca is not None and cb is not None:
		sapt.set_orbitals(ca=ca, cb=cb)

	# OmegaB(exchange) in MO
	# ===========================    
	# OO block(S2)
	omegaB_exch_aa = 0.5*(
			-4 * oe.contract("sA,pb,abas->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
			-4 * oe.contract("bA,rb,pcrc->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
			-4 * oe.contract("sa,ab,pbAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
			-2 * oe.contract("sA,pb,bs->pA", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs', 'A'))
			-2 * oe.contract("sA,rb,pbrs->pA", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs'))
			-2 * oe.contract("bA,rb,pr->pA", sapt.s('ba'), sapt.s('rb'), sapt.potential('ar', 'B'))
			+2 * oe.contract("sa,pb,abAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
			+2 * oe.contract("ba,pb,aA->pA", sapt.s('ba'), sapt.s('ab'), sapt.potential('aa', 'B'))
			+4 * oe.contract("ba,pb,acAc->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('abab'))
	)
	# VV Block(S2)
	omegaB_exch_rr = 0.5*(
			-4 * oe.contract("sR,qb,abas->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('abas'))
			-4 * oe.contract("bR,rb,qcrc->qR", sapt.s('br'), sapt.s('rb'), sapt.v('rbrb'))
			-4 * oe.contract("sa,ab,qbRs->qR", sapt.s('sa'), sapt.s('ab'), sapt.v('rbrs'))
			-2 * oe.contract("sR,qb,bs->qR", sapt.s('sr'), sapt.s('rb'), sapt.potential('bs', 'A'))
			-2 * oe.contract("sR,rb,qbrs->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('rbrs'))
			-2 * oe.contract("bR,rb,qr->qR", sapt.s('br'), sapt.s('rb'), sapt.potential('rr', 'B'))
			+2 * oe.contract("sa,qb,abRs->qR", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs'))
			+2 * oe.contract("ba,qb,aR->qR", sapt.s('ba'), sapt.s('rb'), sapt.potential('ar', 'B'))
			+4 * oe.contract("ba,qb,acRc->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
	)
	# OmegaA(Exchange) in MO
	# =============================
	# OO Block(S2)
	omegaA_exch_bb = 0.5*(
		  -4 * oe.contract("rB,pa,abrb->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('abrb'))
		-4 * oe.contract("aB,sa,cpcs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('abas'))
		-4 * oe.contract("ba,rb,aprB->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
		-2 * oe.contract("rB,pa,ar->pB", sapt.s('rb'), sapt.s('ba'), sapt.potential('ar','B'))
		-2 * oe.contract("rB,sa,aprs->pB", sapt.s('rb'), sapt.s('sa'), sapt.v('abrs'))
		-2 * oe.contract("aB,sa,ps->pB", sapt.s('ab'), sapt.s('sa'), sapt.potential('bs','A'))
		+2 * oe.contract("pa,rb,abrB->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')) 
		+2 * oe.contract("pa,ab,bB->pB", sapt.s('ba'), sapt.s('ab'), sapt.potential('bb','A'))
		+4 * oe.contract("pa,ab,cbcB->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('abab'))
	)   
	# VV Block(S2)
	omegaA_exch_ss = 0.5*(
		-4 * oe.contract("rS,qa,abrb->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('abrb'))
		-4 * oe.contract("aS,sa,cqcs->qS", sapt.s('as'), sapt.s('sa'), sapt.v('asas'))
		-4 * oe.contract("ba,rb,aqrS->qS", sapt.s('ba'), sapt.s('rb'), sapt.v('asrs'))
		-2 * oe.contract("rS,qa,ar->qS", sapt.s('rs'), sapt.s('sa'), sapt.potential('ar','B'))
		-2 * oe.contract("rS,sa,aqrs->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('asrs'))
		-2 * oe.contract("aS,sa,qs->qS", sapt.s('as'), sapt.s('sa'), sapt.potential('ss','A'))
		+2 * oe.contract("qa,rb,abrS->qS", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs'))
		+2 * oe.contract("qa,ab,bS->qS", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs','A'))
		+4 * oe.contract("qa,ab,cbcS->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
	) 

	sinf = sinfinity(sapt)
	# Non-S2 block
	omegaA_exch_bs = sinf.omega_exchA_bs
	omegaB_exch_ar = sinf.omega_exchB_ar

	# S2 block
	omegaA_exch_bb_sym = 0.5*(omegaA_exch_bb + omegaA_exch_bb.T)
	omegaA_exch_ss_sym = 0.5*(omegaA_exch_ss + omegaA_exch_ss.T)

	# S2 block
	omegaB_exch_aa_sym = 0.5*(omegaB_exch_aa + omegaB_exch_aa.T)
	omegaB_exch_rr_sym = 0.5*(omegaB_exch_rr + omegaB_exch_rr.T)

	### OmegaB_exch in MO
	# OO[S2], 	OV[Non-S2]
	# VO[Non-S2], 	VV[S2]
	omegaB_exch_MO = np.block([[omegaB_exch_aa_sym, omegaB_exch_ar],
				[omegaB_exch_ar.T, omegaB_exch_rr_sym]])

	### OmegaA_exch in MO
	# OO[S2], 	OV[Non-S2]
	# VO[Non-S2], 	VV[S2]
	omegaA_exch_MO = np.block([[omegaA_exch_bb_sym, omegaA_exch_bs],
				[omegaA_exch_bs.T, omegaA_exch_ss_sym]])
	return omegaA_exch_MO, omegaB_exch_MO



def get_Exch_s2(sapt:helper_SAPT, ca=None, cb=None):
	"""
	Returns Exchnage energy(S2) for arbitary modified orbitals using helper_SAPT methods
	"""
	if ca is not None and cb is not None:
			sapt.set_orbitals(ca=ca, cb=cb)

	### Start E100 Exchange
	# exch_timer = sapt_timer('exchange')
	vt_abba = sapt.vt('abba')
	vt_abaa = sapt.vt('abaa')
	vt_abbb = sapt.vt('abbb')
	vt_abab = sapt.vt('abab')
	s_ab = sapt.s('ab')

	Exch100 = oe.contract('abba', vt_abba, optimize=True)

	tmp = 2 * vt_abaa - vt_abaa.swapaxes(2, 3)
	Exch100 += oe.contract('Ab,abaA', s_ab, tmp, optimize=True)

	tmp = 2 * vt_abbb - vt_abbb.swapaxes(2, 3)
	Exch100 += oe.contract('Ba,abBb', s_ab.T, tmp, optimize=True)

	Exch100 -= 2 * oe.contract('Ab,BA,abaB', s_ab, s_ab.T, vt_abab, optimize=True)
	Exch100 -= 2 * oe.contract('AB,Ba,abAb', s_ab, s_ab.T, vt_abab, optimize=True)
	Exch100 += oe.contract('Ab,Ba,abAB', s_ab, s_ab.T, vt_abab, optimize=True)

	Exch100 *= -2
	# exch_timer.stop()

	return Exch100.item()

def get_Exch_sinf(sapt:helper_SAPT, ca=None, cb=None):
	"""
	Returns Exchange energy(Non-S2) using helper_SAPT methods
	"""

	if ca is not None and cb is not None:
			sapt.set_orbitals(ca=ca, cb=cb)

	sinf = sinfinity(sapt)

	###  Matrices and tensors from helper_sapt
	s_ab = sapt.s("ab")
	s_ba = sapt.s("ba")
	s_sa = sapt.s("sa")
	s_rb = sapt.s("rb")

	### Exch100_Sinfinity
	Exch100_Sinf = -2 * oe.contract(
			"Bb,aB,sa,bs", sinf.A_bb, s_ab, s_sa, sinf.omegaA_bs
	)
	# print(Exch100_Sinf)
	Exch100_Sinf -= 2 * oe.contract(
			"Aa,bA,rb,ar", sinf.B_aa, s_ba, s_rb, sinf.omegaB_ar
	)
	# print(Exch100_Sinf)
	Exch100_Sinf -= 2 * oe.contract(
			"Bb,Aa,sA,rB,abrs", sinf.A_bb, sinf.B_aa, s_sa, s_rb, sinf.v_abrs
	)
	# print(Exch100_Sinf)
	Exch100_Sinf += 4 * oe.contract(
			"BD,AC,Da,Cb,sA,rB,abrs",
			sinf.A_bb,
			sinf.B_aa,
			s_ba,
			s_ab,
			s_sa,
			s_rb,
			sinf.v_abrs,
	)
	# print(Exch100_Sinf)

	return Exch100_Sinf

def get_Elst(sapt:helper_SAPT, ca, cb):
	"""
	Returns electrostatic energy using helper_SAPT methods
	"""
	sapt.set_orbitals(ca=ca, cb=cb)

	### Start E100 Electrostatics
	Elst10 = 4 * oe.contract('abab', sapt.vt('abab'), optimize=True)
	### End E100 Electrostatics
	return Elst10

def form_omega_exchange_off_diag_s2(sapt:helper_SAPT, 
						   ca=None, cb=None,
						   sym_all=True,
						   ):
	"""
	Form omega exchange, off-diagonal(OO-VV) in S2
	"""
	if ca is not None and cb is not None:
		sapt.set_orbitals(ca=ca, cb=cb)

	# OO Block
	omegaA_exch_OO_VP_s2 =0.5*(
		-4 * oe.contract("rp,Ba,abrb->pB", sapt.s('rb'), sapt.s('ba'), sapt.v('abrb')) 
		-4 * oe.contract("ap,sa,cBcs->pB", sapt.s('ab'), sapt.s('sa'), sapt.v('abas'))
		-4 * oe.contract("ba,rb,aBrp->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
		-2 * oe.contract("rp,Ba,ar->pB", sapt.s('rb'), sapt.s('ba'), sapt.potential('ar','B')) 
		-2 * oe.contract("rp,sa,aBrs->pB",sapt.s('rb'), sapt.s('sa'), sapt.v('abrs')) 
		-2 * oe.contract("ap,sa,Bs->pB", sapt.s('ab'), sapt.s('sa'), sapt.potential('bs','A'))
		+2 * oe.contract("Ba,rb,abrp->pB", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')) 
		+2 * oe.contract("Ba,ab,bp->pB", sapt.s('ba'), sapt.s('ab'), sapt.potential('bb','A')) 
		+4 * oe.contract("Ba,ab,cbcp->pB", sapt.s('ba'), sapt.s('ab'), sapt.v('abab')))
	
	# VV Block 
	omegaA_exch_VV_VP_s2 =0.5*(
		-4 * oe.contract("rq,Sa,abrb->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('abrb'))
		-4 * oe.contract("aq,sa,cScs->qS", sapt.s('as'), sapt.s('sa'), sapt.v('asas')) 
		-4 * oe.contract("ba,rb,aSrq->qS", sapt.s('ba'), sapt.s('rb'), sapt.v('asrs'))
		-2 * oe.contract("rq,Sa,ar->qS", sapt.s('rs'), sapt.s('sa'), sapt.potential('ar','B')) 
		-2 * oe.contract("rq,sa,aSrs->qS", sapt.s('rs'), sapt.s('sa'), sapt.v('asrs'))
		-2 * oe.contract("aq,sa,Ss->qS", sapt.s('as'), sapt.s('sa'), sapt.potential('ss','A')) 
		+2 * oe.contract("Sa,rb,abrq->qS", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
		+2 * oe.contract("Sa,ab,bq->qS", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs','A')) 
		+4 * oe.contract("Sa,ab,cbcq->qS", sapt.s('sa'), sapt.s('ab'), sapt.v('abas')))
	
	# OO block
	omegaB_exch_OO_VP_s2 = 0.5*(
		-4 * oe.contract("sA,pb,abas->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))
		-4 * oe.contract("bA,rb,pcrc->pA", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb')) 
		-4 * oe.contract("sa,ab,pbAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas'))  
		-2 * oe.contract("sA,pb,bs->pA", sapt.s('sa'), sapt.s('ab'), sapt.potential('bs','A')) 
		-2 * oe.contract("sA,rb,pbrs->pA", sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
		-2 * oe.contract("bA,rb,pr->pA", sapt.s('ba'), sapt.s('rb'), sapt.potential('ar','B')) 
		+2 * oe.contract("sa,pb,abAs->pA", sapt.s('sa'), sapt.s('ab'), sapt.v('abas')) 
		+2 * oe.contract("ba,pb,aA->pA", sapt.s('ba'), sapt.s('ab'), sapt.potential('aa','B'))  
		+4 * oe.contract("ba,pb,acAc->pA", sapt.s('ba'), sapt.s('ab'), sapt.v('abab')))
	
	# VV Block
	omegaB_exch_VV_VP_s2 =0.5*(
		-4 * oe.contract("sR,qb,abas->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('abas'))
		-4 * oe.contract("bR,rb,qcrc->qR", sapt.s('br'), sapt.s('rb'), sapt.v('rbrb')) 
		-4 * oe.contract("sa,ab,qbRs->qR", sapt.s('sa'), sapt.s('ab'), sapt.v('rbrs')) 
		-2 * oe.contract("sR,qb,bs->qR",  sapt.s('sr'), sapt.s('rb'), sapt.potential('bs','A')) 
		-2 * oe.contract("sR,rb,qbrs->qR", sapt.s('sr'), sapt.s('rb'), sapt.v('rbrs')) 
		-2 * oe.contract("bR,rb,qr->qR",  sapt.s('br'), sapt.s('rb'), sapt.potential('rr','B'))
		+2 * oe.contract("sa,qb,abRs->qR",sapt.s('sa'), sapt.s('rb'), sapt.v('abrs')) 
		+2 * oe.contract("ba,qb,aR->qR",  sapt.s('ba'), sapt.s('rb'), sapt.potential('ar','B'))
		+4 * oe.contract("ba,qb,acRc->qR", sapt.s('ba'), sapt.s('rb'), sapt.v('abrb'))
		)
	
	return omegaA_exch_OO_VP_s2, omegaA_exch_VV_VP_s2, omegaB_exch_OO_VP_s2, omegaB_exch_VV_VP_s2

def form_omega_exchange_off_diag_s4(sapt:helper_SAPT, 
						   ca=None, cb=None,
						   sym_all=True,
						   ):
	"""
	Form omega exchange, off-diagonal(OO-VV) in S4
	"""
	if ca is not None and cb is not None:
			sapt.set_orbitals(ca=ca, cb=cb)   
	
	s_ab = sapt.s('ab')
	s_ba = s_ab.T
	s_as = sapt.s('as')
	s_sa = s_as.T
	s_rb = sapt.s('rb')
	s_br = s_rb.T
	s_rs = sapt.s('rs')
	s_sr = s_rs.T
	
	v_abrb = sapt.v('abrb')
	v_abrs = sapt.v('abrs')
	v_abab = sapt.v('abab') 
	v_abas = sapt.v('abas')
	v_rbrs = sapt.v('rbrs')
	v_asrs = sapt.v('asrs')
	v_asas = sapt.v('asas')
	v_abab = sapt.v('abab')
	v_rbrb = sapt.v('rbrb')
	
	VA_bb = sapt.potential('bb','A')
	VA_bs = sapt.potential('bs','A')
	VA_ss = sapt.potential('ss','A')
 
	VB_aa = sapt.potential('aa','B')
	VB_ar = sapt.potential('ar','B')
	VB_rr = sapt.potential('rr','B')

	omegaA_exch_OO_VP_s4 = 0.5*(
		2 * oe.contract("Ba,bc,ab,rd,cdrp->pB", s_ba, s_ba, s_ab, s_rb, v_abrb) 
		+2 * oe.contract("ba,Bc,rb,ad,cdrp->pB", s_ba, s_ba, s_rb, s_ab, v_abrb)
		+2 * oe.contract("ba,Bc,cb,ae,ep->pB", s_ba, s_ba, s_ab, s_ab, VA_bb)
		+4 * oe.contract("rp,sa,Bc,ab,cbrs->pB", s_rb, s_sa, s_ba, s_ab, v_abrs)
		+4 * oe.contract("cp,ba,sc,rb,aBrs->pB", s_ab, s_ba, s_sa, s_rb, v_abrs)
		+4 * oe.contract("ba,Bc,cb,ae,fefp->pB", s_ba, s_ba, s_ab, s_ab, v_abab)
		-2 * oe.contract("rp,Ba,sc,ab,cbrs->pB", s_rb, s_ba, s_sa, s_ab, v_abrs)
		-2 * oe.contract("rp,Ba,bc,ab,cr->pB", s_rb, s_ba, s_ba, s_ab, VB_ar)
		-2 * oe.contract("rp,sa,bc,ab,cBrs->pB", s_rb, s_sa, s_ba, s_ab, v_abrs)
		-2 * oe.contract("ap,sa,Bc,rb,cbrs->pB", s_ab, s_sa, s_ba, s_rb, v_abrs)
		-2 * oe.contract("ap,ba,Bc,rb,cr->pB", s_ab, s_ba, s_ba, s_rb, VB_ar)
		-2 * oe.contract("ap,ba,sc,rb,cBrs->pB", s_ab, s_ba, s_sa, s_rb, v_abrs)
		-2 * oe.contract("cp,Ba,sc,ab,bs->pB", s_ab, s_ba, s_sa, s_ab, VA_bs)
		-2 * oe.contract("cp,sa,bc,ab,Bs->pB", s_ab, s_sa, s_ba, s_ab, VA_bs)
		-4 * oe.contract("rp,Ba,bc,ab,cere->pB", s_rb, s_ba, s_ba, s_ab, v_abrb)
		-4 * oe.contract("ap,ba,Bc,rb,cere->pB", s_ab, s_ba, s_ba, s_rb, v_abrb)
		-4 * oe.contract("cp,Ba,sc,ab,ebes->pB", s_ab, s_ba, s_sa, s_ab, v_abas)
		-4 * oe.contract("cp,sa,bc,ab,eBes->pB", s_ab, s_sa, s_ba, s_ab, v_abas)
		-4 * oe.contract("Ba,bc,rb,ad,cdrp->pB", s_ba, s_ba, s_rb, s_ab, v_abrb)
		-4 * oe.contract("ca,bd,ab,rc,dBrp->pB", s_ba, s_ba, s_ab, s_rb, v_abrb)
	)
 
	omegaA_exch_VV_VP_s4 = 0.5*(
		+2 * oe.contract("Sa,bc,ab,rd,cdrq->qS", s_sa, s_ba, s_ab,s_rb, v_abrs)
		+2 * oe.contract("ba,Sc,rb,ad,cdrq->qS", s_ba, s_sa, s_rb,s_ab, v_abrs)
		+2 * oe.contract("ba,Sc,cb,ae,eq->qS", s_ba, s_sa, s_ab,s_ab, VA_bs)
		+4 * oe.contract("rq,sa,Sc,ab,cbrs->qS", s_rs, s_sa, s_sa,s_ab, v_abrs)
		+4 * oe.contract("cq,ba,sc,rb,aSrs->qS", s_as, s_ba, s_sa,s_rb, v_asrs)
		+4 * oe.contract("ba,Sc,cb,ae,fefq->qS", s_ba, s_sa, s_ab,s_ab, v_abas)
		-2 * oe.contract("rq,Sa,sc,ab,cbrs->qS", s_rs, s_sa, s_sa,s_ab, v_abrs)
		-2 * oe.contract("rq,Sa,bc,ab,cr->qS", s_rs, s_sa, s_ba,s_ab, VB_ar) 
		-2 * oe.contract("rq,sa,bc,ab,cSrs->qS", s_rs, s_sa, s_ba,s_ab, v_asrs)
		-2 * oe.contract("aq,sa,Sc,rb,cbrs->qS", s_as, s_sa, s_sa,s_rb, v_abrs)
		-2 * oe.contract("aq,ba,Sc,rb,cr->qS", s_as, s_ba, s_sa,s_rb, VB_ar)
		-2 * oe.contract("aq,ba,sc,rb,cSrs->qS", s_as, s_ba, s_sa,s_rb, v_asrs)
		-2 * oe.contract("cq,Sa,sc,ab,bs->qS", s_as, s_sa, s_sa,s_ab, VA_bs)
		-2 * oe.contract("cq,sa,bc,ab,Ss->qS", s_as, s_sa, s_ba,s_ab, VA_ss)
		-4 * oe.contract("rq,Sa,bc,ab,cere->qS", s_rs, s_sa, s_ba,s_ab, v_abrb)
		-4 * oe.contract("aq,ba,Sc,rb,cere->qS", s_as, s_ba, s_sa,s_rb, v_abrb)
		-4 * oe.contract("cq,Sa,sc,ab,ebes->qS", s_as, s_sa, s_sa,s_ab, v_abas)
		-4 * oe.contract("cq,sa,bc,ab,eSes->qS", s_as, s_sa, s_ba,s_ab, v_asas)
		-4 * oe.contract("Sa,bc,rb,ad,cdrq->qS", s_sa, s_ba, s_rb,s_ab, v_abrs)
		-4 * oe.contract("ca,bd,ab,rc,dSrq->qS", s_ba, s_ba, s_ab,s_rb, v_asrs)
	)

	omegaB_exch_OO_VP_s4 = 0.5*(
		+2 * oe.contract("sa,bc,ab,Ad,cdps->pA", s_sa, s_ba, s_ab,s_ab, v_abas)
		+2 * oe.contract("ba,sc,Ab,ad,cdps->pA", s_ba, s_sa, s_ab,s_ab, v_abas)
		+2 * oe.contract("ca,bd,ab,Ac,dp->pA", s_ba, s_ba, s_ab,s_ab, VB_aa)
		+4 * oe.contract("sp,ba,rb,Ac,acrs->pA", s_sa, s_ba, s_rb,s_ab, v_abrs)
		+4 * oe.contract("cp,sa,ab,rc,Abrs->pA", s_ba, s_sa, s_ab,s_rb, v_abrs)
		+4 * oe.contract("ca,bd,ab,Ac,dgpg->pA", s_ba, s_ba, s_ab,s_ab, v_abab)
		-2 * oe.contract("sp,ba,Ab,rc,acrs->pA", s_sa, s_ba, s_ab,s_rb, v_abrs)
		-2 * oe.contract("sp,ba,Ab,ac,cs->pA", s_sa, s_ba, s_ab,s_ab, VA_bs)
		-2 * oe.contract("sp,ba,rb,ac,Acrs->pA", s_sa, s_ba, s_rb,s_ab, v_abrs) 
		-2 * oe.contract("bp,sa,rb,Ac,acrs->pA", s_ba, s_sa, s_rb,s_ab, v_abrs)
		-2 * oe.contract("bp,sa,ab,Ac,cs->pA", s_ba, s_sa, s_ab,s_ab, VA_bs) 
		-2 * oe.contract("bp,sa,ab,rc,Acrs->pA", s_ba, s_sa, s_ab,s_rb, v_abrs)
		-2 * oe.contract("cp,ba,Ab,rc,ar->pA", s_ba, s_ba, s_ab,s_rb, VB_ar)
		-2 * oe.contract("cp,ba,rb,ac,Ar->pA", s_ba, s_ba, s_rb,s_ab, VB_ar) 
		-4 * oe.contract("sp,ba,Ab,ac,dcds->pA", s_sa, s_ba, s_ab,s_ab, v_abas)
		-4 * oe.contract("bp,sa,ab,Ac,dcds->pA", s_ba, s_sa, s_ab,s_ab, v_abas)
		-4 * oe.contract("cp,ba,Ab,rc,aere->pA", s_ba, s_ba, s_ab,s_rb, v_abrb)
		-4 * oe.contract("cp,ba,rb,ac,Aere->pA", s_ba, s_ba, s_rb,s_ab, v_abrb)
		-4 * oe.contract("sa,bc,Ab,ad,cdps->pA", s_sa, s_ba, s_ab,s_ab, v_abas)
		-4 * oe.contract("ba,sc,cb,ae,Aeps->pA", s_ba, s_sa, s_ab,s_ab, v_abas)
	)

	omegaB_exch_VV_VP_s4 = 0.5*(
		+2 * oe.contract("sa,bc,ab,Rd,cdqs->qR", s_sa, s_ba, s_ab,s_rb, v_abrs)
		+2 * oe.contract("ba,sc,Rb,ad,cdqs->qR", s_ba, s_sa, s_rb,s_ab, v_abrs)
		+2 * oe.contract("ca,bd,ab,Rc,dq->qR", s_ba, s_ba, s_ab, s_rb, VB_ar)
		+4 * oe.contract("sq,ba,rb,Rc,acrs->qR", s_sr, s_ba, s_rb,s_rb, v_abrs)
		+4 * oe.contract("cq,sa,ab,rc,Rbrs->qR", s_br, s_sa, s_ab,s_rb, v_rbrs)
		+4 * oe.contract("ca,bd,ab,Rc,dgqg->qR", s_ba, s_ba, s_ab,s_rb, v_abrb)
		-2 * oe.contract("sq,ba,Rb,rc,acrs->qR", s_sr, s_ba, s_rb,s_rb, v_abrs)
		-2 * oe.contract("sq,ba,Rb,ac,cs->qR", s_sr, s_ba, s_rb, s_ab, VA_bs) 
		-2 * oe.contract("sq,ba,rb,ac,Rcrs->qR", s_sr, s_ba, s_rb,s_ab, v_rbrs)
		-2 * oe.contract("bq,sa,rb,Rc,acrs->qR", s_br, s_sa, s_rb,s_rb, v_abrs)
		-2 * oe.contract("bq,sa,ab,Rc,cs->qR", s_br, s_sa, s_ab, s_rb, VA_bs) 
		-2 * oe.contract("bq,sa,ab,rc,Rcrs->qR", s_br, s_sa, s_ab,s_rb, v_rbrs)
		-2 * oe.contract("cq,ba,Rb,rc,ar->qR", s_br, s_ba, s_rb, s_rb, VB_ar) 
		-2 * oe.contract("cq,ba,rb,ac,Rr->qR", s_br, s_ba, s_rb, s_ab, VB_rr) 
		-4 * oe.contract("sq,ba,Rb,ac,dcds->qR", s_sr, s_ba, s_rb,s_ab, v_abas)
		-4 * oe.contract("bq,sa,ab,Rc,dcds->qR", s_br, s_sa, s_ab,s_rb, v_abas)
		-4 * oe.contract("cq,ba,Rb,rc,aere->qR", s_br, s_ba, s_rb,s_rb, v_abrb)
		-4 * oe.contract("cq,ba,rb,ac,Rere->qR", s_br, s_ba, s_rb,s_ab, v_rbrb)
		-4 * oe.contract("sa,bc,Rb,ad,cdqs->qR", s_sa, s_ba, s_rb,s_ab, v_abrs)
		-4 * oe.contract("ba,sc,cb,ae,Reqs->qR", s_ba, s_sa, s_ab,s_ab, v_rbrs)
	)

# 	if sym_all:
# 		#  Omegas symmetrized
# 		# 1/2(VP +PV)
# 		omegaB_exch_OO_sym = 0.5*(omegaB_exch_OO_VP + omegaB_exch_OO_VP.T)
# 		omegaB_exch_OV_sym = 0.5*(omegaB_exch_OV_VP + omegaB_exch_OV_PV)    
# 		omegaB_exch_VO_sym = 0.5*(omegaB_exch_VO_VP + omegaB_exch_VO_PV)
# 		omegaB_exch_VV_sym = 0.5*(omegaB_exch_VV_VP + omegaB_exch_VV_VP.T)

# 		omegaA_exch_OO_sym = 0.5*(omegaA_exch_OO_VP + omegaA_exch_OO_VP.T)
# 		omegaA_exch_OV_sym = 0.5*(omegaA_exch_OV_VP + omegaA_exch_OV_PV)
# 		omegaA_exch_VO_sym = 0.5*(omegaA_exch_VO_VP + omegaA_exch_VO_PV)
# 		omegaA_exch_VV_sym = 0.5*(omegaA_exch_VV_VP + omegaA_exch_VV_VP.T)

# 		# Forming omega exchange matrix
# 		omegaA_exchange_s4 = np.block([[omegaA_exch_OO_sym, omegaA_exch_OV_sym],
# 								   [omegaA_exch_VO_sym, omegaA_exch_VV_sym]])
# 		omegaB_exchange_s4 = np.block([[omegaB_exch_OO_sym, omegaB_exch_OV_sym],
# 								   [omegaB_exch_VO_sym, omegaB_exch_VV_sym]])

	return omegaA_exch_OO_VP_s4, omegaA_exch_VV_VP_s4, omegaB_exch_OO_VP_s4, omegaB_exch_VV_VP_s4

#================================================================================
#					 Main Methods for call in SART utils 						#
#================================================================================

def form_omega_exchange_w_sym(sapt:helper_SAPT, ca=None, cb=None, oo_vv=None, ov_vo='S2'):
    """
    Main method to form Omega-exchange for different combinations, 
    with symmetrized construction scheme
    """        
    if oo_vv not in [None, 'S2', 'S4']:
        raise KeyError(f'{oo_vv} approximation is not available for OO/VV blocks of Omega(Exchange)')
    if ov_vo not in ['S2', 'Sinf']:
        raise KeyError(f'{ov_vo} approximation is not available for OO/VV blocks of Omega(Exchange)')
    
    if oo_vv is None:
        # Make OO-VV ZEROES!!!!!
        raise RuntimeError(f'OV-VO case is not available for symmetrized omega scheme')
    elif oo_vv is not None and ov_vo is not None:
        if oo_vv == 'S2' and ov_vo == 'S2':
            omega_exchA, omega_exchB = form_omega_exchange_s2_sym(sapt = sapt,
                                                                  ca= ca,
                                                                  cb =cb)
        elif oo_vv == 'S2' and ov_vo == 'Sinf':
            omega_exchA, omega_exchB = form_omega_exchange_sinf_sym(sapt = sapt,
                                                                  ca= ca,
                                                                  cb =cb)
        elif oo_vv == 'S4' and ov_vo == 'Sinf':
            omega_exchA, omega_exchB = form_omega_exchange_sinf_sym(sapt = sapt,
                                                                  ca= ca,
                                                                  cb =cb,
                                                                  s4_bool= True)
        # elif oo_vv == 'Sinf' and ov_vo == 'Sinf':
        #     pass
        else:
            raise NotImplementedError(f'OO/VV,{oo_vv} and OV/VO,{ov_vo} approximation is not available for Omega(Exchange)')
        return omega_exchA, omega_exchB

def form_omega_exchange_no_sym(sapt:helper_SAPT, ca=None, cb=None, oo_vv=None, ov_vo='S2'):
    """
    Main method to form Omega-exchange for different combinations, 
    without symmetrized construction scheme
    ***NOTE: Only implemented for OV-VO only case
    """   
    
    if oo_vv not in [None, 'S2', 'S4']:
        raise KeyError(f'{oo_vv} approximation is not available for OO/VV blocks of Omega(Exchange)')
    if ov_vo not in ['S2', 'Sinf']:
        raise KeyError(f'{ov_vo} approximation is not available for OO/VV blocks of Omega(Exchange)')
    
    if oo_vv is None:
        # Make OO-VV ZEROES!!!!!
        if ov_vo == 'S2':
            omega_exchA, omega_exchB = form_omega_exchange_s2(sapt = sapt,
                                                            ca= ca,
                                                            cb =cb)
            return omega_exchA, omega_exchB
        elif ov_vo == 'Sinf':
            omega_exchA, omega_exchB = form_omega_exchange_sinf(sapt = sapt,
                                                            ca= ca,
                                                            cb =cb)
            return omega_exchA, omega_exchB 
    elif oo_vv is not None and ov_vo is not None:
            # More precisely, NotImplementedError
            raise RuntimeError(f'OO/VV({oo_vv}) and OV/VO({ov_vo}) options for not available for Omega(Exchange), no sym scheme')      
	

# ================================== For method TEST ===============================================
if __name__ == '__main__':
	import psi4

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
	# dimer = psi4.geometry("""
	# 	O   -0.066999140   0.000000000   1.494354740
	# 	H    0.815734270   0.000000000   1.865866390
	# 	H    0.068855100   0.000000000   0.539142770
	# 	--
	# 	O    0.062547750   0.000000000  -1.422632080
	# 	H   -0.406965400  -0.760178410  -1.771744500
	# 	H   -0.406965400   0.760178410  -1.771744500
	# 	symmetry c1
	# 	""")

	dimer = psi4.geometry(he_li_str_10)

	psi4.set_options({
				# 'basis': 'jun-cc-pVDZ',
				'basis': 'sto-3g',
				'scf_type': 'direct',
				'e_convergence': 1e-8,
				'd_convergence': 1e-8})

	sapt = helper_SAPT(dimer, memory=8)
	sinf = sinfinity(sapt=sapt)

	sapt_CA = sapt.wfnA.Ca()
	sapt_CB = sapt.wfnB.Ca()

	# S2 Exchange
	exch = get_Exch_s2(sapt, ca= sapt_CA, cb= sapt_CB)
	print('S2 Exchange:', exch)

	# Non-S2 Exchange
	Exch_sinf = get_Exch_sinf(sapt, ca= sapt_CA, cb= sapt_CB)
	print('Non-S2 Exchange:', Exch_sinf)

