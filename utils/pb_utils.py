"""
Utilities for HHF and PB methods
"""

import numpy as np
import opt_einsum as oe

def get_penalty(c, s, nocc=None):

    """ Generates the penalty operator
        c: MO coefficient matrix
        s: Overlap matrix in AO basis
        nocc: Number of occupied orbitals
    """
    n = s.shape[0]
    r_0 = np.zeros((n,n))
    if nocc == 0:
        return r_0
    else:
        if nocc is not None:
            Cocc = c[:, :nocc]
            r = np.dot(s, Cocc).dot(Cocc.T).dot(s)
            return r
        else:
            r = np.dot(s, c).dot(c.T).dot(s)
            return r
  
def get_dimer_orthogalised(ca, cb, S):
    """ Orthogonalizes the monomer MO coefficient matrces
        ca : MO coefficients of monomer A
        cb : MO coefficients of monomer B
        S : Overlap matrix in AO   
    """
    from scipy.linalg import fractional_matrix_power
    # Orthogalise the DCBS
    # Form the set of Occupied: CA and CB

    nA = ca.shape[1]
    nB = cb.shape[1]
    d_ab = np.column_stack((ca, cb))

    # To get T matrix
    taa = np.eye(nA)
    tbb = np.eye(nB)
    tab = np.transpose(ca).dot(S).dot(cb)
    tba = np.transpose(cb).dot(S).dot(ca)

    t = np.block([[taa, tab],[tba, tbb]])
    t_exp = fractional_matrix_power(t, -0.5)
    d_ab_til = d_ab.dot(t_exp)
    nMO = d_ab_til.shape[1]

    cA_til = d_ab_til[:, :nA]
    cB_til = d_ab_til[:, nA:]

    return cA_til, cB_til

def get_elst(Ca, Cb, I, VA, VB, Enucint):
    """
    Returns total interaction electrostatic energy
    Ca: Occupied C for A
    Cb: Occupied C for B
    I: ERI tensor
    VA: Nuclear Potential of A
    VB: Nuclear Potential of B
    Enucint: Interacting part of nuc-nuc energy
    """
    Da = oe.contract('pi,qi->pq', Ca, Ca)
    Db = oe.contract('pi,qi->pq', Cb, Cb)
    e_1 = oe.contract('pq,pqrs,rs', Da, I, Db)
    e_2 = np.trace(np.dot(VA, Db)) 
    e_3 = np.trace(np.dot(VB, Da))
    E_elst = e_1*4 + e_2*2 + e_3*2 + Enucint
    return E_elst

def get_elst_nuc_e(Ca, Cb, I, VA, VB, Enucint):
    """
    Returns total interaction electrostatic energy
    Ca: Occupied C for A
    Cb: Occupied C for B
    I: ERI tensor
    VA: Nuclear Potential of A
    VB: Nuclear Potential of B
    Enucint: Interacting part of nuc-nuc energy
    """
    Da = oe.contract('pi,qi->pq', Ca, Ca)
    Db = oe.contract('pi,qi->pq', Cb, Cb)
    # e_1 = oe.contract('pq,pqrs,rs', Da, I, Db)
    e_2 = np.trace(np.dot(VA, Db)) 
    e_3 = np.trace(np.dot(VB, Da))

    E_elst = e_2*2 + e_3*2 + Enucint
    print('Nuc(A)-e(B):', e_2*2)
    print('Nuc(B)-e(A):', e_3*2)
    print('E-nuc-int:', Enucint)
    print('Total-elst:', E_elst)

    # E_elst = e_1*4 + e_2*2 + e_3*2 + Enucint
    
    return E_elst

def get_exch(Ca, Cb, I):
    """
    Returns Exchange Energy
    Ca: Occupied C for A
    Cb: Occupied C for B
    I : ERI tensor
    """
    Da = oe.contract('pi,qi->pq', Ca, Ca)
    Db = oe.contract('pi,qi->pq', Cb, Cb)
    E_exch = -2*oe.contract('pq,prqs,rs', Da, I, Db)
    print('sq E_exch =', E_exch)
    return E_exch

def get_fock(h,i,c):
    """
    Returns fock matrix
    h: 1 e- hamiltonian
    i: Electron Repulsion Integrals
    c: MO coefficient matrix   
    """
    d = oe.contract('pi,qi->pq', c, c)  
    j = oe.contract('pqrs,rs->pq', i, d)
    k = oe.contract('prqs,rs->pq', i, d)

    f = h + 2*j - k
    return f

def diagonalise_fock(f, orthogonalizer, nocc= None):
    """
    Assumes Orthogonalizer A = S^(-1/2)

    Diagonalizes Fock matrix 
    and returns C coeffients for occupied orbitals, 
    if nocc is specified, else returns complete C coefficients
    """
    a = orthogonalizer
    f_i = a.dot(f).dot(a)                   
    e, ca_temp = np.linalg.eigh(f_i)             
    ca_i = a.dot(ca_temp)
    
    if nocc is not None :                         
        ca_occ = ca_i[:, :nocc] 
        return ca_occ
    else:
        return ca_i
