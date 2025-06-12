"""
Second-quantized SAPT S^infinity
"""

from time import time
import psi4
import numpy as np
import opt_einsum as oe

from utils.helper_SAPT import helper_SAPT
# from helper_SAPT_DF import helper_SAPT

def get_A_bb(s_ba, s_ab):
    gamma_bb = oe.contract("Ba,ab->Bb", s_ba, s_ab)

    lam, U = np.linalg.eigh(gamma_bb)

    gamma_bb_prime = np.diag(lam)
    A_bb_prime = oe.contract("Bb,Bb->Bb", gamma_bb_prime, 1 / (1 - gamma_bb_prime))
    A_bb = U.dot(A_bb_prime.dot(U.T))
    A_bb = np.identity(A_bb.shape[0]) + A_bb

    return A_bb


def get_B_aa(s_ba, s_ab):
    delta_aa = oe.contract("ba,Ab->Aa", s_ba, s_ab)

    lam, U = np.linalg.eigh(delta_aa)

    delta_aa_prime = np.diag(lam)
    B_aa_prime = oe.contract("Aa,Aa->Aa", delta_aa_prime, 1 / (1 - delta_aa_prime))
    B_aa = U.dot(B_aa_prime.dot(U.T))
    B_aa = np.identity(B_aa.shape[0]) + B_aa

    return B_aa


class sinfinity:
    def __init__(self, sapt: helper_SAPT, df_basis=None) -> None:
        ### Starting initialisation
        psi4.core.print_out("\nInitializing Sinfinity object...\n")
        t_start = time()

        ###  Matrices and tensors from helper_sapt
        s_ab = sapt.s("ab")
        s_ba = sapt.s("ba")
        s_sa = sapt.s("sa")
        s_as = sapt.s("as")
        s_rb = sapt.s("rb")
        s_sr = sapt.s("sr")
        s_rs = sapt.s("rs")
        s_br = sapt.s("br")

        vA_bb = sapt.potential("bb", "A")
        vA_bs = sapt.potential("bs", "A")
        vA_ss = sapt.potential("ss", "A")

        vB_aa = sapt.potential("aa", "B")
        vB_ar = sapt.potential("ar", "B")
        vB_rr = sapt.potential("rr", "B")

        ### Calculating A and B matrices
        self.A_bb = get_A_bb(s_ba, s_ab)
        self.B_aa = get_B_aa(s_ba, s_ab)

        self.H_ab = oe.contract("Bb,aB->ab", self.A_bb, s_ab)
        self.H_ba = oe.contract("Ba,bB->ba", s_ba, self.A_bb)
        self.I_rb = oe.contract("Bb,rB->rb", self.A_bb, s_rb)
        self.I_br = oe.contract("Br,bB->br", s_br, self.A_bb)
        self.J_sa = oe.contract("Aa,sA->sa", self.B_aa, s_sa)
        self.J_as = oe.contract("As,aA->as", s_as, self.B_aa)

        self.G_sr = s_sr + oe.contract("br,Bb,aB,sa->sr", s_br, self.A_bb, s_ab, s_sa)
        self.G_rs = s_rs + oe.contract("as,Aa,bA,rb->rs", s_as, self.B_aa, s_ba, s_rb)

        self.C_rr = np.identity(sapt.sizes["r"]) - oe.contract(
            "bR,Bb,rB->rR", s_br, self.A_bb, s_rb
        )
        self.D_ss = np.identity(sapt.sizes["s"]) - oe.contract(
            "aS,Aa,sA->sS", s_as, self.B_aa, s_sa
        )

        self.E_ra = oe.contract("ba,Bb,rB->ra", s_ba, self.A_bb, s_rb)
        self.E_ar = oe.contract("aB,Bb,br->ar", s_ab, self.A_bb, s_br)
        self.F_sb = oe.contract("ab,Aa,sA->sb", s_ab, self.B_aa, s_sa)
        self.F_bs = oe.contract("bA,Aa,as->bs", s_ba, self.B_aa, s_as)

        # dispersion denominator
        self.e_rsab = 1 / (
            -sapt.eps("r", dim=4)
            - sapt.eps("s", dim=3)
            + sapt.eps("a", dim=2)
            + sapt.eps("b")
        )

        if df_basis:
            # MO transformed df ERIs
            self.Qaa = sapt.df_ints("aa")
            self.Qar = sapt.df_ints("ar")
            self.Qrr = sapt.df_ints("rr")
            self.Qbb = sapt.df_ints("bb")
            self.Qbs = sapt.df_ints("bs")
            self.Qss = sapt.df_ints("ss")

            ### omegaA and omegaB
            self.omegaA_bs = vA_bs + 2 * oe.contract("Qaa,Qbs->bs", self.Qaa, self.Qbs)
            self.omegaB_ar = vB_ar + 2 * oe.contract("Qar,Qbb->ar", self.Qar, self.Qbb)

            self.omegaB_rr = 2 * oe.contract("QrR,Qbb->rR", self.Qrr, self.Qbb) + vB_rr
            self.omegaB_aa = 2 * oe.contract("QaA,Qbb->aA", self.Qaa, self.Qbb) + vB_aa

            self.omegaA_ss = 2 * oe.contract("Qaa,QsS->sS", self.Qaa, self.Qss) + vA_ss
            self.omegaA_bb = 2 * oe.contract("Qaa,QbB->bB", self.Qaa, self.Qbb) + vA_bb

            ### omega_exchA and omega_exchB
            self.omega_exchA_bs = (
                oe.contract("Bb,sS,bs->BS", self.A_bb, self.D_ss, self.omegaA_bs)
                - oe.contract("rs,ba,ar->bs", self.G_rs, self.H_ba, self.omegaB_ar)
                - 2
                * oe.contract(
                    "ra,Bb,sS,Qar,Qbs->BS",
                    self.E_ra,
                    self.A_bb,
                    self.D_ss,
                    self.Qar,
                    self.Qbs,
                )
                - oe.contract(
                    "Bb,rS,sa,Qar,Qbs->BS",
                    self.A_bb,
                    self.G_rs,
                    self.J_sa,
                    self.Qar,
                    self.Qbs,
                )
                + oe.contract(
                    "rb,sS,Ba,Qar,Qbs->BS",
                    self.I_rb,
                    self.D_ss,
                    self.H_ba,
                    self.Qar,
                    self.Qbs,
                )
                + 2
                * oe.contract(
                    "sb,rS,Ba,Qar,Qbs->BS",
                    self.F_sb,
                    self.G_rs,
                    self.H_ba,
                    self.Qar,
                    self.Qbs,
                )
            ) - self.omegaA_bs

            self.omega_exchA_sb = (
                -2 * oe.contract("Qar,Qbs,ra->sb", self.Qar, self.Qbs, self.E_ra)
                + oe.contract("bB,sb->sB", self.omegaA_bb, self.F_sb)
                - oe.contract("Ss,sb->Sb", self.omegaA_ss, self.F_sb)
                + oe.contract(
                    "Qar,QbB,sa,rb->sB",
                    self.Qar,
                    self.Qbb,
                    self.J_sa,
                    self.I_rb,
                )
                - oe.contract(
                    "Qar,QSs,sa,rb->Sb",
                    self.Qar,
                    self.Qss,
                    self.J_sa,
                    self.I_rb,
                )
                - oe.contract("ar,sa,rb->sb", self.omegaB_ar, self.J_sa, self.I_rb)
                - oe.contract("BS,Sb,sB->sb", self.omegaA_bs, self.F_sb, self.F_sb)
                - oe.contract(
                    "Qar,QBS,Sb,sa,rB->sb",
                    self.Qar,
                    self.Qbs,
                    self.F_sb,
                    self.J_sa,
                    self.I_rb,
                )
                - oe.contract(
                    "Qar,QBS,sB,Sa,rb->sb",
                    self.Qar,
                    self.Qbs,
                    self.F_sb,
                    self.J_sa,
                    self.I_rb,
                )
                + 2
                * oe.contract(
                    "Qar,QBS,SB,sa,rb->sb",
                    self.Qar,
                    self.Qbs,
                    self.F_sb,
                    self.J_sa,
                    self.I_rb,
                )
                + 2
                * oe.contract(
                    "Qar,QBS,Sb,sB,ra->sb",
                    self.Qar,
                    self.Qbs,
                    self.F_sb,
                    self.F_sb,
                    self.E_ra,
                )
            )

            self.omega_exchB_ar = (
                oe.contract("Aa,rR,ar->AR", self.B_aa, self.C_rr, self.omegaB_ar)
                - oe.contract("sr,ab,bs->ar", self.G_sr, self.H_ab, self.omegaA_bs)
                - 2
                * oe.contract(
                    "Aa,rR,sb,Qar,Qbs->AR",
                    self.B_aa,
                    self.C_rr,
                    self.F_sb,
                    self.Qar,
                    self.Qbs,
                )
                - oe.contract(
                    "rb,Aa,sR,Qar,Qbs->AR",
                    self.I_rb,
                    self.B_aa,
                    self.G_sr,
                    self.Qar,
                    self.Qbs,
                )
                + oe.contract(
                    "rR,Ab,sa,Qar,Qbs->AR",
                    self.C_rr,
                    self.H_ab,
                    self.J_sa,
                    self.Qar,
                    self.Qbs,
                )
                + 2
                * oe.contract(
                    "ra,sR,Ab,Qar,Qbs->AR",
                    self.E_ra,
                    self.G_sr,
                    self.H_ab,
                    self.Qar,
                    self.Qbs,
                )
            ) - self.omegaB_ar

            self.omega_exchB_ra = (
                -2 * oe.contract("Qar,Qbs,sb->ra", self.Qar, self.Qbs, self.F_sb)
                + oe.contract("aA,ra->rA", self.omegaB_aa, self.E_ra)
                - oe.contract("Rr,ra->Ra", self.omegaB_rr, self.E_ra)
                + oe.contract(
                    "QaA,Qbs,rb,sa->rA",
                    self.Qaa,
                    self.Qbs,
                    self.I_rb,
                    self.J_sa,
                )
                - oe.contract(
                    "QRr,Qbs,rb,sa->Ra",
                    self.Qrr,
                    self.Qbs,
                    self.I_rb,
                    self.J_sa,
                )
                - oe.contract("bs,rb,sa->ra", self.omegaA_bs, self.I_rb, self.J_sa)
                - oe.contract("AR,Ra,rA->ra", self.omegaB_ar, self.E_ra, self.E_ra)
                - oe.contract(
                    "QAR,Qbs,Ra,rb,sA->ra",
                    self.Qar,
                    self.Qbs,
                    self.E_ra,
                    self.I_rb,
                    self.J_sa,
                )
                - oe.contract(
                    "QAR,Qbs,rA,Rb,sa->ra",
                    self.Qar,
                    self.Qbs,
                    self.E_ra,
                    self.I_rb,
                    self.J_sa,
                )
                + 2
                * oe.contract(
                    "QAR,Qbs,RA,rb,sa->ra",
                    self.Qar,
                    self.Qbs,
                    self.E_ra,
                    self.I_rb,
                    self.J_sa,
                )
                + 2
                * oe.contract(
                    "QAR,Qbs,Ra,rA,sb->ra",
                    self.Qar,
                    self.Qbs,
                    self.E_ra,
                    self.E_ra,
                    self.F_sb,
                )
            )

            ### dispersion amplitudes
            self.t_rsab = np.array(
                oe.contract("Qar,Qbs,rsab->rsab", self.Qar, self.Qbs, self.e_rsab)
            )

        else:
            ### omegaA and omegaB
            v_abas = sapt.v("abas")
            v_abrb = sapt.v("abrb")
            self.v_abrs = sapt.v("abrs")
            v_abab = sapt.v("abab")
            v_rbrs = sapt.v("rbrs")
            v_asrs = sapt.v("asrs")

            self.omegaA_bs = vA_bs + 2 * oe.contract("abas->bs", v_abas)
            self.omegaB_ar = vB_ar + 2 * oe.contract("abrb->ar", v_abrb)

            self.omegaB_rr = 2 * oe.contract("rbRb->rR", sapt.v("rbrb")) + vB_rr
            self.omegaB_aa = 2 * oe.contract("abAb->aA", v_abab) + vB_aa

            self.omegaA_ss = 2 * oe.contract("asaS->sS", sapt.v("asas")) + vA_ss
            self.omegaA_bb = 2 * oe.contract("abaB->bB", v_abab) + vA_bb

            ### omega_exchA and omega_exchB
            self.omega_exchA_bs = (
                oe.contract("Bb,sS,bs->BS", self.A_bb, self.D_ss, self.omegaA_bs)
                - oe.contract("rs,ba,ar->bs", self.G_rs, self.H_ba, self.omegaB_ar)
                - 2
                * oe.contract(
                    "ra,Bb,sS,abrs->BS", self.E_ra, self.A_bb, self.D_ss, self.v_abrs
                )
                - oe.contract(
                    "Bb,rS,sa,abrs->BS", self.A_bb, self.G_rs, self.J_sa, self.v_abrs
                )
                + oe.contract(
                    "rb,sS,Ba,abrs->BS", self.I_rb, self.D_ss, self.H_ba, self.v_abrs
                )
                + 2
                * oe.contract(
                    "sb,rS,Ba,abrs->BS", self.F_sb, self.G_rs, self.H_ba, self.v_abrs
                )
            ) - self.omegaA_bs

            self.omega_exchA_sb = (
                -2 * oe.contract("abrs,ra->sb", self.v_abrs, self.E_ra)
                + oe.contract("bB,sb->sB", self.omegaA_bb, self.F_sb)
                - oe.contract("Ss,sb->Sb", self.omegaA_ss, self.F_sb)
                + oe.contract("abrB,sa,rb->sB", v_abrb, self.J_sa, self.I_rb)
                - oe.contract("aSrs,sa,rb->Sb", v_asrs, self.J_sa, self.I_rb)
                - oe.contract("ar,sa,rb->sb", self.omegaB_ar, self.J_sa, self.I_rb)
                - oe.contract("BS,Sb,sB->sb", self.omegaA_bs, self.F_sb, self.F_sb)
                - oe.contract(
                    "aBrS,Sb,sa,rB->sb", self.v_abrs, self.F_sb, self.J_sa, self.I_rb
                )
                - oe.contract(
                    "aBrS,sB,Sa,rb->sb", self.v_abrs, self.F_sb, self.J_sa, self.I_rb
                )
                + 2
                * oe.contract(
                    "aBrS,SB,sa,rb->sb", self.v_abrs, self.F_sb, self.J_sa, self.I_rb
                )
                + 2
                * oe.contract(
                    "aBrS,Sb,sB,ra->sb", self.v_abrs, self.F_sb, self.F_sb, self.E_ra
                )
            )

            self.omega_exchB_ar = (
                oe.contract("Aa,rR,ar->AR", self.B_aa, self.C_rr, self.omegaB_ar)
                - oe.contract("sr,ab,bs->ar", self.G_sr, self.H_ab, self.omegaA_bs)
                - 2
                * oe.contract(
                    "Aa,rR,sb,abrs->AR", self.B_aa, self.C_rr, self.F_sb, self.v_abrs
                )
                - oe.contract(
                    "rb,Aa,sR,abrs->AR", self.I_rb, self.B_aa, self.G_sr, self.v_abrs
                )
                + oe.contract(
                    "rR,Ab,sa,abrs->AR", self.C_rr, self.H_ab, self.J_sa, self.v_abrs
                )
                + 2
                * oe.contract(
                    "ra,sR,Ab,abrs->AR", self.E_ra, self.G_sr, self.H_ab, self.v_abrs
                )
            ) - self.omegaB_ar

            self.omega_exchB_ra = (
                -2 * oe.contract("abrs,sb->ra", self.v_abrs, self.F_sb)
                + oe.contract("aA,ra->rA", self.omegaB_aa, self.E_ra)
                - oe.contract("Rr,ra->Ra", self.omegaB_rr, self.E_ra)
                + oe.contract("abAs,rb,sa->rA", v_abas, self.I_rb, self.J_sa)
                - oe.contract("Rbrs,rb,sa->Ra", v_rbrs, self.I_rb, self.J_sa)
                - oe.contract("bs,rb,sa->ra", self.omegaA_bs, self.I_rb, self.J_sa)
                - oe.contract("AR,Ra,rA->ra", self.omegaB_ar, self.E_ra, self.E_ra)
                - oe.contract(
                    "AbRs,Ra,rb,sA->ra", self.v_abrs, self.E_ra, self.I_rb, self.J_sa
                )
                - oe.contract(
                    "AbRs,rA,Rb,sa->ra", self.v_abrs, self.E_ra, self.I_rb, self.J_sa
                )
                + 2
                * oe.contract(
                    "AbRs,RA,rb,sa->ra", self.v_abrs, self.E_ra, self.I_rb, self.J_sa
                )
                + 2
                * oe.contract(
                    "AbRs,Ra,rA,sb->ra", self.v_abrs, self.E_ra, self.E_ra, self.F_sb
                )
            )

            ### dispersion amplitudes
            self.t_rsab = np.array(
                oe.contract("abrs,rsab->rsab", self.v_abrs, self.e_rsab)
            )

        ### Non-response amplitudes
        e_ra = 1 / (-sapt.eps("r", dim=2) + sapt.eps("a"))
        self.tB_ra = self.omegaB_ar.T * e_ra

        e_sb = 1 / (-sapt.eps("s", dim=2) + sapt.eps("b"))
        self.tA_sb = self.omegaA_bs.T * e_sb

        ### Print time
        psi4.core.print_out(
            f"...finished initializing Sinfinity object in {(time() - t_start):5.2f} seconds.\n"
        )
