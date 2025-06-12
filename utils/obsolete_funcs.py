
# ==============================================================================
# Obsolete functions (kept for historical comparison)
# ==============================================================================

# OBSOLETE: This function is kept for historical comparison.
# The main routine for delta L and M is now compute_delta_LMX.
def compute_delta_LA(CA, CB, h_sapt, D, input_dict):
    """
    Computes the Landshoff delta term for monomer A.

    Parameters:
    - CA (numpy.ndarray): Coefficient matrix of MOs for monomer A (shape: n_basis x nmo_A).
    - CB (numpy.ndarray): Coefficient matrix of MOs for monomer B (shape: n_basis x nmo_B).
    - h_sapt (helper_SAPT): Instance containing necessary data.
    - D (numpy.ndarray): Inverse MO overlap matrix (nocc_total x nocc_total).

    Returns:
    - delta_LA (float): The computed Landshoff delta for monomer A.
    """
    # Get number of MOs and occupied orbitals from h_sapt
    nmo_A = h_sapt.nmo  # Total number of MOs in monomer A
    nmo_B = h_sapt.nmo  # Total number of MOs in monomer B
    nocc_A = h_sapt.ndocc_A  # Number of occupied MOs in monomer A
    nocc_B = h_sapt.ndocc_B  # Number of occupied MOs in monomer B
    nocc_total = nocc_A + nocc_B

    # Extract occupied MO coefficients
    CA_occ = CA[:, :nocc_A]  # Shape: n_basis x nocc_A
    CB_occ = CB[:, :nocc_B]  # Shape: n_basis x nocc_B
    C_occ = np.hstack((CA_occ, CB_occ))  # Shape: n_basis x nocc_total

    # Retrieve the MintsHelper for monomer A
    mintsA = psi4.core.MintsHelper(h_sapt.wfnA.basisset())

    # Compute the kinetic energy and potential matrices
    TA = np.asarray(mintsA.ao_kinetic())
    VA = np.asarray(mintsA.ao_potential())

    # Assemble the core Hamiltonian
    HA = TA + VA

    # Compute the density matrix for monomer A
    DenA = np.einsum('pi, qi -> pq', CA_occ, CA_occ)

    # Compute the AO two-electron integrals for monomer A
    I_AO_A = np.asarray(mintsA.ao_eri())

    # Compute Coulomb and Exchange matrices
    JA = np.einsum('pqrs, rs -> pq', I_AO_A, DenA)
    KA = np.einsum('prqs, rs -> pq', I_AO_A, DenA)

    # Compute the Fock operator
    FA = HA + 2 * JA - KA

    # Transform FA to MO basis
    FA_MO = np.einsum('pi, pq, qj -> ij', CA_occ, FA, C_occ)  # Shape: nocc_A x nocc_total
    if input_dict.get('debug_print_verbose', False):
        print('FA MO partial A + AB :\n', FA_MO)

    # Extract D matrix elements
    D_rp = D[:, :nocc_A]  # Shape: nocc_total x nocc_A

    # First term
    term1 = 2*np.einsum('ij, ji ->', FA_MO, D_rp)
    if input_dict.get('debug_print_verbose', False):
        print('term1 A in old routine:\n', term1 )

    # Second term
    FA_diag = np.einsum('pi, pq, qi -> i', CA_occ, FA, CA_occ)
    term2 = 2*np.sum(FA_diag)
    print('term2 A in old routine:\n', term2) # This print remains as per original for historical code

    delta_LA = term1 - term2
    return delta_LA

# OBSOLETE: This function is kept for historical comparison.
# The main routine for delta L and M is now compute_delta_LMX.
def compute_delta_LB(CA, CB, h_sapt, D, input_dict):
    """
    Computes the Landshoff delta term for monomer B.

    Parameters:
    - CA (numpy.ndarray): Coefficient matrix of MOs for monomer A (shape: n_basis x nmo_A).
    - CB (numpy.ndarray): Coefficient matrix of MOs for monomer B (shape: n_basis x nmo_B).
    - h_sapt (helper_SAPT): Instance containing necessary data.
    - D (numpy.ndarray): Inverse MO overlap matrix (nocc_total x nocc_total).

    Returns:
    - delta_LB (float): The computed Landshoff delta for monomer B.
    """
    # Get number of MOs and occupied orbitals from h_sapt
    nmo_A = h_sapt.nmo  # Total number of MOs in monomer A
    nmo_B = h_sapt.nmo  # Total number of MOs in monomer B
    nocc_A = h_sapt.ndocc_A  # Number of occupied MOs in monomer A
    nocc_B = h_sapt.ndocc_B  # Number of occupied MOs in monomer B
    nocc_total = nocc_A + nocc_B

    # Extract occupied MO coefficients
    CA_occ = CA[:, :nocc_A]  # Shape: n_basis x nocc_A
    CB_occ = CB[:, :nocc_B]  # Shape: n_basis x nocc_B
    C_occ = np.hstack((CA_occ, CB_occ))  # Shape: n_basis x nocc_total

    # Retrieve the MintsHelper for monomer B
    mintsB = psi4.core.MintsHelper(h_sapt.wfnB.basisset())

    # Compute the kinetic energy and potential matrices
    TB = np.asarray(mintsB.ao_kinetic())
    VB = np.asarray(mintsB.ao_potential())

    # Assemble the core Hamiltonian
    HB = TB + VB

    # Compute the density matrix for monomer B
    DenB = np.einsum('pi, qi -> pq', CB_occ, CB_occ)

    # Compute the AO two-electron integrals for monomer B
    I_AO_B = np.asarray(mintsB.ao_eri())

    # Compute Coulomb and Exchange matrices
    JB = np.einsum('pqrs, rs -> pq', I_AO_B, DenB)
    KB = np.einsum('prqs, rs -> pq', I_AO_B, DenB)

    # Compute the Fock operator
    FB = HB + 2 * JB - KB

    # Transform FB to MO basis
    FB_MO = np.einsum('pi, pq, qj -> ij', CB_occ, FB, C_occ)  # Shape: nocc_B x nocc_total
    if input_dict.get("debug_print_verbose", False):
        print("FB MO partial B + AB :\n", FB_MO)

    # Extract D matrix elements
    D_rp = D[:, nocc_A:]  # Shape: nocc_total x nocc_B

    # First term
    term1 = 2*np.einsum('ij, ji ->', FB_MO, D_rp)
    if input_dict.get("debug_print_verbose", False):
        print("DEBUG compute_delta_LB: term1 B in old routine:\n", term1 )

    # Second term
    FB_diag = np.einsum('pi, pq, qi -> i', CB_occ, FB, CB_occ)
    term2 = 2*np.sum(FB_diag)
    print('term2 A in old routine:\n', term2) # This print remains as per original for historical code

    delta_LB = term1 - term2
    return delta_LB

