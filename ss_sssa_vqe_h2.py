"""
SS-SSSA-VQE (symmetry-guided subspace-search VQE) -- efficient implementation.

Author: Dibyendu Mondal

implementation of ``h2_a1g_triplet_last.py`` (H2 / 6-31G, 3A1g sector) following
Mondal & Maitra, PCCP 28, 10865 (2026):

  Sec. 2.3.1  symmetry/spin-adapted multi-determinantal references (Eqs. 8-12)
  Sec. 2.3.2  symmetry-scalar, spin-free generalized UCC generators (Eqs. 14-19),
              pruned to those acting non-trivially on a reference determinant
  Eq. 22      weighted energy functional  E = w_A <psi_A|H|psi_A> + w_B <psi_B|H|psi_B>

Qiskit-Nature is used only to build the second-quantized operators and their Jordan-Wigner
mappings (identical to those of the original script).  The circuit/Estimator
machinery is replaced by a direct statevector engine that applies exactly the same
first-order Lie-Trotter sequence of Pauli rotations that Qiskit's EvolvedOperatorAnsatz
synthesizes, so every energy printed during the optimization reproduces the original.
"""
import numpy as np
from scipy.optimize import minimize
from pyscf import gto, scf, symm, fci, ao2mo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.properties import AngularMomentum

# ------------------------------------------------------------------ settings
ATOMS = [["H", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, 1.1)]]
BASIS = "6-31g"
POINT_GROUP = "Dooh"
FCI_IRREPS = ["A1g", "A1u"]   # sectors printed by the symmetry-adapted PySCF FCI
NROOTS_FCI = 5

# Reference states (Eqs. 9-12): each entry is a spatial-orbital pair (chi_m, chi_e) whose
# direct product gives the target irrep Sigma (here A1g: sg x sg*, su x su*).
REF_PAIRS = [(0, 2), (1, 3)]
SPIN = "triplet"              # "singlet" -> (|X> + |Y>)/sqrt2, "triplet" -> (|X> - |Y>)/sqrt2
WEIGHTS = (0.65, 0.35)        # w_A > w_B: psi_A -> lowest root, psi_B -> next root
N_REPEAT = 2                  # repetitions of the symmetry-scalar unitary
GRADIENT = "fd"               # "fd": SciPy finite differences (step-by-step identical to the
                              #       original);  "adjoint": exact analytic gradient (faster)
PRINT_EVERY_EVAL = True       # print (E_A, E_B) at every cost evaluation, like the original
N_FCI_PRINT = 20


# ------------------------------------------------------------------ orbital symmetries
mol = gto.M(atom=ATOMS, basis=BASIS, symmetry=POINT_GROUP)
print(mol.topgroup, mol.groupname)

mf = scf.RHF(mol).run()
fs = fci.FCI(mol, mf.mo_coeff)
for irrep in FCI_IRREPS:
    fs.wfnsym = irrep
    e_sym, _ = fs.kernel(nroots=NROOTS_FCI)
    print(f"{irrep} Root", e_sym)

mo = symm.symmetrize_orb(mol, mf.mo_coeff)
orb_sym = list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo))
n = len(orb_sym)                 # spatial orbitals; alpha = 0..n-1, beta = n..2n-1
sym = orb_sym + orb_sym          # irrep label of every spin orbital
print("num of orbitals", n, " num of electrons", sum(mol.nelec))
print(sym)


# ------------------------------------------------------------------ reference states
def ref_dets(m, e):
    """The two complementary determinants |chi_e chi_m~> and |chi_m chi_e~> (Eqs. 9-10)."""
    return (frozenset({e, n + m}), frozenset({m, n + e}))


REF_DETS = [d for m, e in REF_PAIRS for d in ref_dets(m, e)]
print("reference pairs:", [(f"{m}({orb_sym[m]})", f"{e}({orb_sym[e]})") for m, e in REF_PAIRS])


# ------------------------------------------------------------------ symmetry-scalar generators
def sym_ok(h0, h1, p0, p1):
    s = sym
    return ((s[h0] == s[h1] and s[p0] == s[p1]) or (s[h0] == s[p0] and s[h1] == s[p1])
            or (s[h0] == s[p1] and s[h1] == s[p0]))


def spin_free_generators():
    """Spin-free generalized excitations in the original operator order; each group is the
    list of spin-orbital excitations whose sum is one symmetry-scalar generator."""
    groups = []
    # E_i^a = tau_i^a + tau_i~^a~                                            (Eq. 14)
    for i in range(n):
        for a in range(i + 1, n):
            if sym[i] == sym[a]:
                groups.append([((i,), (a,)), ((n + i,), (n + a,))])
    # paired doubles E_ii^aa = tau_ii~^aa~                                   (Eq. 18)
    for i in range(n):
        for a in range(i + 1, n):
            if sym_ok(i, n + i, a, n + a):
                groups.append([((i, n + i), (a, n + a))])
    # two-component doubles                                                  (Eqs. 16-17)
    for i in range(n):                                   # same hole: tau_ii~^ab~ + tau_ii~^ba~
        for a in range(i + 1, n):
            for b in range(a + 1, n):
                if sym_ok(i, n + i, a, n + b):
                    groups.append([((i, n + i), (a, n + b)), ((i, n + i), (b, n + a))])
    for i in range(n):                                   # same particle: tau_ij~^aa~ + tau_ji~^aa~
        for a in range(i + 1, n):
            for j in range(i + 1, a):
                if sym_ok(i, n + j, a, n + a):
                    groups.append([((i, n + j), (a, n + a)), ((j, n + i), (a, n + a))])
    for i in range(n):                                   # hole = particle: tau_ia~^ab~ + tau_ai~^ba~
        for a in range(i + 1, n):
            for b in range(a + 1, n):
                if sym_ok(i, n + a, a, n + b):
                    groups.append([((i, n + a), (a, n + b)), ((a, n + i), (b, n + a))])
    # four-component doubles  tau_ij^ab + tau_ji~^ba~ + tau_ij~^ab~ + tau_i~j~^a~b~  (Eq. 15)
    for i in range(n):
        for a in range(i + 1, n):
            for j in range(i + 1, n):
                for b in range(j + 1, n):
                    if len({i, j, a, b}) == 4 and sym_ok(i, j, a, b):
                        aa = ((i, j), (a, b))
                        ab = ((i, n + j), (a, n + b))
                        ba = ((j, n + i), (b, n + a))
                        bb = ((n + i, n + j), (n + a, n + b))
                        groups.append([aa, ba, ab, bb])
    return groups


def acts_on_reference(exc):
    """True if the excitation acts non-trivially on at least one reference determinant."""
    holes, parts = exc
    return any(all(h in d for h in holes) and not any(p in d for p in parts) for d in REF_DETS)


# Paper rule: keep a generator if any of its components acts on a reference determinant.
# (The original script tested only one component per generator; for H2 both give this pool.)
groups = [g for g in spin_free_generators() if any(acts_on_reference(c) for c in g)]
print("symmetry-scalar generators (components):")
for comps in groups:
    print("  ", comps)


# ------------------------------------------------------------------ qubit Hamiltonian
# MO phases from the eigensolver are arbitrary (they even change with the number of BLAS
# threads), and flipping one flips the sign of the optimal amplitudes.  Fix them: the first
# non-negligible AO coefficient of every MO is positive.  This reproduces the phases that
# Qiskit's PySCFDriver produced in the reference run of the original script.
C = mf.mo_coeff.copy()
for k in range(n):
    if C[np.argmax(np.abs(C[:, k]) > 1e-8), k] < 0:
        C[:, k] *= -1
h1_mo = C.T @ mf.get_hcore() @ C
eri_mo = ao2mo.restore(1, ao2mo.full(mol, C, aosym=4), n)   # same integrals as PySCFDriver
mapper = JordanWignerMapper()
nso = 2 * n
shift = mol.energy_nuc()
print("Number Of Particles:", mol.nelec, " Number of Spin Orbitals:", nso)
print("Nuclear Repulsion Energy:", shift)

H = mapper.map(ElectronicEnergy.from_raw_integrals(h1_mo, eri_mo).second_q_op())
H = H.to_matrix(sparse=True).tocsr()
S2 = mapper.map(AngularMomentum(n).second_q_ops()["AngularMomentum"])
S2 = S2.to_matrix(sparse=True).tocsr()
dim = H.shape[0]


def generator(exc):
    """i (T - T^dagger), exactly as qiskit_nature's UCC builds it."""
    label = " ".join([f"+_{k}" for k in exc[0]] + [f"-_{k}" for k in exc[1]])
    op = FermionicOp({label: 1}, num_spin_orbitals=nso)
    return 1j * (op - op.adjoint())


qubit_gens = []
for comps in groups:
    op = generator(comps[0])
    for c in comps[1:]:
        op = op + generator(c)
    qubit_gens.append(mapper.map(op))


# ------------------------------------------------------------------ statevector engine
def pauli_action(label):
    """Return (perm, phase) such that P @ psi == phase * psi[perm]."""
    nq = len(label)
    x = z = ny = 0
    for q, ch in enumerate(reversed(label)):          # Qiskit labels: rightmost char = qubit 0
        if ch in "XY":
            x |= 1 << q
        if ch in "ZY":
            z |= 1 << q
        ny += ch == "Y"
    idx = np.arange(1 << nq)
    perm = idx ^ x
    parity = np.array([bin(k & z).count("1") & 1 for k in range(1 << nq)])
    phase = (1j ** ny) * (1 - 2 * parity[perm])
    return perm, phase


# Lie-Trotter rotations exp(-i theta_k c P), in the order EvolvedOperatorAnsatz applies them
n_ops = len(qubit_gens)
n_params = N_REPEAT * n_ops
ROT = []
cnot = 0
for r in range(N_REPEAT):
    for k, op in enumerate(qubit_gens):
        for lab, c in op.to_list():
            perm, phase = pauli_action(lab)
            ROT.append((r * n_ops + k, float(np.real(c)), perm, phase))
            cnot += 2 * (sum(ch != "I" for ch in lab) - 1)


def reference_state(m, e, spin):
    """(|chi_e chi_m~> -/+ |chi_m chi_e~>)/sqrt2 -- the state the original H/CNOT circuit makes."""
    v = np.zeros(dim, complex)
    det_x, det_y = ref_dets(m, e)
    v[sum(1 << q for q in det_x)] = 1 / np.sqrt(2)
    v[sum(1 << q for q in det_y)] = (-1 if spin == "triplet" else 1) / np.sqrt(2)
    return v


PSI0 = np.column_stack([reference_state(m, e, SPIN) for m, e in REF_PAIRS])
W = np.asarray(WEIGHTS)
# state prep of one reference: X gates + H on one qubit + CNOTs to the other differing qubits
cnot += len(ref_dets(*REF_PAIRS[0])[0] ^ ref_dets(*REF_PAIRS[0])[1]) - 1
print("Number of parameters", n_params, " Pauli rotations", len(ROT), " Cnot gate", cnot)


def evolve(theta):
    psi = PSI0.copy()
    for k, c, perm, phase in ROT:
        a = theta[k] * c
        psi = np.cos(a) * psi - 1j * np.sin(a) * (phase[:, None] * psi[perm])
    return psi


def expval(op, psi):
    return np.real(np.sum(psi.conj() * (op @ psi), axis=0))


def cost(theta):
    E = expval(H, evolve(theta))
    if PRINT_EVERY_EVAL:
        print(*(E + shift))
    return float(W @ E)


def cost_and_grad(theta):
    """Weighted cost and its exact gradient (adjoint / reverse-mode differentiation)."""
    psi = evolve(theta)
    hpsi = H @ psi
    E = np.real(np.sum(psi.conj() * hpsi, axis=0))
    lam = hpsi * W
    grad = np.zeros(n_params)
    for k, c, perm, phase in reversed(ROT):
        ppsi = phase[:, None] * psi[perm]
        grad[k] += 2 * c * np.imag(np.vdot(lam, ppsi))
        a = theta[k] * c
        cs, sn = np.cos(a), 1j * np.sin(a)
        psi = cs * psi + sn * ppsi
        lam = cs * lam + sn * (phase[:, None] * lam[perm])
    if PRINT_EVERY_EVAL:
        print(*(E + shift))
    return float(W @ E), grad


# ------------------------------------------------------------------ run
E_ref, S_ref = expval(H, PSI0) + shift, expval(S2, PSI0)
for s in range(len(REF_PAIRS)):
    print(f"Reference {s + 1}: energy {E_ref[s]}  <S^2> {S_ref[s]}")

x0 = np.zeros(n_params)
if GRADIENT == "adjoint":
    out = minimize(cost_and_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 50000}, tol=1e-8)
else:
    out = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 50000}, tol=1e-8)
print("cost_function", "-" * 56, "\n", out)
print("optimal_point", out.x)

psi = evolve(out.x)
E_opt, S_opt = expval(H, psi) + shift, expval(S2, psi)
for s, name in enumerate(["GS State", "Ex State"] + [f"State {k}" for k in range(3, len(REF_PAIRS) + 1)]):
    print(f"{name} {E_opt[s]}  spin {S_opt[s]}")

# ------------------------------------------------------------------ exact reference
# Diagonalize H in the N = sum(nelec) sector (same as NumPyEigensolver + particle filter).
nelec = sum(mol.nelec)
sector = np.array([k for k in range(dim) if bin(k).count("1") == nelec])
Hs = H[sector][:, sector].toarray()
evals, evecs = np.linalg.eigh(Hs)
s2 = np.real(np.einsum("ik,ij,jk->k", evecs.conj(), S2[sector][:, sector].toarray(), evecs))
print("FCI Results", "-" * 50)
print(f"{'root':>4} {'E_total (Ha)':>18} {'<S^2>':>8}")
for k in range(min(N_FCI_PRINT, len(evals))):
    print(f"{k:4d} {evals[k] + shift:18.12f} {s2[k]:8.4f}")
