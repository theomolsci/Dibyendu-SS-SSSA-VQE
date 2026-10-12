# Dibyendu-SS-SSSA-VQE
This folder contains the code, results and plots for ground and excited state energies for Subspace Search Spatial and Spin Symmetry-Adapted VQE (SS-SSSA-VQE)

This repository contains a Python implementation of the **SS-SSSA-VQE** (spatial- and spin-symmetry-adapted subspace-search variational quantum eigensolver) algorithm, published in:

> D. Mondal and R. Maitra, *Determination of molecular excited states via symmetry guided subspace search variational quantum eigensolver*, **Phys. Chem. Chem. Phys.**, 2026, **28**, 10865–10877. DOI: [10.1039/d5cp04998a](https://doi.org/10.1039/d5cp04998a)

The example computes the two lowest **³A₁g** (triplet, totally symmetric) eigenstates of **H₂** in the **6-31G** basis (8 qubits) at R = 1.1 Å.

---

## Idea in brief

Conventional weighted subspace-search VQE targets several eigenstates at once by evolving orthogonal references with one common unitary. As more states are targeted, the cost landscape gets harder and the optimizer more often gets stuck in local minima. SS-SSSA-VQE avoids this by exploiting molecular **point-group** and **spin** symmetry:

1. **Partition the Hilbert space by symmetry.** Each (spatial irrep Σ, spin S) sector is treated independently, and states in different sectors are orthogonal by construction.
2. **Symmetry-adapted references.** Within a sector, two or more orthogonal multi-determinant references are built from orbital pairs whose irreps multiply to Σ, combined as singlet `(|X⟩ + |Y⟩)/√2` or triplet `(|X⟩ − |Y⟩)/√2` (Eqs. 8–12).
3. **Symmetry-scalar, spin-free unitary.** Generalized single and double excitations are kept only if they are totally symmetric. They are grouped into spin-free generators (Eqs. 14–18) and pruned to those acting on at least one reference determinant.
4. **Weighted cost.** All references evolve under the same unitary, and `E = w_A⟨ψ_A|H|ψ_A⟩ + w_B⟨ψ_B|H|ψ_B⟩` is minimized (Eq. 22). With `w_A > w_B`, ψ_A converges to the lowest root of the sector and ψ_B to the next one.

The result is fewer parameters, shallower circuits and fewer function evaluations than UCCGSD-based weighted SS-VQE, VQD or qEOM.

## Citation

If you use this code, please cite:

```bibtex
@article{Mondal2026SSSSAVQE,
  author  = {Mondal, Dibyendu and Maitra, Rahul},
  title   = {Determination of molecular excited states via symmetry guided subspace search variational quantum eigensolver},
  journal = {Phys. Chem. Chem. Phys.},
  year    = {2026},
  volume  = {28},
  pages   = {10865--10877},
  doi     = {10.1039/d5cp04998a}
}
```
