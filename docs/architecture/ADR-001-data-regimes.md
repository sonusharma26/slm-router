# ADR-001: Match evaluation to the observation regime

**Status:** Accepted — 2026-07-27 (`V2-006`)

Full-information matrices use direct held-out replay and paired uncertainty estimates. Contextual-bandit logs may use IPS, SNIPS, DR, or SWITCH only when exact propensities, support, overlap, effective sample size, delayed linkage, and provenance checks pass. Sequential plans use simulation, shadowing, and bounded randomized canaries; v2.0 makes no general offline-RL claim.

These regimes are declared on evaluation inputs and cannot share an implicit evaluator. This replaces FQE on complete oracle matrices.
