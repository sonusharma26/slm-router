# ADR-002: Feasibility before lexicographic optimization

**Status:** Accepted — 2026-07-27 (`V2-006`)

The v1 scalar reward is removed from the v2 path. The compiler first enforces governance, capability, call, quality-risk, spend, and deadline constraints. It then minimizes quality shortfall, expected cost, tail latency, calls, and complexity in that order. No feasible plan results in the policy's declared abstention or fallback behavior, never an implicit cheapest-model call.
