# PBRC reproducibility and maintenance

Preregistered Belief Revision Contracts for Evidence-Gated Multi-Agent AI Deliberation.

## Branch status

This review branch contains repaired legacy simulation enforcement and regression tests. The complete consolidated code-and-results package has been supplied separately to the maintainer. Its large KAIROS and SciFact records are not yet imported into this branch. See docs/CONSOLIDATED_ARTIFACT_STATUS.md for the exact handoff and provenance.

Do not treat the 100-item examples in KAIROS/eval_results as the submitted 3000-item run. KAIROS/KAIROS/eval_results_pbrc_full is an obsolete placeholder, not a full result directory. The full package provides the real 3000-item logs and the separate SciFact main200 study.

## Install and test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
python -m pytest tests/test_router_hardening.py
python -m pbrc.experiments.run_all --out reproduced/sim --seed 0
```

Simulation reference CSVs and figures remain in results/sim. The repaired code retains their numerical results for the supplied simple contracts. General set predicates now require an actually sufficient, nonempty validated witness before operator execution. Fallback runs once, and unobserved global coverage is None.

## Honest adapter scope

The legacy BenchForm adapter supports only social_only. The former tool_tokens documentation was unsupported. The legacy live KAIROS adapter is an example with an unimplemented model-call hook and unverified schema assumptions, not an end-to-end integration. The genuine evidence-enabled SciFact pipeline is a separate component of the consolidated package.

The benchmark code/data directories currently tracked here are historical vendor copies. New upstream acquisitions use scripts/download_benchmarks.sh and are isolated under external, with repository-identity checks before checkout.

PBRC constrains admission under stated assumptions. It is not a truth oracle. Synthetic default token validation does not authenticate evidence in production. The SciFact result is 53.0% versus 49.0% RAW, statistically inconclusive at n=200. The discarded perfect-result v0.3.x pilot must not be used as evidence.

No repository-wide license is newly granted by this maintenance change. Retain existing third-party licenses and select the original PBRC code license deliberately before claiming a uniform license.
