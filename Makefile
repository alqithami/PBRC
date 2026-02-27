.PHONY: sim

# Reproduce all simulation figures/tables used in the paper.
sim:
	PYTHONPATH=src python -m pbrc.experiments.run_all --out results/sim --seed 0

