#!/usr/bin/env bash
set -euo pipefail
python -m pbrc.experiments.run_all --out results/sim --seed 0
echo "Done. See results/sim/"
