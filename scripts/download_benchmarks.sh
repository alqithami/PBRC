#!/usr/bin/env bash
set -euo pipefail

# Benchmarks used in the paper's evaluation protocol.
# We pin to specific commits to make results reproducible.
#
# BenchForm (ICLR 2025 Oral): https://github.com/Zhiyuan-Weng/BenchForm
#   pinned commit: 6425ecaebd9dd273a13ea28e32767c452e04c6a6 (Feb 6, 2026)
#
# KAIROS / peer pressure benchmark: https://github.com/declare-lab/KAIROS
#   pinned commit: a22b81cb1c7b448122b6520b4da11c924a242824 (Dec 19, 2025)

if [ ! -d "BenchForm" ]; then
  git clone https://github.com/Zhiyuan-Weng/BenchForm.git
fi
cd BenchForm
git fetch --all
git checkout 6425ecaebd9dd273a13ea28e32767c452e04c6a6
cd ..

if [ ! -d "KAIROS" ]; then
  git clone https://github.com/declare-lab/KAIROS.git
fi
cd KAIROS
git fetch --all
git checkout a22b81cb1c7b448122b6520b4da11c924a242824
cd ..

echo "Done. Follow each benchmark repo's README for environment setup."
