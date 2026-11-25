#!/usr/bin/env bash
# Usage: bash scripts/run_experiments.sh   (edit the stages you want; each line is independent)
set -euo pipefail
export PYTHONPATH=src

# --- E0: data ---------------------------------------------------------------
# Default = streaming (no preprocessing): train/eval read data/raw/BraTS2021 lazily.
# Optional: build only the split + slice index up front (also done automatically).
# python -m samri.data.stream --raw data/raw/BraTS2021 --out data/brats21_meta --layout brats21 --workers 8
# Optional: preprocessed uint8 memmaps (faster epochs, more disk), then add `--set data.mode=processed`.
# python -m samri.data.preprocess --raw data/raw/BraTS2021 --out data/processed/brats21 --layout brats21 --workers 8

# --- E1/E2: zero-shot SAM and MedSAM (no training) ---------------------------
python -m samri.eval --config configs/e1_zeroshot_sam.yaml
python -m samri.eval --config configs/e2_medsam.yaml

# --- E4/E5/E6: trained models, 3 seeds each ----------------------------------
for cfg in e4_naive_lora e5a_stem e5b_fusion e6a_stem_25d; do
  for seed in 1 2 3; do
    name=${cfg}_s${seed}
    python -m samri.train --config configs/${cfg}.yaml --set seed=${seed} name=${name}
    python -m samri.eval  --config configs/${cfg}.yaml --set seed=${seed} name=${name} --ckpt runs/${name}/best.pt
  done
done

# --- E7: low-data curves (example for one method) ----------------------------
for frac in 0.05 0.1 0.25 0.5; do
  for seed in 1 2 3; do
    for cfg in e4_naive_lora e5a_stem; do
      name=${cfg}_f${frac}_s${seed}
      python -m samri.train --config configs/${cfg}.yaml --set seed=${seed} name=${name} data.train_fraction=${frac}
      python -m samri.eval  --config configs/${cfg}.yaml --set seed=${seed} name=${name} data.train_fraction=${frac} --ckpt runs/${name}/best.pt
    done
  done
done

# --- E8: external test (real NIfTI, no preprocessing; point at the 2nd dataset) --------------
# python -m samri.eval --config configs/e5a_stem.yaml --ckpt runs/e5a_stem_s1/best.pt \
#   --set name=e5a_stem_s1_ext data.mode=stream data.raw=data/raw/External data.split_mode=all-test \
#         data.splits=data/external_meta/splits.json data.index=data/external_meta/index.csv

python -m samri.report summary runs/*/results_test.csv
