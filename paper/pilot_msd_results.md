# Pilot run — MSD Task01 brain tumour (real data)

> **Status: pilot only.** BraTS 2021 is not present on this machine and Synapse credentials were
> unavailable, so this run uses the public **Medical Segmentation Decathlon Task01** brain-tumour
> set (CC-BY-SA 4.0, no login). MSD Task01 is derived from BraTS 2016/17 and overlaps BraTS
> training years, so **it must not be used as an external test set and these numbers are not the
> paper's BraTS 2021 result.** Its purpose is to exercise the full real-data pipeline and produce
> a first honest learning signal.

## Setup

| Item | Value |
|---|---|
| Data | MSD Task01 BrainTumour, converted to BraTS layout (`scripts/prepare_msd.py`), labels remapped 1→2, 2→1, 3→4 |
| Cases | 484 labelled; patient split 339 / 48 / 97 (train/val/test, seed 42) |
| Training subset | 41 patients (12% of train), seeded nested subset |
| Model | SAM ViT-B + 4→3 channel stem + LoRA (r=8) + region-conditioned decoder (WT/TC/ET), 4 sequences |
| Trainable params | 4.35 M |
| Training | 2 epochs × 1,328 steps, batch 6, AdamW, cosine schedule, bf16 autocast |
| Hardware | 1× NVIDIA L40S (46 GB), ~33 min/epoch |
| Prompts | GT-derived tight box (+ jitter 0/10 px in the 1024 frame); 2D per-slice inference stacked to 3D |
| Checkpoint | epoch 1 (best 2D val Dice) |

## Model selection (validation, 48 patients)

| Epoch | 2D val Dice (160 sampled items, box jitter 5) |
|---|---|
| 0 | 0.811 |
| 1 | **0.829** |

## Held-out test (97 patients, 3D Dice / HD95)

| Region | Dice @ jitter 0 | HD95 mm | Dice @ jitter 10 | HD95 mm |
|---|---|---|---|---|
| WT | **0.887 ± 0.060** (median 0.904) | 4.47 (median 2.83) | 0.886 ± 0.061 | 4.47 |
| TC | **0.816 ± 0.162** (median 0.871) | 4.37 (median 3.16) | 0.812 ± 0.162 | 4.41 |
| ET | **0.784 ± 0.178** (median 0.832) | 3.23 (median 2.24) | 0.778 ± 0.182 | 3.21 |

Details:
- Empty ET in 4/97 test patients; scored Dice 1 / HD95 0 by convention. Excluding them:
  ET Dice **0.775 ± 0.176**, HD95 3.37 mm.
- Jitter 10 costs little: −0.002 (WT), −0.005 (TC), −0.006 (ET) Dice.
- Predicted/GT volume ratio (median): WT 1.006, TC 1.080, ET 1.082 → slight over-segmentation.
- No patient had a zero-voxel prediction for WT/TC; the 4 zero-ET predictions are the empty-GT cases.

## Files

| File | Contents |
|---|---|
| `runs/pilot_msd/log.json` | per-epoch train loss / val 2D Dice / wall time |
| `runs/pilot_msd/results_val_epoch0.csv` | 288 val rows (48 patients × 3 regions × 2 jitters) |
| `runs/pilot_msd/results_test.csv` | 582 test rows (97 patients) |
| `runs/pilot_msd/summary_val_epoch0.csv`, `summary_test.csv` | the aggregated tables above |
| `runs/pilot_msd/epoch0.pt`, `epoch1.pt` | snapshots; `best.pt` = epoch 1 |
| `runs/pilot_msd_train.log`, `runs/pilot_msd_eval_*.log` | raw logs |

## Caveats

- Prompted segmentation with **GT-derived boxes**, not automatic detection; do not compare
  directly to nnU-Net (automatic) without stating this.
- Only 41 training patients and 2 epochs — a lower bound on the eventual BraTS result.
- No MedSAM / nnU-Net / naive-RGB baselines were run in this pilot.
- Val 3D metrics shown above are from the epoch-1 checkpoint; the epoch-0 val 3D snapshot gave
  WT 0.896 / TC 0.792 / ET 0.769 (48 patients), consistent with the test table.

## Reproduce

```bash
# data (public, no login)
curl -L -o data/msd/Task01_BrainTumour.tar \
  https://huggingface.co/datasets/Novel-BioMedAI/Medical_Segmentation_Decathlon/resolve/main/Task01_BrainTumour.tar
tar -xf data/msd/Task01_BrainTumour.tar -C data/msd
PYTHONPATH=src python scripts/prepare_msd.py --msd data/msd/Task01_BrainTumour --out data/raw/MSD --workers 8
PYTHONPATH=src python -m samri.data.stream --raw data/raw/MSD --out data/msd_meta \
  --layout brats21 --workers 8 --size 256 --cache data/msd_cache

# train + evaluate
PYTHONPATH=src python -m samri.train --config configs/pilot_msd.yaml \
  --set name=pilot_msd train.epochs=2 data.train_fraction=0.12 train.batch_size=6
PYTHONPATH=src python -m samri.eval --config configs/pilot_msd.yaml \
  --set name=pilot_msd eval.batch_size=8 --split test \
  --ckpt runs/pilot_msd/epoch1.pt --out runs/pilot_msd/results_test.csv
```
