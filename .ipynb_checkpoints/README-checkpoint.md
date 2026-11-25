# sam-mri: multi-sequence promptable SAM for brain-MRI segmentation (BraTS)

Research codebase for the project: **does a multi-sequence input adapter (and 2.5D context) make SAM a better
promptable MRI segmenter, especially with little labeled data and noisy prompts?**
Baselines: zero-shot SAM, MedSAM, a fair LoRA-fine-tuned SAM with naive RGB input, and nnU-Net.

> **Status:** the data pipeline, metrics, prompts and configs are unit-tested (`tests/test_core.py`, no GPU needed).
> The PyTorch parts (model, train, eval) **have now been executed on real MRI data** (1× L40S):
> `tests/smoke_torch.py` passes, and a real pilot on MSD Task01 brain tumour trained end-to-end.
> Held-out 3D Dice (97 patients, GT-derived box prompts): **WT 0.887 / TC 0.816 / ET 0.784**.
> See `paper/pilot_msd_results.md` and §2.1 of `plan.md`. BraTS 2021 itself is still missing on
> this machine (no Synapse credentials), so these are pilot numbers, not the paper result.

## 0. Setup (on the L40S machine)
```bash
pip install -r requirements.txt && pip install -e .
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# MedSAM: download medsam_vit_b.pth from the MedSAM GitHub README -> checkpoints/medsam_vit_b.pth
PYTHONPATH=src python tests/test_core.py      # torch-free checks
PYTHONPATH=src python tests/smoke_torch.py    # tiny-SAM end-to-end check. Do not skip.
```

## 1. Data
Register on Synapse for BraTS 2021 (adult glioma, ~1,250 cases), put cases in
`data/raw/BraTS2021/<case>/<case>_{t1,t1ce,t2,flair,seg}.nii(.gz)`.

**No BraTS yet?** The MSD Task01 brain-tumour set is public (no login) and BraTS-derived; use it
as a pipeline pilot with `scripts/prepare_msd.py` + `configs/pilot_msd.yaml`
(see `paper/pilot_msd_results.md`). Do not report it as a BraTS 2021 result, and never use it as
an external test set (it overlaps BraTS training years).

**Default: streaming mode (no preprocessing).** Training and evaluation read the real NIfTI
volumes lazily, normalise each slice to uint8 in the worker, and write a per-patient uint8 cache on
first touch (`data/stream_cache/`, ~30-40 MB/patient). The first epoch converts; later epochs
memory-map, so it is fast without a separate preprocessing step. On first run the patient split and
slice index are also built once into `data/brats21_meta/`.

```bash
# optional: build the split + slice index (+ seg cache) ahead of time (train/eval do this too)
python -m samri.data.stream --raw data/raw/BraTS2021 --out data/brats21_meta --layout brats21 --workers 8
python -m samri.train --config configs/e5a_stem.yaml --set seed=1 name=e5a_stem_s1
```

**Optional: preprocessed mode** (memmapped uint8 arrays; faster epochs, more disk). Use
`data.mode=processed`; per sequence percentile clip (0.5-99.5 over brain voxels) + min-max to
uint8 (brain 1..255, background 0), in-plane padded to 256x256, tumour extent ±2 slices.
```bash
python -m samri.data.preprocess --raw data/raw/BraTS2021 --out data/processed/brats21 --layout brats21 --workers 8
python -m samri.train --config configs/e5a_stem.yaml --set data.mode=processed seed=1 name=e5a_stem_s1
```
`data/splits.json` = fixed patient-level 70/10/20 split (seed 42). **It is never overwritten unless
`--force-split`.** No synthetic data is used for training: `tests/smoke_torch.py` writes a
throwaway toy volume under `/tmp` only to exercise the code, never into `data/`.
External test set: preprocess a second glioma dataset into another folder with `--split-mode all-test`
(check the file naming; adapt `LAYOUTS` in `preprocess.py` if it differs). Do not use MSD Task01 (derived from BraTS).

## 2. Experiments (`bash scripts/run_experiments.sh` runs them all; every line also works alone)
| ID | Config | What |
|---|---|---|
| E1 | `e1_zeroshot_sam` | SAM zero-shot, GT-derived box/click prompts |
| E2 | `e2_medsam` | MedSAM zero-shot (box) |
| E3 | `scripts/prepare_nnunet.py`, `eval_nnunet.py` | nnU-Net 3D on the identical split (hard baseline) |
| E4 | `e4_naive_lora` | SAM + LoRA + decoder, T1ce/T2/FLAIR as RGB (fair baseline) |
| E5a | `e5a_stem` | 4 sequences -> learned 4->3 stem (starts equal to E4) |
| E5b | `e5b_fusion` | each sequence through the encoder, embeddings fused (4x cost) |
| E6 | `e6a_stem_25d`, `e6b_fusion_25d` | + neighbouring slices |
| E7 | `--set data.train_fraction=0.05 ...` | low-data curves (nested subsets) |
| E8 | external test + `eval.jitter` | generalization and prompt robustness |

```bash
python -m samri.train --config configs/e5a_stem.yaml --set seed=1 name=e5a_stem_s1
python -m samri.eval  --config configs/e5a_stem.yaml --set seed=1 name=e5a_stem_s1 --ckpt runs/e5a_stem_s1/best.pt
python -m samri.report summary runs/*/results_test.csv
python -m samri.report compare runs/e4_naive_lora_s1/results_test.csv runs/e5a_stem_s1/results_test.csv   # paired Wilcoxon
```
Do E0-E4 before your own method. Run 3 seeds (`seed=1,2,3`) for every trained model.

## 3. Evaluation protocol (write this in the paper, exactly)
- **Oracle prompts:** for each axial slice with non-empty GT region, one prompt is simulated from GT
  (tight box + uniform jitter of 0/5/10/20 px in the 1024 frame, or the deepest interior click).
  Slices with empty GT get no prediction. This is the MedSAM-style protocol, **not** fully automatic segmentation,
  so do not compare it head-to-head with nnU-Net's automatic output without saying so.
- SAM is binary: WT, TC, ET are three separate prompted tasks.
- Predictions are stacked to 3D; Dice and HD95 (mm, voxel spacing) per patient and region.
  Both-empty -> Dice 1 / HD95 0; one-empty -> Dice 0 / HD95 373.13 (BraTS convention). Report how many patients have empty ET.
- Statistics: mean +- std over patients (seeds averaged per patient first), paired Wilcoxon vs baseline.
- Deviation from a naive recipe: tumor-free slices are not trained on (a box prompt is undefined for them).

## 4. Known limits / things to decide
- Contribution must be one sharp claim, e.g. "multi-sequence adapter X beats naive RGB at <=25% labels, same prompts".
  Ablate E4 vs E5a/b vs E6; the human-fallback / auto-prompt ideas are out of scope here.
- E1/E2 use the naive 3-channel mapping; also report a grayscale variant (FLAIR for WT/TC, T1ce for ET) so
  baselines are not strawmen.
- No clinical claims: "benchmark segmentation performance" only.
- `adapter_lr`, LoRA rank, epochs are starting values; tune on the validation split only. Never on test.
