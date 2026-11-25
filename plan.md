# Plan — sam-mri: multi-sequence promptable SAM for brain-MRI (BraTS)

> **Status:** draft / living document. This is *just a plan* — reorder, cut, or rewrite any
> section. Nothing here is locked in. Last updated: 2026-11-24.

---

## 0. TL;DR

Build and validate the claim: **a multi-sequence input adapter (+ optional 2.5D context) makes
SAM a better promptable MRI segmenter than naive RGB, especially with little labeled data and
noisy prompts** — with fair baselines (zero-shot SAM, MedSAM, LoRA-SAM, nnU-Net), 3 seeds,
patient-level statistics, and a clear reproduction path.

Separately (optional deliverable, see §7): turn the 2D/2.5D slice model into an **interactive
2D→3D viewer** where a user prompt on a slice is propagated across slices and stacked into a 3D
mask for clear, near-real-time visualization. Today the repo only stacks 2D predictions into 3D
offline during evaluation — it is *not* an interactive real-time tool.

---

## 1. Direct answer: "does this project turn 2D segments into 3D segments from a user prompt?"

**Not as an interactive, real-time visualization tool — not yet.** Precisely:

- **Backbone is 2D/2.5D.** Preprocessing stores axial slices `img.npy (4, D, S, S)` and
  `seg.npy (D, S, S)`. SAM encodes one slice; `context: 1` feeds slices `z-1, z, z+1` as extra
  input channels (2.5D), but the output is still a per-slice 2D mask.
- **3D is assembled only in evaluation.** `src/samri/eval.py` loops over GT-positive slices,
  predicts a 2D mask for each, writes them into a 3D `pred` array (`pred[zb] = ...`), then scores
  3D Dice/HD95. Prompts there are **simulated from the GT**, not user prompts.
- **No interactive viewer / realtime path exists.** No renderer, no viewer, no live inference
  service, no prompt→propagate→volume module. Slices with empty GT get no prediction, so it is
  also not a fully-automatic 3D segmenter.

So: it is a *slice-wise promptable segmenter evaluated as a 3D volume*. Making the interactive
"prompt on 2D → build 3D volume → visualize in realtime" experience is a **new feature** and is
planned in §7, not something the current code already does.

---

## 2. What is already done (this session)

- **Removed synthetic data** under `data/smoke_raw/` and `data/smoke_processed/` (the
  `Synth_0000…0005` fixtures). `tests/smoke_torch.py` regenerates them at runtime.
- **Model improvement #1 — region-conditioned decoding** (`src/samri/models/sam_wrapper.py`):
  zero-initialized WT/TC/ET embedding added to prompt tokens in `decode()`; enabled via
  `model.region_cond` (default `true` in `configs/base.yaml`). Starts as a no-op.
- **Model improvement #2 — region-balanced training sampling** (`src/samri/data/dataset.py`):
  each region repeated to the largest region's count per epoch (deterministic by seed).
- **Wiring:** `train.py` / `eval.py` pass `region` through; `SamMRI.forward` accepts it.
- **Verified:** `tests/test_core.py` ✅ and `tests/smoke_torch.py` ✅ (train → eval → report).
- **Skills installed:** K-Dense `scientific-agent-skills` (177 skills) registered globally at
  `~/.config/opencode/kdense-skills/skills` via `~/.config/opencode/opencode.jsonc`
  (there is no literal `kdense-scientist` skill — see §3).

### 2.1 Real-data pilot result — MSD Task01 (done 2026-11-24)

BraTS 2021 is **not on this machine and there are no Synapse credentials**, so the first real
training run used the public **Medical Segmentation Decathlon Task01 BrainTumour** set
(CC-BY-SA 4.0, no login; BraTS-2016/17-derived). It exercises the *entire* real pipeline:
raw NIfTI → lazy stream cache → 2D promptable training → 3D volume eval.

| Region | 3D Dice (97 held-out patients) | HD95 mm |
|---|---|---|
| WT | **0.887 ± 0.060** | 4.47 |
| TC | **0.816 ± 0.162** | 4.37 |
| ET | **0.784 ± 0.178** | 3.23 |

- Model: SAM ViT-B + 4→3 stem + LoRA(r=8) + region-conditioned decoder; 4.35 M trainable.
- 2 epochs × 1,328 steps on 41 train patients (12% subset), batch 6, ~33 min/epoch on L40S.
- Val 2D Dice 0.811 → 0.829 (epoch 0 → 1); test evaluated once with the epoch-1 checkpoint.
- Empty ET in 4/97; excluding them ET Dice = 0.775. Jitter 10 px costs ≤0.006 Dice.
- Full write-up + reproduce commands: `paper/pilot_msd_results.md`; raw rows:
  `runs/pilot_msd/results_test.csv`, `results_val_epoch0.csv`.
- **Caveats:** prompted with GT-derived boxes (not automatic); overlaps BraTS training years, so
  it is a pilot / pipeline check — **never an external test set and not the BraTS 2021 result**.
  Only 2 epochs and 41 patients, so this is a lower bound for the real grid.

---

## 3. Skills to use, by phase

Installed collection: `K-Dense-AI/scientific-agent-skills` (177 skills). There is **no single
`kdense-scientist` skill**; use the workflow-appropriate ones below.

| Phase | Primary skills | Why |
|---|---|---|
| Design / preregister | `experimental-design`, `hypothesis-generation`, `scientific-critical-thinking` | factor layout, controls, confounders, validity |
| Data / EDA | `exploratory-data-analysis`, `get-available-resources` | leakage/missingness audit, resource limits |
| Literature | `literature-review`, `paper-lookup`, `research-lookup` | related work, baseline provenance |
| Training / GPU | `pytorch-lightning` (optional refactor), `optimize-for-gpu`, `arbor` | perf, multi-run tracking |
| Statistics | `statistical-analysis`, `statistical-power` | paired tests, effect sizes, CIs |
| Figures | `scientific-visualization`, `matplotlib`, `seaborn` | publication figures, colour review |
| Writing | `scientific-writing`, `peer-review`, `venue-templates` | manuscript, rebuttal, formatting |
| Viewer (optional) | `plotly`, `pydicom`, `imaging-data-commons` | interactive 3D, DICOM/volume I/O |

Rule: confirm each skill's dependencies before running it; the repo's own README warns to install
only what is needed (we installed the full collection as requested; trim later if noisy).

---

## 4. Goals & success criteria

- **Primary claim:** `E5a/E5b` beat `E4` (naive RGB) at 5–25% labels with identical prompts.
- **Secondary:** 2.5D context (`E6a/E6b`) helps under prompt jitter (10–20 px).
- **Guardrails:** honest baselines, 3 seeds, patient-level paired stats, no test-set tuning,
  report empty-ET counts and HD95 conventions.
- **Deliverables:** reproducible scripts, results tables, statistical report, paper draft,
  optional interactive 2D→3D viewer.

### 4.1 Statistical / design protocol (K-Dense `experimental-design` + `scientific-critical-thinking`)

- **Unit of analysis = patient.** Slices are repeated measures, *not* independent replicates.
  Never report slice-level n as sample size (pseudoreplication). Aggregate per patient first.
- **Design = paired / repeated-measures within patient.** Every patient is segmented by every
  method with the *same* simulated prompts; compare methods with a paired Wilcoxon signed-rank
  test per region.
- **Pre-specified primary endpoint:** Dice for WT/TC/ET at box jitter 0 and 10 px.
  **Secondary:** HD95, jitter {0,5,10,20} and point-prompt robustness, low-data slope.
- **Multiplicity:** Holm–Bonferroni (or BH-FDR) across the pre-specified family; always report
  effect size + 95% CI, not just p.
- **Power:** 3 seeds estimate *model* variance, not statistical power. Power comes from the
  number of test patients (~n≈250); compute the minimum detectable paired effect before test.
- **Nuisance factors to pair/stratify:** patient heterogeneity, tumour size, empty-ET cases,
  scanner/site, available sequences. Stratify low-data subsets by patient; report empty-ET counts.
- **Randomization:** fixed patient-level split (seed 42); deterministic prompt seeds
  `(pid, z, region, jitter)`; model seeds 1/2/3.
- **Controls:** naive RGB (E4), grayscale SAM, MedSAM, nnU-Net; prompted methods get identical
  prompts. State that GT-derived prompts are optimistic (MedSAM-style), not automatic.
- **Val/test discipline:** tune every hyper-parameter on val only; analyse test once; freeze the
  analysis plan (endpoints, tests, multiplicity) before touching test.

---

## 5. Phased plan

### P0 — Environment & data (gate for everything)
- [ ] `pip install -r requirements.txt && pip install -e .`
- [x] Download SAM ViT-B → `checkpoints/sam_vit_b_01ec64.pth`. MedSAM → `checkpoints/medsam_vit_b.pth` still missing.
- [ ] **BLOCKED** Obtain BraTS 2021 (Synapse) into `data/raw/BraTS2021/<case>/` — no local copy
      and no Synapse token on this machine. Needs the user to provide data or credentials.
- [x] **Substitute real dataset (pilot only):** MSD Task01 brain tumour downloaded (public),
      converted to BraTS layout by `scripts/prepare_msd.py`, split 339/48/97 patients
      (`data/msd_meta`), stream cache in `data/msd_cache`. Results in §2.1.
- [x] Real-data **streaming** path: `data.mode: stream` reads NIfTI lazily, no preprocessing; a
      per-patient uint8 cache is written on first touch (`data/stream_cache/`) and memory-mapped
      afterwards. `data/brats21_meta/{splits.json,index.csv}` are built once. (Implemented;
      validated on real NIfTI input + real SAM weights; awaits the BraTS download.)
  ```bash
  python -m samri.data.stream --raw data/raw/BraTS2021 --out data/brats21_meta --layout brats21 --workers 8
  ```
- [ ] Optional faster-epoch path: preprocess to uint8 memmaps and run with `data.mode=processed`.
  ```bash
  python -m samri.data.preprocess --raw data/raw/BraTS2021 \
    --out data/processed/brats21 --layout brats21 --workers 8
  ```
- [ ] (external set) preprocess a second glioma dataset with `--split-mode all-test`.
- [x] `PYTHONPATH=src python tests/test_core.py` green (synthetic data removed; smoke uses `/tmp` only).
- **Exit:** `data/brats21_meta/splits.json` fixed (seed 42), index CSV sane, first real batches load.

### P1 — Baselines E1–E4 (do before the method)
- [ ] E1 zero-shot SAM: `python -m samri.eval --config configs/e1_zeroshot_sam.yaml`
- [ ] E2 MedSAM: `python -m samri.eval --config configs/e2_medsam.yaml`
- [ ] E1/E2 grayscale variants (FLAIR for WT/TC, T1ce for ET) — avoid strawmen.
- [ ] E3 nnU-Net: `scripts/prepare_nnunet.py` → train → `scripts/eval_nnunet.py`.
- [ ] E4 naive LoRA-SAM (RGB), 3 seeds.
- **Exit:** baseline table with honesty notes (prompted vs automatic).

### P2 — Core method E5/E6
- [ ] E5a learned 4→3 stem; E5b per-sequence encoder + fusion.
- [ ] E6a stem + 2.5D; E6b fusion + 2.5D.
- [ ] 3 seeds each; paired Wilcoxon vs E4.

#### Three training tracks (confirmed — T3a chosen)

**T1 — 2D promptable slice segmentation (core).** One slice in, box/point prompt, 2D mask out;
Dice+BCE per region. This is the headline claim (multi-sequence adapter vs naive RGB). *This is
what E4/E5/E6 already are.*

**T2 — 2.5D context + inter-slice consistency.** Feed `z-1, z, z+1` (context=1) and add a
consistency term so adjacent slices agree where anatomy overlaps (mask smoothness along `z`).
Cheap, no new parameters, improves 3D coherence. *E6a/E6b + consistency loss.*

**T3 — 2D→3D aggregation / refinement (the "trained 2D→3D" track).** Two variants:
- **T3a (loss-only — SELECTED):** keep the 2D model, add a **3D volume loss** across the stacked
  per-slice predictions (3D Dice+BCE on a contiguous z-block/patient volume, plus `z`-continuity/TV).
  No new parameters, no extra overfitting risk at ≤25% labels.
- **T3b (learned refiner — deferred/optional):** prompt-conditioned 3D head (z-axis transformer or
  tiny 3D U-Net) over stacked 2D embeddings. Only if T3a plateaus and time allows.

Evaluate T3 as a **prompted 3D pipeline** (prompt on slice(s) → propagate → stack → 3D mask),
and report it separately from nnU-Net (which is automatic, not prompted).

#### Experiment dimensions (all four axes)

You asked to train on **all** of these; they are four independent axes, not four competing plans:

| Axis | Levels | Notes |
|---|---|---|
| **Track** | T1 (2D) · T2 (2.5D+consistency) · T3a (trained 2D→3D) | one model family, three training objectives |
| **Region** | WT · TC · ET | one shared, region-conditioned model (region_cond); eval per region |
| **Input sequences** | all-4 · T1ce/T2/FLAIR (E4 baseline) · single-seq (FLAIR→WT/TC, T1ce→ET) | sequence-ablation axis |
| **Dataset setting** | BraTS internal · external test · low-data {5,10,25,50}% | generalization + data-efficiency axis |

Design rule: run the **full grid for the best track** (T3a) and a **subset grid for T1/T2**
(no need to multiply every cell). Regions are always conditioned + reported separately; sequence
ablations and dataset settings are run on the winning adapter to keep the budget sane.

**Framing guardrail:** keep the *sharp claim* on T1/T2 ("adapter beats naive RGB at low labels").
Treat T3 as an extension/ablation showing the adapter also yields better prompted 3D
reconstruction — do **not** let it dilute the main contribution or claim automatic-segmentation
superiority.

### P3 — Model-improvement ablations (new, from this session)
- [ ] `region_cond` on/off (default now on).
- [ ] balanced region sampling on/off.
- [ ] loss variants: Dice+BCE (current) vs Tversky/focal; deep supervision (optional).
- [ ] 2.5D context 0 vs 1; stem init variants.
- [ ] thoroughness: tune `adapter_lr`, LoRA rank, epochs **on val only**.

### P4 — Low-data curves (E7)
- [ ] `data.train_fraction ∈ {0.05, 0.1, 0.25, 0.5}`, nested subsets (`data.data_seed`), 3 seeds.
- [ ] Run for E4 and E5a at minimum; add E6 if budget allows.
- [ ] Plot Dice vs fraction with CIs.

### P5 — Generalization & prompt robustness (E8)
- [ ] External test set (all-test preprocessing).
- [ ] Prompt jitter `{0,5,10,20}` px + point prompts; report per-jitter curves.
- [ ] Failure analysis: empty ET, tiny lesions, border slices.

### P6 — Statistics & reporting
- [ ] `samri.report summary runs/*/results_test.csv`
- [ ] `samri.report compare A.csv B.csv` (paired Wilcoxon)
- [ ] Use `statistical-analysis` skill: effect sizes, CIs, multiplicity, assumptions.
- [ ] Freeze result tables; archive configs + seeds + git SHA.

### P7 — Paper / report
- [ ] `scientific-writing` for manuscript; `venue-templates` for formatting.
- [ ] `scientific-visualization` for figures (qualitative overlays + curves).
- [ ] `peer-review` for internal mock review before submission.

---

## 6. Experiment matrix (edit freely)

| ID | Config / command | Purpose | Seeds |
|---|---|---|---|
| E1 | `configs/e1_zeroshot_sam.yaml` | zero-shot SAM + grayscale variant | — |
| E2 | `configs/e2_medsam.yaml` | MedSAM box baseline | — |
| E3 | `scripts/prepare_nnunet.py` + `eval_nnunet.py` | nnU-Net hard baseline | 1 fold |
| E4 | `configs/e4_naive_lora.yaml` | LoRA-SAM, naive RGB | 1,2,3 |
| E5a | `configs/e5a_stem.yaml` | learned 4→3 stem | 1,2,3 |
| E5b | `configs/e5b_fusion.yaml` | per-seq encoder + fusion | 1,2,3 |
| E6a | `configs/e6a_stem_25d.yaml` | stem + 2.5D | 1,2,3 |
| E6b | `configs/e6b_fusion_25d.yaml` | fusion + 2.5D | 1,2,3 |
| E7 | `--set data.train_fraction=…` | low-data curves | 1,2,3 |
| E8 | `--set data.root=<external> …` | external test + jitter | 1,2,3 |
| E9 | `--set model.region_cond=false` | region-cond ablation | on best method |
| E10 | (add flag) balanced-sampling ablation | sampling ablation | on best method |
| E11 | T3a: 3D volume loss (`--set train.vol_loss=true`) **SELECTED** | trained 2D→3D (loss-only) | 1,2,3 |
| E12 | T3b: learned 3D refiner (`configs/e12_3d_refiner.yaml`) | optional / deferred | 1,2,3 |
| E13 | `--set data.sequences=[t1ce,t2,flair]` | sequence ablation (3-seq) | on best |
| E14 | `--set data.sequences=[flair]` / `[t1ce]` | single-sequence ablation | on best |

```bash
# canonical run loop (train + eval), per config and seed
for cfg in e4_naive_lora e5a_stem e5b_fusion e6a_stem_25d; do
  for seed in 1 2 3; do
    name=${cfg}_s${seed}
    python -m samri.train --config configs/${cfg}.yaml --set seed=${seed} name=${name}
    python -m samri.eval  --config configs/${cfg}.yaml --set seed=${seed} name=${name} \
      --ckpt runs/${name}/best.pt
  done
done
python -m samri.report summary runs/*/results_test.csv
```

---

## 7. Optional deliverable: interactive 2D→3D promptable viewer (proposed)

**Goal:** user clicks/drags a prompt on an axial slice → model segments that slice → optionally
propagate to neighbours → stack into a 3D mask → render clearly, near-real-time.

Proposed pieces (all new; nothing exists yet):
- `samri.infer.VolumePredictor`: encode a slice once, cache embeddings, decode per prompt (fast).
- Prompt propagation: decode on the prompted slice, then use the mask centroid/box to prompt
  `z±1, z±2…` until Dice/area stabilizes (or user stops); fill non-prompted slices.
- `samri.viewer`: interactive 3D (Plotly or napari). Show 2D slice + mask, a 3D surface/volume,
  and region overlays (WT/TC/ET). Optional DICOM/NIfTI export via `nibabel`/`pydicom`.
- "Realtime" budget: target < ~1 s per slice on GPU with cached embeddings; document latency.
- Explicitly label it *research/visualization*, not a clinical tool.

Decisions to confirm: Plotly (browser) vs napari (desktop); propagation policy; whether to
expose prompts in 3D or only per-slice; do we ship this in v1 or defer.

---

## 8. Open questions / decision log (change anytime)

| # | Question | Decision / current lean | Revisit when |
|---|---|---|---|
| 0 | Train on "three things" = ? | **ALL**: tracks T1/T2/T3 + regions WT/TC/ET + sequence ablations + dataset settings | budget pressure |
| 0b | 2D→3D variant? | **T3a loss-only** (3D Dice + z-continuity); T3b refiner deferred | if T3a plateaus |
| 0c | Data pipeline? | **stream real BraTS NIfTI** (`data.mode: stream`); no synthetic data, preprocessing optional | if epoch I/O is the bottleneck |
| 1 | Sharpest claim? | "adapter beats naive RGB at ≤25% labels" | after P1 baselines |
| 2 | Region conditioning by default? | yes (zero-init, config-gated) | if ablation is neutral |
| 3 | Balanced sampling by default? | yes | if it distorts val Dice |
| 4 | 2.5D context worth 4× encoder cost? | only for E6b if it wins | after P2 |
| 5 | Ship viewer in v1? | optional / stretch | after P5 |
| 6 | External dataset choice? | second glioma set (not MSD Task01) | P0 |
| 7 | Trim installed skills? | keep for now | if context/noise hurts |

---

## 9. Risks & mitigations

- **No GPU / no data:** full E1–E8 blocked → smoke pipeline is the gate; document exact commands.
- **Unfair baselines:** add grayscale SAM and tuned LoRA; never test-tune.
- **Class imbalance (ET):** balanced sampling + per-region reporting + empty-ET counts.
- **Prompt leakage:** prompts are GT-derived (MedSAM-style); state it, don't claim automatic.
- **Overfitting on val:** nested low-data subsets, 3 seeds, patient-level splits.
- **Reproducibility:** fix seeds, save `config.json` per run, record git SHA + versions.

---

## 10. Definition of done (v1)

- [ ] Reproducible script run from raw BraTS to final tables.
- [ ] E1–E8 (+ E9/E10 ablations) with 3 seeds where trainable.
- [ ] Paired stats + effect sizes + CIs; honest limitations section.
- [ ] Paper draft + figures.
- [ ] (optional) interactive 2D→3D viewer demo.
