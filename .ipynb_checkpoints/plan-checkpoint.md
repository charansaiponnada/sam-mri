# Plan — sam-mri: multi-sequence promptable SAM for brain-MRI (BraTS)

> **Status:** draft / living document. This is *just a plan* — reorder, cut, or rewrite any
> section. Nothing here is locked in. Last updated: 2026-11-19.

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

---

## 5. Phased plan

### P0 — Environment & data (gate for everything)
- [ ] `pip install -r requirements.txt && pip install -e .`
- [ ] Download SAM ViT-B → `checkpoints/sam_vit_b_01ec64.pth`; MedSAM → `checkpoints/medsam_vit_b.pth`.
- [ ] Obtain BraTS 2021 (Synapse) into `data/raw/BraTS2021/<case>/`.
- [ ] Preprocess:
  ```bash
  python -m samri.data.preprocess --raw data/raw/BraTS2021 \
    --out data/processed/brats21 --layout brats21 --workers 8
  ```
- [ ] (external set) preprocess a second glioma dataset with `--split-mode all-test`.
- [ ] `PYTHONPATH=src python tests/test_core.py && PYTHONPATH=src python tests/smoke_torch.py`
- **Exit:** splits.json fixed (seed 42), index/patients CSVs sane, smoke green.

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

| # | Question | Current lean | Revisit when |
|---|---|---|---|
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
