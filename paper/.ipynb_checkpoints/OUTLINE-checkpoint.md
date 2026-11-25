# Paper outline (write alongside experiments)

**Working claim (fill in after E4/E5):** "<adapter> lets SAM use all four MRI sequences, improving Dice/HD95 over naive
RGB mapping by X at <=Y% labeled patients, with the same prompts, and holds on an external cohort."

1. **Introduction**: promptable foundation models for medical images; multi-sequence MRI does not fit 3-channel RGB; contribution (1 sentence) + 3 bullets.
2. **Related work**: MedSAM, SAM-Med2D/3D, SAMed (LoRA), MA-SAM (3D adapters), multi-sequence BraTS methods, nnU-Net. (Verify each on arXiv/Scholar; check for 2025-26 overlap with your exact claim.)
3. **Method**: SAM + LoRA + decoder; input adapters (identity / stem / fusion); 2.5D context; prompt simulation; loss.
4. **Experimental setup**: BraTS 2021 split (counts), external cohort, preprocessing, prompts, baselines, metrics, seeds, hardware (L40S), training budget.
5. **Results**: Table 1 main (Dice/HD95 x WT/TC/ET, mean +- std, p-values); Fig: low-data curve; Fig: prompt-jitter robustness; Table: external test.
6. **Ablations**: sequence subsets, stem vs fusion, +/- 2.5D, LoRA rank, prompt type.
7. **Discussion/limitations**: oracle prompts, single tumor type, 2D slices, no clinical validation.
8. **Reproducibility**: code, configs, split file, seeds; dataset licenses and citations (BraTS papers).
Journal checklist: ethics/data-use statement, dataset citations, code link, statistical tests, limitations section.
