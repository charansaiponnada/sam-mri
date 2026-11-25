# Notes on the book-chapter experiments ("Adapting Promptable Foundation Models to Brain MRI: SAM-Style Segmentation on BraTS")

## Prediction dimensionality: which 3D variant was chosen, and why
- 2D: SAM ViT-B encodes one slice; mask is per-slice (existing behaviour).
- 2.5D: `data.context: 1` feeds slices z-1, z, z+1 as extra channels (12 input channels with the
  4-sequence stem); still a per-slice 2D head.
- **3D (chosen): T3a loss-only volumetric objective** (`configs/chapter/3d.yaml`, `train.vol_loss`).
  Contiguous z-blocks of 3 slices (centre slice valid, neighbours inherit the centre prompt with
  `valid=False`) are forward-passed to the 2D backbone; the loss is per-slice Dice+BCE on valid
  slices plus a block-level 3D Dice and a z-continuity (total variation) term. No new parameters,
  no encoder change, no extra inference machinery; evaluated by stacking per-slice predictions into
  a 3D mask (Dice / HD95 with voxel spacing).

## Prompt design
- Initial predictions: single click, box, box with jitter {0,5,10,20} px (oracle-simulated from GT).
- Iterative: starting from the jitter-0 box prediction, one corrective click per step placed on the
  largest error component (FN→positive, FP→negative), emitting rows for 1/3/5 clicks
  (`samri.eval` `eval.refine`, `samri.prompts_iter`).

## Dataset label
- Every run used the MSD Task01 brain-tumour pilot (public, CC-BY-SA 4.0) via
  `configs/chapter/data_msd.yaml`; switching to BraTS 2021 is a one-line `_base_` change in
  `configs/chapter/base.yaml` (`data_msd.yaml` -> `data_brats21.yaml`). Results in this chapter are
  therefore "MSD Task01 pilot, not BraTS 2021".

## Current status / numbers
- Real-data pilot (1 epoch x 41 patients, e5a/LoRA) on MSD Task01: val Dice 0.811 -> 0.829 (epoch 0/1);
  held-out 97-patient 3D Dice: WT 0.887 / TC 0.816 / ET 0.784.
- Chapter numbers (25% train budget, 2 epochs, seeds 1/2/3, LoRA) are pending the chapter runs
  (`bash scripts/run_chapter_experiments.sh`, STAGE=train then STAGE=prompt/STAGE=report).

## Limitations (expected in the final text)
- Prompts are oracle/GT-simulated; never labelled automatic segmentation.
- Only 25% of the (MSD pilot) training set and 2 epochs per arm — absolute Dice values are
  lower bounds, comparisons (same protocol, same schedule) are what the chapter claims.
- MedSAM / nnU-Net reference rows not run yet in this session (MedSAM checkpoint absent; nnU-Net
  predictions absent) and therefore listed as "not run".
