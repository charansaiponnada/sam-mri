"""Convert MSD Task01 (BrainTumour) into the BraTS-style folder layout used by samri.

The Medical Segmentation Decathlon Task01 brain-tumour set is public (no login) and is derived
from BraTS. This mirror stores one 4D file per case::

    Task01_BrainTumour/imagesTr/BRATS_XXX.nii.gz   # (240,240,155,4): FLAIR,T1,T1ce,T2
    Task01_BrainTumour/labelsTr/BRATS_XXX.nii.gz   # 0,1=edema,2=non-enh,3=enhancing

(Older distributions use ``BRATS_XXX_000{0..3}.nii.gz``; both are handled.)

This script writes ``<out>/<case>/<case>_{t1,t1ce,t2,flair,seg}.nii.gz`` with the segmentation
remapped to the canonical BraTS labels {0, 1=NCR, 2=ED, 4=ET}.

    python scripts/prepare_msd.py --msd data/msd/Task01_BrainTumour --out data/raw/MSD --workers 8

Pilot use only: MSD overlaps with BraTS training years, so do not use it as an external test set
or mix it into the BraTS 2021 runs.
"""
import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

# MSD 4D channel index -> samri sequence name (dataset.json: FLAIR, T1w, t1gd, T2w)
MODALITY_INDEX = {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}
# older per-file distribution: suffix -> sequence
MODALITY_SUFFIX = {"flair": "_0000", "t1": "_0001", "t1ce": "_0002", "t2": "_0003"}
# MSD labels {0,1=edema,2=non-enhancing,3=enhancing} -> canonical BraTS {0,1=NCR,2=ED,4=ET}
LUT = np.array([0, 2, 1, 4], dtype=np.uint8)


def _one(task):
    images, labels, out, case = task
    odir = Path(out) / case
    odir.mkdir(parents=True, exist_ok=True)
    src4d = Path(images) / f"{case}.nii.gz"
    if src4d.exists():
        img = nib.load(str(src4d))
        if img.ndim != 4 or img.shape[-1] < 4:
            raise ValueError(f"{case}: expected a 4D image with >=4 channels, got {img.shape}")
        data = np.asarray(img.dataobj)                       # (X, Y, Z, 4)
        for seq, k in MODALITY_INDEX.items():
            hdr = img.header.copy()
            hdr.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(np.asarray(data[..., k], dtype=np.float32), img.affine, hdr),
                     str(odir / f"{case}_{seq}.nii.gz"))
    else:
        for seq, suffix in MODALITY_SUFFIX.items():
            src = Path(images) / f"{case}{suffix}.nii.gz"
            dst = odir / f"{case}_{seq}.nii.gz"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
    seg_src = Path(labels) / f"{case}.nii.gz"
    img = nib.load(str(seg_src))
    lab = np.rint(np.asarray(img.dataobj)).astype(np.int16)
    if lab.max() > 3:
        raise ValueError(f"{case}: unexpected MSD labels {np.unique(lab)}")
    seg = LUT[lab]
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(seg, img.affine, hdr), str(odir / f"{case}_seg.nii.gz"))
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msd", required=True, help="Task01_BrainTumour root")
    ap.add_argument("--out", required=True, help="output root in BraTS-style layout")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    msd = Path(a.msd)
    images, labels = msd / "imagesTr", msd / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise SystemExit(f"expected {images} and {labels} (extract Task01_BrainTumour.tar first)")
    files = [p for p in images.glob("*.nii.gz") if not p.name.startswith("._")]
    cases = sorted({re.sub(r"(_0000)?\.nii\.gz$", "", p.name) for p in files})
    if not cases:
        raise SystemExit(f"no cases found in {images}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = [(str(images), str(labels), a.out, c) for c in cases]
    with Pool(a.workers) as pool:
        for i, case in enumerate(pool.imap_unordered(_one, tasks), 1):
            if i % 50 == 0 or i == len(cases):
                print(f"{i}/{len(cases)} cases")
    print(f"wrote {len(cases)} cases to {a.out}")


if __name__ == "__main__":
    main()
