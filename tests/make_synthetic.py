"""Tiny fake BraTS21-style dataset (ellipsoid 'tumors') for pipeline smoke tests."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def make(root, n=6, shape=(60, 60, 24), seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    X, Y, Z = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    for i in range(n):
        pid = f"Synth_{i:04d}"
        d = root / pid
        d.mkdir(parents=True, exist_ok=True)
        c = np.array(shape) / 2 + rng.uniform(-6, 6, 3)
        r = rng.uniform(8, 13, 3)
        dist = ((X - c[0]) / r[0]) ** 2 + ((Y - c[1]) / r[1]) ** 2 + ((Z - c[2]) / (r[2] / 2.2)) ** 2
        seg = np.zeros(shape, np.float32)
        seg[dist < 1.0] = 2          # edema
        seg[dist < 0.5] = 1          # core
        seg[dist < 0.2] = 4          # enhancing
        brain = ((X - 30) ** 2 + (Y - 30) ** 2 + ((Z - 12) * 2) ** 2) < 26 ** 2
        for s in ("t1", "t1ce", "t2", "flair"):
            v = (rng.normal(100, 10, shape) + 60 * (seg > 0) + 40 * (seg == 4)).astype(np.float32) * brain
            nib.save(nib.Nifti1Image(v, np.diag([1, 1, 1.5, 1])), d / f"{pid}_{s}.nii.gz")
        nib.save(nib.Nifti1Image(seg, np.diag([1, 1, 1.5, 1])), d / f"{pid}_seg.nii.gz")


if __name__ == "__main__":
    make(sys.argv[1])
