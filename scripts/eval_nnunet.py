"""Score nnU-Net predictions with the same Dice/HD95 code and CSV format as samri.eval (full native volumes)."""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json  # noqa: E402

from samri.common import region_mask  # noqa: E402
from samri.data.preprocess import LAYOUTS, find_file  # noqa: E402
from samri.metrics import dice, hd95  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--layout", default="brats21", choices=list(LAYOUTS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--run", default="nnunet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = LAYOUTS[a.layout]
    pids = json.loads(Path(a.splits).read_text())[a.split]
    rows = []
    for pid in pids:
        gt_img = nib.load(str(find_file(Path(a.raw) / pid, L["patterns"]["seg"])))
        gt = np.rint(gt_img.get_fdata()).astype(np.uint8)
        if L["et_label"] != 4:
            gt[gt == L["et_label"]] = 4
        pred = np.rint(nib.load(str(Path(a.pred_dir) / f"{pid}.nii.gz")).get_fdata()).astype(np.uint8)
        pm = {"WT": pred > 0, "TC": (pred == 2) | (pred == 3), "ET": pred == 3}  # nnU-Net label convention from prepare_nnunet.py
        sp = tuple(float(z) for z in gt_img.header.get_zooms()[:3])
        for r in ("WT", "TC", "ET"):
            g = region_mask(gt, r)
            rows.append(dict(pid=pid, region=r, prompt="none", jitter=0, dice=dice(pm[r], g), hd95=hd95(pm[r], g, sp),
                             gt_voxels=int(g.sum()), pred_voxels=int(pm[r].sum()), n_slices=-1,
                             run=a.run, split=a.split, seed=0))
    out = Path(a.out or f"runs/{a.run}/results_{a.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
