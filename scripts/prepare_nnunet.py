"""Export the SAME patient split to nnU-Net v2 format (region-based BraTS setup).

  python scripts/prepare_nnunet.py --raw data/raw/BraTS2021 --splits data/brats21_meta/splits.json \
      --nnunet-raw $nnUNet_raw --dataset-id 501 --layout brats21

Then:
  nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
  cp <out>/splits_final.json $nnUNet_preprocessed/Dataset501_BraTSsamri/splits_final.json   # train/val = our split
  nnUNetv2_train 501 3d_fullres 0
  nnUNetv2_predict -i $nnUNet_raw/Dataset501_BraTSsamri/imagesTs -o runs/nnunet_preds -d 501 -c 3d_fullres -f 0
  python scripts/eval_nnunet.py --pred-dir runs/nnunet_preds --raw data/raw/BraTS2021 --splits ... --layout brats21
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from samri.data.preprocess import LAYOUTS, find_file  # noqa: E402

SEQ_ORDER = ["t1", "t1ce", "t2", "flair"]


def remap_to_nnunet(seg, et_label):
    out = np.zeros_like(seg, dtype=np.uint8)
    out[seg == 2] = 1            # edema
    out[seg == 1] = 2            # necrotic / non-enhancing core
    out[seg == et_label] = 3     # enhancing tumor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--nnunet-raw", required=True)
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--layout", default="brats21", choices=list(LAYOUTS))
    a = ap.parse_args()
    L, splits = LAYOUTS[a.layout], json.loads(Path(a.splits).read_text())
    # Case ids must be flat for nnU-Net (nested Synapse layouts produce "sub/case").
    cid = lambda p: p.replace("/", "_").replace("\\", "_")
    ds = Path(a.nnunet_raw) / f"Dataset{a.dataset_id:03d}_BraTSsamri"
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    nn_splits = {}
    for part, imgs, labs in (("train", "imagesTr", "labelsTr"), ("val", "imagesTr", "labelsTr"), ("test", "imagesTs", "labelsTs")):
        nn_splits[part] = [cid(p) for p in splits[part]]
        for pid in splits[part]:
            pdir = Path(a.raw) / pid
            for i, s in enumerate(SEQ_ORDER):
                shutil.copy(find_file(pdir, L["patterns"][s]), ds / imgs / f"{cid(pid)}_{i:04d}.nii.gz")
            seg = nib.load(str(find_file(pdir, L["patterns"]["seg"])))
            data = np.rint(seg.get_fdata()).astype(np.uint8)
            hdr = seg.header.copy()
            hdr.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(remap_to_nnunet(data, L["et_label"]), seg.affine, hdr), ds / labs / f"{cid(pid)}.nii.gz")
    (ds / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"},
        "labels": {"background": 0, "whole tumor": [1, 2, 3], "tumor core": [2, 3], "enhancing tumor": [3]},
        "regions_class_order": [1, 2, 3],
        "numTraining": len(splits["train"]) + len(splits["val"]),
        "file_ending": ".nii.gz"}, indent=1))
    (ds / "splits_final.json").write_text(json.dumps(
        [{"train": nn_splits["train"], "val": nn_splits["val"]}], indent=1))
    print("wrote", ds, "| copy splits_final.json into nnUNet_preprocessed after plan_and_preprocess")


if __name__ == "__main__":
    main()
