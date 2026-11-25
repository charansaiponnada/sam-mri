"""Torch-free checks: preprocessing, split, metrics, prompts, config inheritance."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from make_synthetic import make  # noqa: E402
from samri.common import region_mask  # noqa: E402
from samri.config import load_config  # noqa: E402
from samri.data.prompts import box_from_mask, jitter_box, point_from_mask  # noqa: E402
from samri.metrics import dice, hd95  # noqa: E402
from samri.prompts_iter import corrective_clicks  # noqa: E402


def test_metrics():
    a = np.zeros((10, 10, 10), bool); a[2:6, 2:6, 2:6] = True
    assert dice(a, a) == 1.0 and hd95(a, a) == 0.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0
    assert dice(np.zeros_like(a), a) == 0.0 and hd95(np.zeros_like(a), a) == 373.13
    b = np.roll(a, 2, axis=0)
    assert 0 < dice(a, b) < 1 and hd95(a, b) > 0
    assert abs(hd95(a, b, spacing=(2, 2, 2)) - 2 * hd95(a, b)) < 1e-6   # spacing is respected
    print("metrics ok")


def test_prompts():
    m = np.zeros((64, 64), bool); m[10:30, 20:50] = True
    assert list(box_from_mask(m)) == [20, 10, 49, 29]
    rng = np.random.default_rng(0)
    for _ in range(100):
        b = jitter_box(box_from_mask(m), 5, rng, 64)
        assert b[2] > b[0] and b[3] > b[1] and b.min() >= 0 and b.max() <= 63
    x, y = point_from_mask(m)
    assert m[int(y), int(x)]
    print("prompts ok")


def test_metrics_edge_cases():
    z = np.zeros((8, 8, 8), bool)
    assert dice(z, z) == 1.0 and hd95(z, z) == 0.0
    one = z.copy(); one[4, 4, 4] = True
    assert dice(one, one) == 1.0 and hd95(one, one) == 0.0
    assert dice(one, z) == 0.0 and hd95(one, z) == 373.13
    a = z.copy(); a[1:3, 1:3, 1:3] = True
    b = z.copy(); b[3:5, 1:3, 1:3] = True
    assert abs(hd95(a, b) - 2.0) < 1e-6          # far faces are 2 voxels apart (95th pct)
    assert abs(hd95(a, b, spacing=(3, 1, 1)) - 6.0) < 1e-6
    print("metric edge cases ok")


def test_iterative_prompts():
    gt = np.zeros((40, 40), bool); gt[10:30, 10:30] = True
    # missed tumour -> positive click inside the FN component
    c = corrective_clicks(np.zeros_like(gt), gt, 3)
    assert len(c) == 1 and c[0][2] == 1 and gt[c[0][1], c[0][0]]
    # over-segmentation -> negative click inside the FP ring
    pred = np.zeros_like(gt); pred[5:35, 5:35] = True
    c = corrective_clicks(pred, gt, 1)
    assert c and c[0][2] == 0 and pred[c[0][1], c[0][0]] and not gt[c[0][1], c[0][0]]
    # both error types: a 2-click budget covers one of each
    pred[15:20, 15:20] = False
    c = corrective_clicks(pred, gt, 2)
    assert len(c) == 2 and {lab for _, _, lab in c} == {0, 1}
    # no error -> no clicks; budget capped by the number of components
    assert corrective_clicks(gt, gt, 5) == []
    assert len(corrective_clicks(np.zeros_like(gt), gt, 5)) == 1
    print("iterative prompts ok")


def test_config():
    cfg = load_config(ROOT / "configs/e5b_fusion.yaml", ["train.epochs=3", "name=foo"])
    assert cfg["train"]["epochs"] == 3 and cfg["train"]["batch_size"] == 4 and cfg["name"] == "foo"
    assert cfg["data"]["sequences"] == ["t1", "t1ce", "t2", "flair"] and cfg["data"]["native_size"] == 256
    for p in (ROOT / "configs").glob("*.yaml"):
        load_config(p)
    print("config ok")


def test_preprocess():
    with tempfile.TemporaryDirectory() as t:
        raw, out = Path(t) / "raw", Path(t) / "out"
        make(raw)
        subprocess.run([sys.executable, "-m", "samri.data.preprocess", "--raw", str(raw), "--out", str(out),
                        "--workers", "1"], check=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        sp = json.loads((out / "splits.json").read_text())
        assert len(sp["train"]) == 4 and len(sp["val"]) == 1 and len(sp["test"]) == 1
        assert not (set(sp["train"]) & set(sp["test"])) and not (set(sp["train"]) & set(sp["val"]))
        pid = sp["train"][0]
        img, seg = np.load(out / pid / "img.npy"), np.load(out / pid / "seg.npy")
        meta = json.loads((out / pid / "meta.json").read_text())
        assert img.shape[0] == 4 and img.shape[2:] == (256, 256) and seg.shape == img.shape[1:]
        assert img.dtype == np.uint8 and set(np.unique(seg)) <= {0, 1, 2, 4}
        assert (img[:, :, 0, 0] == 0).all()                       # padding is background
        assert abs(meta["spacing"][0] - 1.5) < 1e-6               # z spacing preserved
        idx = pd.read_csv(out / "index.csv")
        assert set(idx.region) == {"WT", "TC", "ET"} and (idx.area > 0).all()
        wt = idx[(idx.pid == pid) & (idx.region == "WT")]
        assert wt.area.sum() == (seg > 0).sum()                    # index matches stored seg
        assert (region_mask(seg, "ET").sum((1, 2)) > 0).sum() == len(idx[(idx.pid == pid) & (idx.region == "ET")])
        # tumour extent +- margin only
        assert meta["n_slices"] < meta["orig_depth"]
    print("preprocess ok")


if __name__ == "__main__":
    test_metrics(); test_metrics_edge_cases(); test_prompts(); test_iterative_prompts()
    test_config(); test_preprocess()
