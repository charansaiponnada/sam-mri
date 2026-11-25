"""End-to-end smoke test with a tiny random SAM on synthetic data (CPU is fine, ~minutes).
Run this FIRST on the GPU machine:   PYTHONPATH=src python tests/smoke_torch.py
It checks the batching/adapter/LoRA logic and runs train -> eval -> report once."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
os.chdir(ROOT)
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

from make_synthetic import make  # noqa: E402
from samri.config import load_config  # noqa: E402
from samri.models.sam_wrapper import FusionHead, add_lora, build_model, build_sam, load_trainable, trainable_state_dict  # noqa: E402


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True, env=ENV)


def unit_checks():
    torch.manual_seed(0)
    cfg = load_config("configs/smoke.yaml")
    model = build_model(cfg).eval()
    # 1) batched one-to-one decoding == per-sample decoding
    x = torch.rand(3, 4, 1024, 1024)
    with torch.no_grad():
        emb = model.encode(x)
        box = torch.tensor([[100., 100, 400, 500], [50, 60, 300, 300], [600, 200, 900, 800]])
        batched = model.decode(emb, box=box)
        single = torch.cat([model.decode(emb[i:i + 1], box=box[i:i + 1]) for i in range(3)])
        assert batched.shape == (3, 1, 256, 256), batched.shape
        assert torch.allclose(batched, single, atol=1e-4), (batched - single).abs().max()
        pt = torch.tensor([[200., 300], [100, 100], [700, 500]])
        assert torch.allclose(model.decode(emb, point=pt),
                              torch.cat([model.decode(emb[i:i + 1], point=pt[i:i + 1]) for i in range(3)]), atol=1e-4)
    # 2) stem starts as the naive [t1ce, t2, flair] mapping
    xs = torch.rand(1, 4, 8, 8)
    assert torch.allclose(model.adapter(xs), xs[:, 1:4], atol=1e-6)
    # 3) fusion head starts as the mean of per-sequence embeddings
    fh = FusionHead(4)
    e = torch.rand(2, 4, 256, 4, 4)
    assert torch.allclose(fh(e), e.mean(1), atol=1e-5)
    # 4) LoRA is a no-op at init
    sam = build_sam("test").eval()
    xi = torch.rand(1, 3, 1024, 1024)
    with torch.no_grad():
        before = sam.image_encoder(xi)
        add_lora(sam.image_encoder, 4, 4)
        assert torch.allclose(before, sam.image_encoder(xi), atol=1e-5)
    # 5) trainable-state roundtrip; only adapter + LoRA + decoder are trainable, encoder base is frozen
    st = trainable_state_dict(model)
    assert st and not any("prompt_encoder" in k or ".qkv.qkv." in k for k in st)
    load_trainable(model, st)
    # 6) fusion model forward (k=1 and k=3) and 2.5D stem
    for extra in ({"model": {"adapter": "fusion"}}, {"model": {"adapter": "fusion"}, "data": {"context": 1}},
                  {"model": {"adapter": "stem"}, "data": {"context": 1}}):
        c = load_config("configs/smoke.yaml")
        for k, v in extra.items():
            c[k].update(v)
        m = build_model(c).eval()
        ch = len(c["data"]["sequences"]) * (2 * c["data"].get("context", 0) + 1)
        with torch.no_grad():
            out = m(torch.rand(2, ch, 1024, 1024), box=box[:2])
        assert out.shape == (2, 1, 256, 256)
    print("unit checks ok")


def pipeline():
    # Use a throwaway dir so the committed repo never keeps synthetic data.
    tmp = Path(tempfile.mkdtemp(prefix="samri_smoke_"))
    raw, proc = tmp / "raw", tmp / "processed"
    for d in (ROOT / "runs/smoke", ROOT / "runs/smoke_vol"):
        shutil.rmtree(d, ignore_errors=True)
    make(raw)
    run("samri.data.preprocess", "--raw", str(raw), "--out", str(proc), "--workers", "1")
    base = [f"data.root={proc}", f"data.splits={proc / 'splits.json'}"]

    # T1/T2 path (single-slice 2D training + region conditioning)
    run("samri.train", "--config", "configs/smoke.yaml", "--set", *base)
    run("samri.eval", "--config", "configs/smoke.yaml", "--set", *base, "--ckpt", "runs/smoke/best.pt")
    df = pd.read_csv("runs/smoke/results_test.csv")
    n_test = len(json.loads((proc / "splits.json").read_text())["test"])
    assert len(df) == n_test * 3 * (2 + 1), len(df)      # patients x regions x (2 box jitters + 1 point)
    assert df.dice.between(0, 1).all() and (df.hd95 >= 0).all()

    # T3a path (contiguous z-blocks + 3D Dice + z-continuity)
    run("samri.train", "--config", "configs/smoke.yaml", "--set", *base,
        "name=smoke_vol", "train.vol_loss=true", "train.vol_block=3", "train.batch_size=1")
    run("samri.eval", "--config", "configs/smoke.yaml", "--set", *base,
        "name=smoke_vol", "--ckpt", "runs/smoke_vol/best.pt")
    dv = pd.read_csv("runs/smoke_vol/results_test.csv")
    assert len(dv) == len(df) and dv.dice.between(0, 1).all() and (dv.hd95 >= 0).all(), len(dv)

    run("samri.report", "summary", "runs/smoke/results_test.csv")
    shutil.rmtree(tmp, ignore_errors=True)
    print("pipeline ok")


if __name__ == "__main__":
    unit_checks()
    pipeline()
