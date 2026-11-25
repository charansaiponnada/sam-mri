"""Volume-level evaluation with GT-derived prompts (box or click), per patient and region.

Protocol (state it in the paper): for every axial slice where the GT region is non-empty, one prompt is
simulated from the GT (tight box + jitter, or the deepest interior click). Slices with empty GT get no
prediction. Predictions are stacked to 3D and scored with Dice / HD95 (mm, using voxel spacing).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from samri.common import REGIONS, region_mask, seed_everything
from samri.config import load_config
from samri.data.dataset import make_volume
from samri.data.prompts import box_from_mask, jitter_box, make_seed, point_from_mask
from samri.metrics import dice, hd95
from samri.models.sam_wrapper import build_model, load_trainable, to_sam_input


@torch.no_grad()
def evaluate_patient(model, vol, cfg, device, amp, prompts, jitters, bs, seed):
    d = cfg["data"]
    S, seqs, ctx, regions = d["native_size"], d["sequences"], d.get("context", 0), d["regions"]
    scale = 1024.0 / S
    seg = np.asarray(vol.seg)
    gts = {r: region_mask(seg, r) for r in regions}
    zs_by = {r: np.where(gts[r].reshape(vol.D, -1).any(1))[0] for r in regions}
    union = sorted(set().union(*[set(z.tolist()) for z in zs_by.values()]))
    pos = {z: i for i, z in enumerate(union)}
    embs = None
    if union:  # encode each needed slice once, reuse for every region / prompt / jitter
        chunks = []
        for i in range(0, len(union), bs):
            x = np.stack([vol.channels(z, seqs, ctx) for z in union[i:i + bs]])
            x = to_sam_input(torch.from_numpy(x).to(device))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                chunks.append(model.encode(x))
        embs = torch.cat(chunks)
    rows = []
    for r in regions:
        gt, zs = gts[r], zs_by[r]
        for prompt in prompts:
            for j in (jitters if prompt == "box" else [0]):
                pred = np.zeros_like(gt)
                for i in range(0, len(zs), bs):
                    zb = zs[i:i + bs]
                    ms = [gt[z] for z in zb]
                    idx = torch.tensor([pos[int(z)] for z in zb], device=device)
                    rngs = [np.random.default_rng(make_seed(vol.pid, int(z), r, j, seed)) for z in zb]
                    if prompt == "box":
                        b = np.stack([jitter_box(box_from_mask(m), j * S / 1024, g, S) for m, g in zip(ms, rngs)])
                        kw = dict(box=torch.from_numpy(b).to(device) * scale)
                    else:
                        p = np.stack([point_from_mask(m) for m in ms])
                        kw = dict(point=torch.from_numpy(p).to(device) * scale)
                    kw["region"] = torch.full((len(zb),), REGIONS.index(r), dtype=torch.long, device=device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg = model.decode(embs[idx], **kw)
                    lg = F.interpolate(lg.float(), (S, S), mode="bilinear", align_corners=False)
                    pred[zb] = (lg[:, 0] > 0).cpu().numpy()
                rows.append(dict(pid=vol.pid, region=r, prompt=prompt, jitter=j,
                                 dice=dice(pred, gt), hd95=hd95(pred, gt, vol.spacing),
                                 gt_voxels=int(gt.sum()), pred_voxels=int(pred.sum()), n_slices=len(zs)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="trained checkpoint; omit for zero-shot SAM / MedSAM")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-patients", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    model = build_model(cfg).to(device).eval()
    if a.ckpt:
        load_trainable(model, torch.load(a.ckpt, map_location="cpu", weights_only=False)["trainable"])
    if cfg["data"].get("mode", "processed") == "stream":
        from samri.data.stream import ensure_stream_setup

        _, splits = ensure_stream_setup(cfg)
    else:
        sp = Path(cfg["data"]["splits"])
        if not sp.exists():
            sp = Path(cfg["data"]["root"]) / "splits.json"
        splits = json.loads(sp.read_text())
    pids = splits[a.split][: a.max_patients or None]
    e = cfg["eval"]
    rows = []
    for pid in tqdm(pids):
        vol = make_volume(cfg, pid)
        for r in evaluate_patient(model, vol, cfg, device, amp, e["prompt"], e["jitter"], e["batch_size"], cfg["seed"]):
            r.update(run=cfg["name"], split=a.split, seed=cfg["seed"])
            rows.append(r)
    out = Path(a.out) if a.out else Path("runs") / cfg["name"] / f"results_{a.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby(["region", "prompt", "jitter"])[["dice", "hd95"]].mean().round(4))
    print("saved", out)


if __name__ == "__main__":
    main()
