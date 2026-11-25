"""Failure analysis: worst cases by stratum, with an overlay gallery.

Strata (automatic, no manual picking):
  - empty ET            : patients with no enhancing tumour in GT (listed, no overlay possible)
  - small ET            : smallest ET volumes among ET-positive patients
  - low ET contrast     : lowest mean ET-vs-ring intensity contrast on T1ce
  - worst overall       : lowest mean Dice across WT/TC/ET

Overlays show the image, GT contour, prediction contour and the box prompt. Predictions are
recomputed for the selected cases only (box jitter 0, oracle prompt), so the gallery is cheap.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation

from samri.common import REGIONS, region_mask, seed_everything
from samri.config import load_config
from samri.data.dataset import make_volume
from samri.data.prompts import box_from_mask
from samri.metrics import dice, hd95
from samri.models.sam_wrapper import build_model, load_trainable, to_sam_input


@torch.no_grad()
def predict_region(model, vol, cfg, region, device, amp):
    d = cfg["data"]
    S, seqs, ctx = d["native_size"], d["sequences"], d.get("context", 0)
    scale = 1024.0 / S
    gt = region_mask(np.asarray(vol.seg), region)
    zs = np.where(gt.reshape(vol.D, -1).any(1))[0]
    pred = np.zeros_like(gt)
    for i in range(0, len(zs), 8):
        zb = zs[i:i + 8]
        x = np.stack([vol.channels(z, seqs, ctx) for z in zb])
        x = to_sam_input(torch.from_numpy(x).to(device))
        b = np.stack([box_from_mask(gt[z]) for z in zb]).astype(np.float32) * scale
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg = model(x, box=torch.from_numpy(b).to(device),
                       region=torch.full((len(zb),), REGIONS.index(region), dtype=torch.long, device=device))
        lg = F.interpolate(lg.float(), (S, S), mode="bilinear", align_corners=False)
        pred[zb] = (lg[:, 0] > 0).cpu().numpy()
    return pred, gt


def et_contrast(vol, seq="t1ce"):
    """Mean (ET - surrounding ring) / brain std on T1ce, averaged over ET slices."""
    seg = np.asarray(vol.seg)
    et = region_mask(seg, "ET")
    zs = np.where(et.reshape(vol.D, -1).any(1))[0]
    vals = []
    for z in zs:
        m = et[int(z)]
        img = vol.channels(int(z), [seq])[0].astype(np.float32) / 255.0
        ring = binary_dilation(m, iterations=3) & ~binary_dilation(m, iterations=1)
        brain = img > 0.02
        if m.sum() < 5 or ring.sum() < 10:
            continue
        vals.append((img[m].mean() - img[ring].mean()) / (img[brain].std() + 1e-6))
    return float(np.mean(vals)) if vals else np.nan


def select_cases(df, vols, k, seq):
    df = df[(df.prompt == "box") & (df.jitter == 0)]
    et = df[df.region == "ET"]
    empty = et[et.gt_voxels == 0].pid.unique()[:k]
    small = et[et.gt_voxels > 0].nsmallest(k, "gt_voxels").pid.tolist()
    contrast = {pid: et_contrast(vols[pid], seq) for pid in et[et.gt_voxels > 0].pid.unique()}
    low = sorted(contrast, key=lambda p: (np.isnan(contrast[p]), contrast[p]))[:k]
    worst = df.groupby("pid").dice.mean().nsmallest(k).index.tolist()
    return dict(empty_et=list(empty), small_et=small, low_contrast=low, worst_overall=worst), contrast


def _plot_case(ax, vol, pred, gt, region, title, seq, S):
    area = gt.reshape(gt.shape[0], -1).sum(1)
    z = int(np.argmax(area)) if area.any() else int(vol.D // 2)
    img = vol.channels(z, [seq])[0]
    ax.imshow(img, cmap="gray")
    if gt[z].any():
        ax.contour(gt[z], levels=[0.5], colors=["#39ff14"], linewidths=0.8)
        x0, y0, x1, y1 = box_from_mask(gt[z])
        ax.add_patch(__import__("matplotlib").patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                               fill=False, edgecolor="yellow", linewidth=1.0))
    if pred[z].any():
        ax.contour(pred[z], levels=[0.5], colors=["#ff3b30"], linewidths=0.8)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--figdir", default="paper/figures")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    model = build_model(cfg).to(device).eval()
    load_trainable(model, torch.load(a.ckpt, map_location="cpu", weights_only=False)["trainable"])

    from samri.data.stream import ensure_stream_setup
    _, splits = ensure_stream_setup(cfg)
    pids = splits[a.split]
    df = pd.read_csv(Path("runs") / a.run / f"results_{a.split}.csv")
    vols = {pid: make_volume(cfg, pid) for pid in pids}
    cases, contrast = select_cases(df, vols, a.k, "t1ce")

    rows = []
    for stratum, plist in cases.items():
        for pid in plist:
            d = df[(df.pid == pid) & (df.region == "ET")]
            rows.append(dict(stratum=stratum, pid=pid,
                             et_contrast=contrast.get(pid, np.nan),
                             et_voxels=int(d.gt_voxels.iloc[0]) if len(d) else 0,
                             dice=float(d.dice.iloc[0]) if len(d) else np.nan,
                             hd95=float(d.hd95.iloc[0]) if len(d) else np.nan))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "failure_cases.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strata = [("small_et", "small ET"), ("low_contrast", "low ET contrast"), ("worst_overall", "worst overall")]
    fig, axes = plt.subplots(len(strata), a.k, figsize=(2.4 * a.k, 2.6 * len(strata)))
    axes = np.atleast_2d(axes)
    for i, (key, label) in enumerate(strata):
        for j, pid in enumerate(cases[key][:a.k]):
            region = "ET" if key != "worst_overall" else "WT"
            pred, gt = predict_region(model, vols[pid], cfg, region, device, amp)
            d = dice(pred, gt); h = hd95(pred, gt, vols[pid].spacing)
            title = f"{pid}\n{label} · {region} · Dice {d:.3f} · HD95 {h:.1f}"
            _plot_case(axes[i, j], vols[pid], pred, gt, region, title, "t1ce", cfg["data"]["native_size"])
    fig.suptitle("Failure cases — MSD Task01 pilot (oracle box prompts; not BraTS 2021)", fontsize=9)
    fig.tight_layout()
    figdir = Path(a.figdir); figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "failure_gallery.pdf", bbox_inches="tight")
    fig.savefig(figdir / "failure_gallery.png", bbox_inches="tight", dpi=200)
    print(f"wrote {out / 'failure_cases.csv'} and {figdir / 'failure_gallery.png'}")


if __name__ == "__main__":
    main()
