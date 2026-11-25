import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from samri.common import seed_everything
from samri.config import load_config
from samri.data.dataset import SliceDataset
from samri.models.sam_wrapper import build_model, to_sam_input, trainable_state_dict


def select_patients(pids, frac, data_seed):
    """Nested random subsets (5% inside 10% inside 25% ...) for low-data curves."""
    if frac >= 1:
        return list(pids)
    order = np.random.default_rng(data_seed).permutation(len(pids))
    n = max(1, int(round(frac * len(pids))))
    return [pids[i] for i in sorted(order[:n])]


def seg_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return dice.mean() + F.binary_cross_entropy_with_logits(logits, target)


def prompts_from_batch(batch, kind, device, S):
    scale = 1024.0 / S
    if kind == "box":
        return dict(box=batch["box"].to(device) * scale)
    return dict(point=batch["point"].to(device) * scale)


def run_forward(model, batch, kind, device, S, amp):
    x = to_sam_input(batch["image"].to(device, non_blocking=True))
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        logits = model(x, **prompts_from_batch(batch, kind, device, S))
    logits = logits.float()
    if logits.shape[-1] != S:
        logits = F.interpolate(logits, (S, S), mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def validate(model, loader, device, S, amp):
    model.eval()
    ds = []
    for batch in loader:
        logits = run_forward(model, batch, "box", device, S, amp)
        pred = (logits > 0).float()
        gt = batch["mask"].to(device)[:, None]
        inter = (pred * gt).sum((1, 2, 3))
        ds.append((2 * inter / (pred.sum((1, 2, 3)) + gt.sum((1, 2, 3)) + 1e-6)).cpu())
    model.train()
    return float(torch.cat(ds).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5 seed=1 name=foo")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed_everything(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, S = cfg["train"], cfg["data"]["native_size"]
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    out = Path("runs") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    splits = json.loads(Path(cfg["data"]["splits"]).read_text())
    train_pids = select_patients(splits["train"], cfg["data"]["train_fraction"], cfg["data"]["data_seed"])
    print(f"train patients: {len(train_pids)} / {len(splits['train'])}  val: {len(splits['val'])}")
    ds_tr = SliceDataset(cfg, train_pids, train=True, seed=cfg["seed"])
    ds_va = SliceDataset(cfg, splits["val"], train=False, jitter=t["val_jitter"], max_items=t["val_items"], seed=0)
    g = torch.Generator().manual_seed(cfg["seed"])
    nw = t["num_workers"]
    dl_tr = DataLoader(ds_tr, t["batch_size"], shuffle=True, num_workers=nw, drop_last=True, generator=g,
                       pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, t["batch_size"], shuffle=False, num_workers=nw)
    print(f"train items: {len(ds_tr)}  val items: {len(ds_va)}")

    model = build_model(cfg).to(device)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise SystemExit("no trainable parameters (zero-shot config?) - use samri.eval directly")
    adapter = [p for n, p in named if n.startswith(("adapter", "fusion"))]
    rest = [p for n, p in named if not n.startswith(("adapter", "fusion"))]
    groups = [dict(params=rest, lr=t["lr"])] + ([dict(params=adapter, lr=t["adapter_lr"])] if adapter else [])
    opt = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
    total = t["epochs"] * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / t["warmup_steps"]) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    print(f"trainable params: {sum(p.numel() for _, p in named) / 1e6:.2f}M")

    rng = np.random.default_rng(cfg["seed"])
    best, step, log = -1.0, 0, []
    for epoch in range(t["epochs"]):
        model.train()
        t0, run = time.time(), []
        for batch in tqdm(dl_tr, desc=f"epoch {epoch}", leave=False):
            kind = t["prompt"] if t["prompt"] != "mixed" else ("box" if rng.random() < 0.5 else "point")
            logits = run_forward(model, batch, kind, device, S, amp)
            loss = seg_loss(logits, batch["mask"].to(device)[:, None])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in named], t["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            run.append(loss.item())
        vd = validate(model, dl_va, device, S, amp)
        row = dict(epoch=epoch, step=step, train_loss=float(np.mean(run)), val_dice2d=vd, sec=time.time() - t0)
        log.append(row)
        print(row)
        ck = dict(trainable=trainable_state_dict(model), cfg=cfg, epoch=epoch, val_dice2d=vd)
        torch.save(ck, out / "last.pt")
        if vd > best:
            best = vd
            torch.save(ck, out / "best.pt")
        (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"done. best val 2D Dice {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
