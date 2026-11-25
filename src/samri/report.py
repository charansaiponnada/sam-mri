"""samri.report summary runs/*/results_test.csv      -> mean +- std table
   samri.report compare A.csv B.csv                 -> paired Wilcoxon of B vs A per region/prompt/jitter"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["region", "prompt", "jitter"]


def summary(paths):
    df = pd.concat([pd.read_csv(p) for p in paths])
    # average over seeds first (per patient), then across patients
    per_pat = df.groupby(["run", "pid"] + KEYS)[["dice", "hd95"]].mean().reset_index()
    print("| run | region | prompt | jitter | n | Dice | HD95 (mm) |\n|---|---|---|---|---|---|---|")
    for (run, *k), g in per_pat.groupby(["run"] + KEYS):
        print(f"| {run} | {k[0]} | {k[1]} | {k[2]} | {len(g)} | {g.dice.mean():.4f} ± {g.dice.std():.4f} "
              f"| {g.hd95.mean():.2f} ± {g.hd95.std():.2f} |")


def compare(a, b):
    A, B = pd.read_csv(a), pd.read_csv(b)
    cols = ["pid"] + KEYS
    A = A.groupby(cols)[["dice", "hd95"]].mean().reset_index()
    B = B.groupby(cols)[["dice", "hd95"]].mean().reset_index()
    m = A.merge(B, on=cols, suffixes=("_a", "_b"))
    print(f"A={a}\nB={b}\n| region | prompt | jitter | n | dDice (B-A) | p | dHD95 (B-A) | p |\n|---|---|---|---|---|---|---|---|")
    for k, g in m.groupby(KEYS):
        out = []
        for met in ("dice", "hd95"):
            d = (g[f"{met}_b"] - g[f"{met}_a"]).values
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            out += [f"{d.mean():+.4f}", f"{p:.3g}"]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {len(g)} | " + " | ".join(out) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("paths", nargs="+")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    args = ap.parse_args()
    summary(args.paths) if args.cmd == "summary" else compare(args.a, args.b)
