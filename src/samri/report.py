"""Reporting for the book chapter.

  samri.report summary runs/*/results_test.csv      -> mean +- std table (legacy)
  samri.report compare A.csv B.csv                  -> paired Wilcoxon (legacy)
  samri.report tables  [--runs 'runs/ch_*']         -> results/{main,prompts,dim}_table.{csv,md} + paper/tables/*.tex
  samri.report figures [--runs 'runs/ch_*']         -> paper/figures/*.pdf|png
  samri.report all                                  -> tables + figures

Every table cell is read from a results CSV produced by a real run; missing arms are reported as
"not run". The dataset label (MSD Task01 pilot vs BraTS 2021) comes from run_meta.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["region", "prompt", "jitter"]
REGIONS = ["WT", "TC", "ET"]

# (run prefix, display label) in chapter order.
METHODS = [
    ("ch_zero_shot", "Zero-shot SAM"),
    ("ch_decoder", "Decoder-only fine-tuning"),
    ("ch_lora", "LoRA"),
    ("ch_bottleneck", "Bottleneck adapters"),
    ("ch_fullft", "Full fine-tuning"),
    ("ch_25d", "2.5D (LoRA)"),
    ("ch_3d", "3D T3a (LoRA)"),
]
PROMPT_SPECS = [
    ("click", "point", 0, 0),
    ("box", "box", 0, 0),
    ("box jitter 5", "box", 5, 0),
    ("box jitter 10", "box", 10, 0),
    ("box jitter 20", "box", 20, 0),
    ("box + 1 click", "box+refine", 0, 1),
    ("box + 3 clicks", "box+refine", 0, 3),
    ("box + 5 clicks", "box+refine", 0, 5),
]


def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (len(p) - i) * p[k])
        out[k] = min(1.0, running)
    return out


def _load_runs(pattern="runs/ch_*"):
    runs = {}
    for d in sorted(Path().glob(pattern)):
        f = d / "results_test.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "n_clicks" not in df:
            df["n_clicks"] = 0
        meta = {}
        if (d / "run_meta.json").exists():
            meta = json.loads((d / "run_meta.json").read_text())
        runs[d.name] = (df, meta)
    return runs


def _arm(runs, prefix):
    return {k: v for k, v in runs.items() if k == prefix or k.startswith(prefix + "_s")}


def _dataset_label(runs):
    raws = {m.get("dataset") for _, m in runs.values() if m.get("dataset")}
    if any("MSD" in r for r in raws):
        return "MSD Task01 pilot, not BraTS 2021"
    return "BraTS 2021"


def _per_patient(df, prompt="box", jitter=0, n_clicks=0):
    s = df[(df.prompt == prompt) & (df.jitter == jitter) & (df.n_clicks.fillna(0) == n_clicks)]
    return s.groupby(["pid", "region"])[["dice", "hd95"]].mean().reset_index()


def _pvals_vs(base, other, region):
    a = base[base.region == region].set_index("pid").dice
    b = other[other.region == region].set_index("pid").dice
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return np.nan
    d = b.loc[common].values - a.loc[common].values
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def main_table(runs, out="results", tex="paper/tables"):
    zs = _arm(runs, "ch_zero_shot")
    zs_df = pd.concat([d for d, _ in zs.values()]) if zs else None
    zs_per = _per_patient(zs_df) if zs_df is not None else None
    rows = []
    for prefix, label in METHODS:
        arm = _arm(runs, prefix)
        if not arm:
            rows.append(dict(method=label, params_M=np.nan, train_h=np.nan, n_seeds=0,
                             **{f"{m}_{r}": np.nan for r in REGIONS for m in ("dice", "dice_std", "hd95", "p")}))
            continue
        df = pd.concat([d for d, _ in arm.values()])
        per = _per_patient(df)
        params = np.mean([m.get("params_trainable", np.nan) for _, m in arm.values()]) / 1e6
        secs = np.mean([m.get("wall_sec_total", np.nan) for _, m in arm.values()])
        row = dict(method=label, params_M=params, train_h=secs / 3600, n_seeds=len(arm))
        pv = {}
        for r in REGIONS:
            g = per[per.region == r]
            row[f"dice_{r}"], row[f"dice_std_{r}"] = g.dice.mean(), g.dice.std()
            row[f"hd95_{r}"] = g.hd95.mean()
            if zs_per is not None and prefix != "ch_zero_shot":
                pv[r] = _pvals_vs(zs_per, per, r)
            else:
                pv[r] = np.nan
        if any(np.isfinite(list(pv.values()))):
            adj = _holm([pv[r] if np.isfinite(pv[r]) else 1.0 for r in REGIONS])
            for r, p in zip(REGIONS, adj):
                row[f"p_{r}"] = p
        else:
            for r in REGIONS:
                row[f"p_{r}"] = np.nan
        rows.append(row)
    # reference rows: only if predictions actually exist
    for prefix, label in [("ch_medsam", "MedSAM zero-shot"), ("nnunet", "nnU-Net (automatic)")]:
        arm = _arm(runs, prefix)
        if not arm:
            rows.append(dict(method=label, params_M=np.nan, train_h=np.nan, n_seeds=0,
                             **{f"{m}_{r}": np.nan for r in REGIONS for m in ("dice", "dice_std", "hd95", "p")}))
            continue
        df = pd.concat([d for d, _ in arm.values()])
        per = _per_patient(df, prompt="box" if prefix == "ch_medsam" else "none")
        row = dict(method=label, params_M=np.nan, train_h=np.nan, n_seeds=len(arm))
        for r in REGIONS:
            g = per[per.region == r]
            row[f"dice_{r}"], row[f"dice_std_{r}"] = g.dice.mean(), g.dice.std()
            row[f"hd95_{r}"] = g.hd95.mean()
            row[f"p_{r}"] = _pvals_vs(zs_per, per, r) if zs_per is not None else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    _write(df, out, tex, "main_table", "Main results", _dataset_label(runs))
    return df


def prompts_table(runs, out="results", tex="paper/tables"):
    arm = _arm(runs, "ch_lora_prompts") or _arm(runs, "ch_lora")
    if not arm:
        print("prompts_table: no ch_lora_prompts / ch_lora runs")
        return pd.DataFrame()
    df = pd.concat([d for d, _ in arm.values()])
    rows = []
    for label, prompt, jitter, n in PROMPT_SPECS:
        per = _per_patient(df, prompt, jitter, n)
        if per.empty:
            continue
        row = dict(prompt=label)
        for r in REGIONS:
            g = per[per.region == r]
            row[f"dice_{r}"] = g.dice.mean()
            row[f"dice_std_{r}"] = g.dice.std()
            row[f"hd95_{r}"] = g.hd95.mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    _write(df, out, tex, "prompts_table", "Prompt design", _dataset_label(runs))
    return df


def dim_table(runs, out="results", tex="paper/tables"):
    arms = [("ch_lora", "2D"), ("ch_25d", "2.5D"), ("ch_3d", "3D (T3a)")]
    rows = []
    for prefix, label in arms:
        arm = _arm(runs, prefix)
        if not arm:
            rows.append(dict(dimension=label, **{f"{m}_{r}": np.nan for r in REGIONS for m in ("dice", "hd95")}))
            continue
        df = pd.concat([d for d, _ in arm.values()])
        per = _per_patient(df)
        row = dict(dimension=label)
        for r in REGIONS:
            g = per[per.region == r]
            row[f"dice_{r}"], row[f"hd95_{r}"] = g.dice.mean(), g.hd95.mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    _write(df, out, tex, "dim_table", "Prediction dimensionality", _dataset_label(runs))
    return df


def _fmt(v, nd=4):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _write(df, out, tex, name, title, dataset):
    out, tex = Path(out), Path(tex)
    out.mkdir(parents=True, exist_ok=True)
    tex.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    cols = list(df.columns)
    md = [f"**{title}** — {dataset}", "", "| " + " | ".join(cols) + " |",
          "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(_fmt(r[c]) if isinstance(r[c], (int, float, np.floating)) else str(r[c])
                                    for c in cols) + " |")
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    # booktabs
    texcols = "l" + "c" * (len(cols) - 1)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{title}. {dataset}.}}",
             rf"\label{{tab:{name}}}", rf"\begin{{tabular}}{{{texcols}}}", r"\toprule",
             " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(_fmt(v) if isinstance(v, (int, float, np.floating)) else str(v).replace("_", r"\_"))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tex / f"{name}.tex").write_text("\n".join(lines) + "\n")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


def figures(runs, out="paper/figures"):
    plt = _style()
    colors = {"WT": "#1b6ca8", "TC": "#e08b00", "ET": "#b2182b"}

    # 1) Dice vs number of prompts (LoRA prompt study)
    arm = _arm(runs, "ch_lora_prompts") or _arm(runs, "ch_lora")
    if arm:
        df = pd.concat([d for d, _ in arm.values()])
        xs, labels = [], []
        for label, prompt, jitter, n in PROMPT_SPECS:
            per = _per_patient(df, prompt, jitter, n)
            if not per.empty:
                xs.append((label, per))
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        x = np.arange(len(xs))
        for r in REGIONS:
            ys = [p[p.region == r].dice.mean() for _, p in xs]
            ax.plot(x, ys, "o-", color=colors[r], label=r)
        ax.set_xticks(x); ax.set_xticklabels([l for l, _ in xs], rotation=30, ha="right")
        ax.set_ylabel("3D Dice"); ax.set_ylim(0.5, 1.0); ax.legend(frameon=False)
        ax.set_title("Prompt design (LoRA, oracle prompts)")
        _save(fig, out, "dice_vs_prompts")

        # 2) Dice vs jitter
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        js = [0, 5, 10, 20]
        for r in REGIONS:
            ys = [_per_patient(df, "box", j).query("region == @r").dice.mean() for j in js]
            ax.plot(js, ys, "o-", color=colors[r], label=r)
        ax.set_xlabel("box jitter (px, 1024 frame)"); ax.set_ylabel("3D Dice")
        ax.set_xticks(js); ax.legend(frameon=False); ax.set_title("Prompt jitter")
        _save(fig, out, "dice_vs_jitter")

    # 3) Dice vs trainable parameters
    rows = []
    for prefix, label in METHODS:
        arm = _arm(runs, prefix)
        if not arm:
            continue
        df = pd.concat([d for d, _ in arm.values()])
        per = _per_patient(df)
        params = np.mean([m.get("params_trainable", np.nan) for _, m in arm.values()]) / 1e6
        for r in REGIONS:
            rows.append(dict(method=label, params_M=params, region=r, dice=per[per.region == r].dice.mean()))
    if rows:
        d = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for r in REGIONS:
            g = d[d.region == r]
            ax.scatter(g.params_M, g.dice, color=colors[r], label=r, s=28)
            for _, row in g.iterrows():
                ax.annotate(row.method, (row.params_M, row.dice), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("trainable parameters (M, log scale)")
        ax.set_ylabel("3D Dice"); ax.legend(frameon=False)
        ax.set_title("Adaptation strategy vs capacity")
        _save(fig, out, "dice_vs_params")


def cmd_all(runs_glob, out, tex, figdir):
    runs = _load_runs(runs_glob)
    if not runs:
        raise SystemExit(f"no runs matched {runs_glob!r} (need results_test.csv)")
    main_table(runs, out, tex)
    prompts_table(runs, out, tex)
    dim_table(runs, out, tex)
    figures(runs, figdir)
    print(f"wrote tables -> {out}, {tex}; figures -> {figdir}")


def summary(paths):
    df = pd.concat([pd.read_csv(p) for p in paths])
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
    for name in ("tables", "figures", "all"):
        p = sub.add_parser(name)
        p.add_argument("--runs", default="runs/ch_*")
        p.add_argument("--out", default="results")
        p.add_argument("--tex", default="paper/tables")
        p.add_argument("--figdir", default="paper/figures")
    args = ap.parse_args()
    if args.cmd == "summary":
        summary(args.paths)
    elif args.cmd == "compare":
        compare(args.a, args.b)
    else:
        runs = _load_runs(args.runs)
        if not runs:
            raise SystemExit(f"no runs matched {args.runs!r}")
        if args.cmd in ("tables", "all"):
            main_table(runs, args.out, args.tex)
            prompts_table(runs, args.out, args.tex)
            dim_table(runs, args.out, args.tex)
        if args.cmd in ("figures", "all"):
            figures(runs, args.figdir)
        print("done")
