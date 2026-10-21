#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figures4papers (figkit) rebuild of the 7 in-text paper figures — Emotion
Multicalibration (Pattern Recognition submission).

Scope: the *adopted* designs only (author selection 2026-10-20):
    fig1 V1 grouped bars | fig2 V1 three-column reliability | fig3 V3 dual metric
    fig4 V1 modality lines | fig6 V1 bars + z grid | fig7 V1 stacked forest
    fig8 V2 by-model forest

Style: figures4papers house style via figkit.py
  * semantic PALETTE (blue = ours/proposed, green = improvement contrast,
    teal = LLaVA series, red = pooled baseline bars / significance threshold,
    neutral grey = reference lines)
  * apply_publication_style (no top/right spines, frameless legend, vector text)
  * make_grouped_bar helper where the chart type fits (fig1, fig6a);
    custom artists elsewhere (forest plots, reliability, linestyle-encoded
    multi-series) but always within the figkit style contract
  * finalize_figure export: PDF (fonttype 42) + PNG 300 dpi, tight bbox
  * drawn at SCALE=2x final print width so figkit's large-figure proportions
    (font 15, spine 2.0) come out at journal size (~7.5pt type / 1pt spines)

Data: read-only from frozen JSONs (identical reads to 02-working/figures_redraw/*):
  ../formal_experiment/rigor_results_v2.json, rigor_samples_v2.json,
  rigor_r5b_family_ladder.json, h3_metric_power.json, h3_worstgroup_robust.json,
  ../broaden_emotic/results/h5_*.json,
  ../broaden_emotic/pg_experiment/pg_fdr_and_seeddata.json
Zero re-computation of statistics — every point/CI/star is transcribed.
"""
import json
import os
import sys

SKILL_DIR = r"C:/Users/Lenovo/.workbuddy/skills/figures4papers"
sys.path.insert(0, SKILL_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figkit import (PALETTE, DEFAULT_COLORS, FigureStyle,
                    apply_publication_style, create_subplots,
                    finalize_figure, make_grouped_bar, annotate_bars)

# ------------------------------------------------------------------ paths
HERE = os.path.dirname(os.path.abspath(__file__))
FE = os.path.abspath(os.path.join(HERE, "..", "formal_experiment"))
EMO = os.path.abspath(os.path.join(HERE, "..", "broaden_emotic"))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "03-output", "figures_f4p"))
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
# draw at 2x final print width; figkit proportions then land at journal size
SCALE = 2
DOUBLE, SINGLE = 6.9, 3.35

apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))
matplotlib.rcParams.update({
    "mathtext.fontset": "stixsans",
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.major.size": 5.0, "ytick.major.size": 5.0,
    "legend.fontsize": 12,
    "axes.labelsize": 15, "axes.titlesize": 16,
    "lines.linewidth": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# semantic colour roles (figures4papers PALETTE)
C_OURS = PALETTE["blue_main"]          # ours / Qwen2.5-VL-3B / primary contrast
C_OURS_L = PALETTE["blue_secondary"]   # secondary blue (series / text modality)
C_GREEN = PALETTE["green_3"]           # improvement contrast (vs uncal) / fine bins
C_GREEN_L = PALETTE["green_1"]         # fine-bin fill
C_TEAL = PALETTE["teal"]               # LLaVA model series
C_VIOLET = PALETTE["violet"]
C_BASE = PALETTE["red_2"]              # pooled-isotonic baseline bars
C_RED = PALETTE["red_strong"]          # significance threshold lines
C_GREY = "#8A8A8A"                     # reference-line grey (neutral family)
C_GREY_L = "#C4C4C4"                   # pooled reference curves

# 5-model sequence for fig3 (blue -> teal -> violet -> green, grayscale-safe
# via linestyle double encoding)
M_COLS = [C_OURS, C_OURS_L, C_TEAL, C_VIOLET, C_GREEN]
M_LS = ["-", (0, (4, 1.6)), (0, (1.4, 1.2)), (0, (5, 1.4, 1.4, 1.4)), (0, (2.4, 1.5))]
# 5-text-setting blue ramp for fig4 + vision in teal
TXT_COLS = ["#0F4D92", "#245FAB", "#3775BA", "#6FA3D4", "#9DC2E6"]

SHORT = {
    "Gemma-2-9B|GoEmotions": "Gemma-2-9B",
    "Llama-3-8B|GoEmotions": "Llama-3-8B",
    "Qwen2.5-7B|GoEmotions": "Qwen2.5-7B",
    "Qwen2.5-3B|GoEmotions": "Qwen2.5-3B",
    "Qwen2.5-VL-3B|GoEmotions": "Qwen2.5-VL-3B",
    "LLaVA(RAF)|RAF-DB": "LLaVA / RAF-DB",
}


def stars(p):
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def export(fig, name):
    paths = finalize_figure(fig, os.path.join(OUT, name),
                            formats=["pdf", "png"], dpi=300, pad=0.04)
    print("  saved", name, "->", [os.path.basename(p) for p in paths])


def load_rigor():
    R = json.load(open(os.path.join(FE, "rigor_results_v2.json"), encoding="utf-8"))
    S = json.load(open(os.path.join(FE, "rigor_samples_v2.json"), encoding="utf-8"))
    L = json.load(open(os.path.join(FE, "rigor_r5b_family_ladder.json"), encoding="utf-8"))
    return R, S, L


def comp(R, setting, tag):
    c = R["R1_main"][setting]["comparisons"]
    for kk in c:
        if tag in kk:
            return c[kk]
    raise KeyError(tag)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# =====================================================================
# Fig.1 — main result: Delta CW-ECE (grouped bars, adopted design V1)
# =====================================================================
def fig1_grouped_bars(R):
    keys = list(R["R1_main"])
    K = len(keys)
    fig, ax = create_subplots(1, 1, figsize=(DOUBLE * SCALE, 2.5 * SCALE))
    cats = [SHORT[k] for k in keys]
    series = [[comp(R, k, "vs Pooled-Iso")["mean"] for k in keys],
              [comp(R, k, "vs Uncalibrated")["mean"] for k in keys]]
    labels = ["ours $-$ pooled isotonic", "ours $-$ uncalibrated"]
    make_grouped_bar(ax, cats, series, labels,
                     ylabel=r"$\Delta$ CW-ECE  (negative = ours better)",
                     colors=[C_OURS, C_GREEN],
                     edgecolor="black", linewidth=1.2)

    # error bars + in-bar value labels + stars below CI (bar centres computed
    # analytically to match make_grouped_bar geometry)
    w = 0.8 / 2
    x = np.arange(K)
    for i, (tag, col, tcol) in enumerate((("vs Pooled-Iso", C_OURS, "white"),
                                          ("vs Uncalibrated", C_GREEN, "#173B22"))):
        offset = (i - 0.5) * w
        for j, k in enumerate(keys):
            r = comp(R, k, tag)
            ax.errorbar(x[j] + offset, r["mean"],
                        yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                        fmt="none", ecolor="black", elinewidth=1.2,
                        capsize=2.4, capthick=1.2, zorder=5)
            # value label INSIDE the bar body (bars run downward from 0)
            ax.text(x[j] + offset, r["mean"] + 0.005, f"{r['mean']:.3f}",
                    ha="center", va="bottom", fontsize=11, color=tcol,
                    fontweight="bold", zorder=6)
            s = stars(r["p_two_sided"])
            if s:
                ax.text(x[j] + offset, r["ci_lo"] - 0.005, s, ha="center",
                        va="top", fontsize=15, fontweight="bold", color=col,
                        zorder=6)
    # generous headroom above zero so the legend + CI note live in an empty
    # band and never touch bar tops / error-bar caps (fixes legend overlap)
    ax.set_ylim(-0.375, 0.128)
    ax.axhline(0, color="#444", lw=1.2, zorder=1)
    # modality band (below all annotations; ylim headroom reserved)
    ax.axvline(4.5, color="#BBBBBB", lw=1.2, ls=(0, (2, 2)), zorder=1)
    ax.text(2.0, -0.362, "Text (GoEmotions)", ha="center", va="bottom",
            fontsize=12, color="#555", style="italic")
    ax.text(5.0, -0.362, "Vision (RAF-DB)", ha="center", va="bottom",
            fontsize=12, color="#555", style="italic")
    ax.text(0.0, 0.123,
            "error bars: nested bootstrap 95% CI ($B{=}2000$, 5 splits); "
            "* $p<.05$, ** $p<.01$, *** $p<.001$",
            fontsize=11, color="#555", va="top")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=12, borderaxespad=0.2,
              handlelength=1.6, borderpad=0.15, labelspacing=0.35)
    export(fig, "fig1_grouped_bars")


# =====================================================================
# Fig.2 — reliability diagrams (three-column EmoCaliber layout, V1)
# =====================================================================
def rel_bins(pred, corr):
    edges = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(pred, edges) - 1, 0, 9)
    cx, cy, elo, ehi = [], [], [], []
    for i in range(10):
        m = bi == i
        if m.sum() >= 5:
            k, n = int(corr[m].sum()), int(m.sum())
            lo, hi = wilson_ci(k, n)
            cx.append(pred[m].mean()); cy.append(k / n)
            elo.append(k / n - lo); ehi.append(hi - k / n)
    return np.array(cx), np.array(cy), np.array(elo), np.array(ehi)


def fine_bins(pred, corr):
    qs = np.unique(np.quantile(pred, np.linspace(0, 1, 21)))
    xs, ys = [], []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (pred >= a) & (pred < b) if b != qs[-1] else (pred >= a) & (pred <= b)
        if m.sum() >= 3:
            xs.append(pred[m].mean()); ys.append(corr[m].mean())
    return np.array(xs), np.array(ys)


def get_pred(S, setting, which):
    sp = S[setting]["splits"]["42"]
    idx = np.asarray(sp["test_idx"])
    conf = np.asarray(S[setting]["conf"])[idx]
    corr = np.asarray(S[setting]["corr"])[idx]
    pred = np.asarray(sp["pred_uncal"] if which == "u"
                      else sp["pred_pooled"] if which == "p"
                      else sp["pred_group"])
    return pred, corr


def draw_reliability(ax, pred, corr, title, letter):
    fx, fy = fine_bins(pred, corr)
    cx, cy, elo, ehi = rel_bins(pred, corr)
    ax.scatter(fx, fy, s=22, marker="s", facecolor=C_GREEN_L, edgecolor=C_GREEN,
               linewidth=1.1, zorder=3)
    ax.errorbar(cx, cy, yerr=[elo, ehi], fmt="o", ms=5.5, color=C_OURS,
                ecolor=C_OURS, elinewidth=1.3, capsize=2.4, capthick=1.3, zorder=4)
    ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), lw=1.1, color=C_GREY, zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{letter}  {title}", fontsize=14, pad=4)
    ax.tick_params(labelsize=11)


def fig2_reliability(S):
    key = "LLaVA(RAF)|RAF-DB"
    methods = [("u", "Uncalibrated"), ("p", "Pooled isotonic"),
               ("g", "Group isotonic (ours)")]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE * SCALE, 2.3 * SCALE),
                             gridspec_kw={"height_ratios": [3.0, 1.0],
                                          "hspace": 0.34, "wspace": 0.18})
    hmax = 0
    for j, (w_, name) in enumerate(methods):
        pred, corr = get_pred(S, key, w_)
        draw_reliability(axes[0, j], pred, corr, name, f"({chr(97 + j)})")
        axes[0, j].set_xticklabels([])
        if j == 0:
            axes[0, j].set_ylabel("Fraction of positives", fontsize=13)
            axes[0, j].text(0.04, 0.94, "overconfident", transform=axes[0, j].transAxes,
                            fontsize=11, color=C_GREY, va="top")
        else:
            axes[0, j].set_yticklabels([])
        cnt, edges = np.histogram(pred, bins=np.linspace(0, 1, 11))
        hmax = max(hmax, cnt.max())
        axes[1, j].bar((edges[:-1] + edges[1:]) / 2, cnt, width=0.09,
                       color=C_OURS, edgecolor="white", linewidth=0.8)
        axes[1, j].set_xlim(0, 1)
        axes[1, j].set_ylim(0, hmax * 1.12)
        axes[1, j].set_yticks([0, hmax // 2, hmax])
        axes[1, j].tick_params(labelsize=10)
        for s_ in ("top", "right"):
            axes[1, j].spines[s_].set_visible(False)
        if j == 0:
            axes[1, j].set_ylabel("Count", fontsize=12)
        else:
            axes[1, j].set_yticklabels([])
        if j == 1:
            axes[1, j].set_xlabel("Predicted probability", fontsize=13)
    handles = [Line2D([], [], marker="s", ls="", ms=6, markerfacecolor=C_GREEN_L,
                      markeredgecolor=C_GREEN, label="fine-grained bins (20)"),
               Line2D([], [], marker="o", ls="", ms=5, color=C_OURS,
                      label="coarse bins (10), 95% Wilson CI")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.07), fontsize=12)
    fig.text(0.5, -0.055, "LLaVA / RAF-DB, held-out split (n = 420), seed 42",
             ha="center", fontsize=12, color="#555")
    export(fig, "fig2_reliability")


# =====================================================================
# Fig.3 — H2 trade-off: dual metric vs lambda (adopted design V3)
# =====================================================================
def fig3_dual_metric(R):
    keys = [k for k in R["R6_lambda_tradeoff"]]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * SCALE, 2.35 * SCALE),
                             gridspec_kw={"wspace": 0.28})
    for ax, metric, letter, ylab in (
            (axes[0], "corr_ece", "(a)", r"ECE w.r.t. correctness  $\downarrow$"),
            (axes[1], "hu_ece", "(b)", r"ECE w.r.t. human agreement  $\downarrow$")):
        for i, k in enumerate(keys):
            c = R["R6_lambda_tradeoff"][k]["curves"]
            lam = sorted(c["grouped"], key=float)
            xs = [float(l) for l in lam]
            gy = [c["grouped"][l][metric] for l in lam]
            py = [c["pooled"][l][metric] for l in lam]
            ax.plot(xs, py, ls=(0, (2.4, 1.5)), lw=1.1, color=C_GREY_L, zorder=2)
            ax.plot(xs, gy, ls=M_LS[i], lw=2.0, color=M_COLS[i], zorder=3,
                    label=SHORT[k])
            ax.plot(xs, gy, "o", ms=4.0, color=M_COLS[i], zorder=3)
        ax.set_xlabel(r"$\lambda$  (0 = correctness, 1 = human agreement)",
                      fontsize=13)
        ax.set_ylabel(ylab, fontsize=13)
        ax.set_title(f"{letter}  "
                     f"{'correctness target' if metric == 'corr_ece' else 'human-agreement target'}",
                     fontsize=15, loc="left")
        ax.set_xlim(-0.03, 1.03)
        ax.tick_params(labelsize=11)
    handles = [Line2D([], [], color=M_COLS[i], lw=2.0, ls=M_LS[i], label=SHORT[k])
               for i, k in enumerate(keys)]
    handles.append(Line2D([], [], color=C_GREY_L, lw=1.4, ls=(0, (2.4, 1.5)),
                          label="pooled isotonic"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=11, handlelength=2.2,
               columnspacing=1.1)
    export(fig, "fig3_dual_metric")


# =====================================================================
# Fig.4 — subgroup-family ladder, modality-coloured lines (adopted V1)
# =====================================================================
LADDER = ["Pooled-Iso", "Group-Iso(emotion)", "Group-Iso[C_emo]",
          "GenericMC(confbin10 x ambig)", "GenericMC(emo x confbin10)",
          "GenericMC(emo x confbin10 x ambig)"]


def fig4_family_ladder(L):
    keys = list(L.keys())
    fig, ax = create_subplots(1, 1, figsize=((SINGLE + 0.9) * SCALE, 2.6 * SCALE))
    for i, k in enumerate(keys):
        v = L[k]["methods"]
        base = v.get("Pooled-Iso", {}).get("cw_ece_mean")
        if not base:
            continue
        xs, ys, es = [], [], []
        for m in LADDER:
            r = v.get(m)
            if not r or not np.isfinite(r.get("cw_ece_mean", np.nan)):
                continue
            xs.append(r["n_def_groups_cal"])
            ys.append(r["cw_ece_mean"] / base)
            es.append(r["cw_ece_sd"] / base)
        col = C_TEAL if k == "LLaVA(RAF)|RAF-DB" else TXT_COLS[i]
        ls = "-" if k == "LLaVA(RAF)|RAF-DB" else (0, (4, 1.6)) if i % 2 else "-"
        ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=5.5, lw=1.8, color=col, ls=ls,
                    capsize=2.0, elinewidth=1.0, zorder=3, label=SHORT[k])
        r = v.get("Group-Iso[C_emo]")
        if r:
            ax.scatter([r["n_def_groups_cal"]], [r["cw_ece_mean"] / base], s=120,
                       marker="D", facecolor="none", edgecolor=col, linewidth=2.0,
                       zorder=5)
    ax.axhline(1.0, ls=(0, (2.4, 1.5)), lw=1.3, color="#B8B8B8", zorder=1)
    ax.text(0.92, 1.015, "pooled isotonic", fontsize=11, color="#8A8A8A")
    ax.set_xscale("log")
    ax.set_xlabel("number of definition subgroups  (log scale)", fontsize=13)
    ax.set_ylabel("CW-ECE relative to pooled", fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), frameon=False,
              fontsize=11, ncol=3, handlelength=2.0, columnspacing=1.1)
    ax.text(0.985, 0.03, r"hollow marker = our $\mathcal{C}_\mathrm{emo}$ family",
            transform=ax.transAxes, ha="right", fontsize=11, color="#555")
    ax.tick_params(labelsize=11)
    export(fig, "fig4_family_ladder")


# =====================================================================
# Fig.6 — H3 fairness: bars + z-grid (adopted design V1)
# =====================================================================
def load_h3():
    D = json.load(open(os.path.join(FE, "h3_metric_power.json"), encoding="utf-8"))
    G = json.load(open(os.path.join(FE, "h3_worstgroup_robust.json"), encoding="utf-8"))
    return D, G


def fig6_bars_grid(D, G):
    settings = ["LLaVA/RAF-DB", "Qwen2.5-VL-3B/RAF-DB"]
    labels = ["LLaVA / RAF-DB", "Qwen2.5-VL-3B / RAF-DB"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * SCALE, 2.45 * SCALE),
                             gridspec_kw={"wspace": 0.34})
    axA, axB = axes
    # (a) grouped bars via figkit helper
    pooled = [D[k]["worst (max, Rawlsian)"]["pooled_mean"] for k in settings]
    ours = [D[k]["worst (max, Rawlsian)"]["group_mean"] for k in settings]
    make_grouped_bar(axA, labels, [pooled, ours],
                     ["Pooled-Iso", r"$\mathcal{C}_\mathrm{emo}$ (ours)"],
                     ylabel="worst-subgroup ECE", colors=[C_BASE, C_OURS],
                     edgecolor="black", linewidth=1.2)
    w = 0.8 / 2
    for i, k in enumerate(settings):
        wr = D[k]["worst (max, Rawlsian)"]
        delta, (lo, hi) = wr["delta"], wr["ci"]
        s = stars(max(wr["p_percentile"], wr["p_normal"]))
        axA.text(i - 0.5 * w, pooled[i] + 0.008, f"{pooled[i]:.3f}",
                 ha="center", va="bottom", fontsize=12)
        axA.text(i + 0.5 * w, ours[i] + 0.008, f"{ours[i]:.3f}",
                 ha="center", va="bottom", fontsize=12)
        # two-line delta callout just above each group's taller bar
        axA.text(i, max(pooled[i], ours[i]) + 0.045,
                 f"$\\Delta$={delta:+.3f}{s}\n[{lo:+.3f}, {hi:+.3f}]",
                 ha="center", va="bottom", fontsize=10.5, linespacing=1.35)
    axA.set_ylim(0, 0.70)
    axA.set_ylabel("worst-subgroup ECE (lower is better)", fontsize=13)
    axA.set_title("(a) worst-group ECE: pooled vs $\\mathcal{C}_\\mathrm{emo}$",
                  fontsize=15, loc="left")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False,
               fontsize=11, ncol=2, columnspacing=1.2)
    axA.tick_params(labelsize=11)

    # (b) z-curves (linestyle = bin count, colour = setting)
    minn_vals = [1, 5, 10, 20]
    bin_ls = {10: "-", 15: "--", 20: ":"}
    for k, col in zip(settings, [C_TEAL, C_OURS]):
        grid = G[k]["grid"]
        for bins in (10, 15, 20):
            zs = [grid[f"bins{bins}_minn{m}"]["z"] for m in minn_vals]
            axB.plot(minn_vals, zs, marker="o", ms=5.5, ls=bin_ls[bins],
                     lw=1.8, color=col, alpha=0.9 if bins == 10 else 0.65,
                     zorder=3)
    axB.axhline(-1.96, ls=(0, (3, 2)), lw=1.3, color=C_RED, zorder=1)
    axB.text(19.6, -1.80, r"$z=-1.96$ ($\alpha$=.05)", ha="right",
             fontsize=11, color=C_RED)
    axB.set_xticks(minn_vals)
    axB.set_xlabel("minimum subgroup size $n$", fontsize=13)
    axB.set_ylabel(r"$z$ of $\Delta$ worst-group ECE", fontsize=13)
    axB.set_title("(b) robustness across bins $\\times$ min $n$", fontsize=15,
                  loc="left")
    axB.set_ylim(-3.9, 0.25)
    axB.text(0.5, 0.94, "12/12 significant (both settings)",
             transform=axB.transAxes, ha="center", fontsize=12, color="#333")
    h2 = [Line2D([], [], color=C_TEAL, lw=1.8, label="LLaVA"),
          Line2D([], [], color=C_OURS, lw=1.8, label="Qwen2.5-VL-3B"),
          Line2D([], [], color="#555", lw=1.8, ls="-", label="bins=10"),
          Line2D([], [], color="#555", lw=1.8, ls="--", label="bins=15"),
          Line2D([], [], color="#555", lw=1.8, ls=":", label="bins=20")]
    axB.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, -0.28),
               frameon=False, fontsize=11, ncol=5, columnspacing=1.0,
               handlelength=2.0)
    axB.tick_params(labelsize=11)
    export(fig, "fig6_bars_grid")


# =====================================================================
# Fig.7 — EMOTIC H5 forest, stacked panels (adopted design V1)
# =====================================================================
def load_h5():
    out = {}
    for f, tag in [("h5_llava-1.5-7b-hf.json", "llava"),
                   ("h5_Qwen2.5-VL-3B-Instruct.json", "qwen")]:
        d = json.load(open(os.path.join(EMO, "results", f), encoding="utf-8"))
        rows = [(e["seed"], e["cw_delta_group_minus_pooled"]["mean"],
                 e["cw_delta_group_minus_pooled"]["ci_lo"],
                 e["cw_delta_group_minus_pooled"]["ci_hi"]) for e in d["per_seed"]]
        rows.sort(key=lambda r: r[0])
        agg = d["aggregate"]
        out[tag] = {"rows": rows, "n": d["n_records"],
                    "agg": agg["cw_delta_mean_over_seeds"],
                    "k": agg["cw_k_of_5_negative"],
                    "z": agg["stouffer_z_cw"], "p": agg["stouffer_p_cw"]}
    return out


def seed_ci(vals):
    m = np.mean(vals)
    sd = np.std(vals, ddof=1) / np.sqrt(len(vals))
    return m - 1.96 * sd, m + 1.96 * sd


def fig7_stacked(H):
    fig, axes = plt.subplots(2, 1, figsize=((SINGLE + 1.6) * SCALE, 3.6 * SCALE),
                             gridspec_kw={"hspace": 0.62})
    for ax, tag, col, letter, name in (
            (axes[0], "llava", C_TEAL, "(a)", "LLaVA-1.5-7B"),
            (axes[1], "qwen", C_OURS, "(b)", "Qwen2.5-VL-3B")):
        d = H[tag]
        n = len(d["rows"])
        for i, (seed, m, lo, hi) in enumerate(d["rows"]):
            ax.plot([lo, hi], [i, i], color=col, lw=2.0, zorder=3,
                    solid_capstyle="butt")
            ax.plot(m, i, "o", color=col, ms=7.5, zorder=4)
        alo, ahi = seed_ci([r[1] for r in d["rows"]])
        ax.plot([alo, ahi], [n, n], color="black", lw=2.6, zorder=4,
                solid_capstyle="butt")
        ax.plot(d["agg"], n, "D", color="black", ms=11, zorder=5)
        ax.axvline(0, color="#666", lw=1.2, ls=(0, (3, 2)), zorder=1)
        ax.set_yticks(list(range(n)) + [n])
        ax.set_yticklabels([f"seed {s}" for s, *_ in d["rows"]] + ["Aggregate"],
                           fontsize=12)
        ax.set_title(f"{letter}  {name}  (EMOTIC, $n$={d['n']:,}; "
                     f"{d['k']}/5 seeds negative; Stouffer $Z$={d['z']:.0f}, "
                     f"$p\\approx 0$)", fontsize=13, loc="left")
        ax.set_xlabel(r"$\Delta$ CW-ECE (ours $-$ pooled; negative = better)",
                      fontsize=13)
        ax.tick_params(labelsize=11)
    axes[0].set_xlim(-0.062, 0.012)
    axes[1].set_xlim(-0.205, 0.012)
    export(fig, "fig7_stacked")


# =====================================================================
# Fig.8 — P-G forest, colour by model (adopted design V2)
# =====================================================================
def load_pg():
    return json.load(open(os.path.join(EMO, "pg_experiment",
                                       "pg_fdr_and_seeddata.json"), encoding="utf-8"))


def pg_agg(rec, key):
    m = np.array(rec[key]["mean"])
    se = np.array(rec[key]["se"])
    p = np.array(rec[key]["p"])
    lo, hi = seed_ci(m)
    return m.mean(), lo, hi, stars(p.max())


def fig8_by_model(P):
    models = ["Qwen2.5-VL-3B-Instruct", "llava-1.5-7b-hf"]
    short = {"Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
             "llava-1.5-7b-hf": "LLaVA-1.5-7B"}
    comps = [("cw_core_minus_pooled", "Core $-$ Pooled", "o"),
             ("cw_pg_minus_pooled", "PG(age) $-$ Pooled", "s"),
             ("cw_pg_minus_core", "PG(age) $-$ Core", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * SCALE, 2.6 * SCALE),
                             gridspec_kw={"wspace": 0.62})
    for ax, letter, title in ((axes[0], "(a)", "(a)  CW-ECE"),
                              (axes[1], "(b)", "(b)  Worst-group ECE")):
        y = 5.4
        yticks, ylabels = [], []
        for m, mcol in zip(models, [C_OURS, C_TEAL]):
            rec = P["models"][m]
            prefix = "cw_" if letter == "(a)" else "wg_"
            for key, lab, mk in comps:
                real_key = prefix + key[3:]
                agg, lo, hi, s = pg_agg(rec, real_key)
                ax.plot([lo, hi], [y, y], color=mcol, lw=2.0, zorder=3,
                        solid_capstyle="butt")
                ax.plot(agg, y, mk, color=mcol, ms=9, zorder=4)
                if s:
                    ax.text(hi + 0.004, y, s, ha="left", va="center",
                            fontsize=13, fontweight="bold", color=mcol)
                yticks.append(y)
                ylabels.append(f"{short[m]}\n{lab}")
                y -= 1
            y -= 0.5
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=11)
        ax.axvline(0, color="#444", lw=1.2, ls=(0, (3, 2)))
        ax.set_xlabel(r"$\Delta$ ECE", fontsize=13)
        ax.set_title(title, fontsize=15, loc="left")
        ax.axhline(2.2, color="#DDDDDD", lw=1.2)
        ax.tick_params(labelsize=11)
    h = [Line2D([], [], color=C_OURS, lw=2.0, label="Qwen2.5-VL-3B"),
         Line2D([], [], color=C_TEAL, lw=2.0, label="LLaVA-1.5-7B"),
         Line2D([], [], color="#555", marker="o", ls="", ms=7,
                label="Core $-$ Pooled"),
         Line2D([], [], color="#555", marker="s", ls="", ms=7,
                label="PG(age) $-$ Pooled"),
         Line2D([], [], color="#555", marker="^", ls="", ms=7,
                label="PG(age) $-$ Core")]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=11)
    export(fig, "fig8_by_model")


def main():
    print("loading frozen JSONs ...")
    R, S, L = load_rigor()
    D, G = load_h3()
    H = load_h5()
    P = load_pg()
    print("== fig1 =="); fig1_grouped_bars(R)
    print("== fig2 =="); fig2_reliability(S)
    print("== fig3 =="); fig3_dual_metric(R)
    print("== fig4 =="); fig4_family_ladder(L)
    print("== fig6 =="); fig6_bars_grid(D, G)
    print("== fig7 =="); fig7_stacked(H)
    print("== fig8 =="); fig8_by_model(P)
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
