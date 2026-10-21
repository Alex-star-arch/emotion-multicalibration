import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

PG = "E:/WorkBuddy-work/emotion-multicalibration-paper/02-working/broaden_emotic/pg_experiment"
FIG = "E:/WorkBuddy-work/emotion-multicalibration-paper/paper/figures"
data = json.load(open(os.path.join(PG, "pg_fdr_and_seeddata.json")))
models = ["Qwen2.5-VL-3B-Instruct", "llava-1.5-7b-hf"]
short = {"Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B", "llava-1.5-7b-hf": "LLaVA-1.5-7B"}

# Okabe-Ito colorblind-safe
C_CORE = "#0072B2"   # core - pooled
C_PG   = "#E69F00"   # pg  - pooled
C_PGC  = "#009E73"   # pg  - core
Z = 1.96

comps = [("cw_core_minus_pooled", "Core - Pooled", C_CORE),
         ("cw_pg_minus_pooled",   "PG(age) - Pooled", C_PG),
         ("cw_pg_minus_core",     "PG(age) - Core", C_PGC)]
comps_w = [("wg_core_minus_pooled", "Core - Pooled", C_CORE),
           ("wg_pg_minus_pooled",   "PG(age) - Pooled", C_PG),
           ("wg_pg_minus_core",     "PG(age) - Core", C_PGC)]

def panel(ax, key_tuples, title):
    # 6 rows: 3 per model
    ymax = 6
    yticks, ylabels = [], []
    y = 5
    for m in models:
        rec = data["models"][m]
        for key, lab, col in key_tuples:
            means = np.array(rec[key]["mean"])
            ses = np.array(rec[key]["se"])
            ys = np.full(len(means), y)
            # per-seed points + CI
            for xi, ei, yy in zip(means, ses, ys):
                ax.errorbar(xi, yy, xerr=Z*ei, fmt='o', color=col, ecolor=col,
                            markersize=5, elinewidth=1.2, capsize=2.5, alpha=0.9)
            # aggregate diamond (cross-seed mean +- 1.96*SE_of_means)
            agg = means.mean()
            agg_se = means.std(ddof=1) / np.sqrt(len(means))
            ax.plot(agg, y, marker='D', color='black', markersize=9)
            ax.errorbar(agg, y, xerr=Z*agg_se, fmt='none', ecolor='black',
                        elinewidth=1.4, capsize=3)
            yticks.append(y); ylabels.append(f"{short[m]}\n{lab}")
            y -= 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(r"$\Delta$ ECE  (negative = better)", fontsize=9)
    # group separators
    ax.axhline(2.5, color='lightgray', linewidth=0.8)
    ax.set_ylim(-0.6, 5.8)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.6))
panel(ax1, comps, "(a) CW-ECE: task-weighted calibration error")
panel(ax2, comps_w, "(b) Worst-group ECE (age x gender): Rawlsian fairness")
fig.tight_layout()

out_pdf = os.path.join(FIG, "fig8_pg_forest.pdf")
out_png = os.path.join(FIG, "fig8_pg_forest.png")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print("saved", out_pdf, out_png)

# quick numeric sanity print
for m in models:
    rec = data["models"][m]
    print(short[m], "CW pg-core mean(seeds)=%.4f" % np.mean(rec["cw_pg_minus_core"]["mean"]),
          "WG pg-pool mean=%.4f" % np.mean(rec["wg_pg_minus_pooled"]["mean"]))
