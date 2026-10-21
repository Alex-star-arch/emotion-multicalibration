#!/usr/bin/env python3
"""
Inner-loop experiment: can a better-powered fairness metric rescue H3
on the EXISTING data, without any new images or weights?

Context
-------
Probing the A100 showed there is no larger corpus available:
  - upstream LLaVA/RAF-DB records are exactly 1400 rows
  - upstream Qwen2.5-VL-3B/RAF-DB records are exactly 1400 rows
  - only 1540 RAF-DB images sit on the instance (1400 eval + 140 cal)
  - the HF cache holds no weights (12 KB)
So "get more rows" is blocked. Before conceding, ask whether the *estimand* is
the problem.

Hypothesis
----------
The paper's fairness metric gap_pp = max - min per-subgroup ECE is dominated by
the MINIMUM, which lives in the lower tail and is highly unstable when each cell
holds ~20 test rows. Dropping it should cut variance without changing what we
mean by fairness: the standard Rawlsian definition of fairness is the WORST
group's loss, not the spread.

Competing metrics:
  gap       : max - min   (current)
  worst     : max         (standard worst-group / min-max fairness)
  sd        : std across subgroup ECE (average dispersion, smoothest)

If `worst` or `sd` reaches |z| >= 1.96 while `gap` does not, the honest move is
to change the estimand and say so explicitly -- not to change the data.
"""

import json
import os

import numpy as np

from h3_power_scan import (load, make_family, group_iso, pooled_iso,
                           boot_summary, SEEDS, CAL_RATIO, B_BOOT)

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.normpath(os.path.join(HERE, "..", "..", "01-input", "external_data"))

QWEN_RECORDS = os.path.join(EXT, "qwen25vl3b_rafdb_eval1400.jsonl")
MANIFEST = os.path.join(EXT, "rafdb_eval1400.jsonl")


# --------------------------------------------------------------- metrics
def _per_group_ece(sg, conf, corr, bins=10, min_n=1):
    sg = np.asarray(sg, object)
    d = {}
    for u in np.unique(sg):
        m = sg == u
        if m.sum() >= min_n:
            d[str(u)] = _ece(conf[m], corr[m], bins)
    return np.array(list(d.values())) if d else np.array([])


def _ece(conf, corr, bins=10):
    conf = np.asarray(conf, float)
    corr = np.asarray(corr, int)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(conf) * abs(conf[m].mean() - corr[m].mean())
    return tot


def metric_gap(sg, conf, corr):
    v = _per_group_ece(sg, conf, corr)
    return float(v.max() - v.min()) if len(v) >= 2 else float("nan")


def metric_worst(sg, conf, corr):
    v = _per_group_ece(sg, conf, corr)
    return float(v.max()) if len(v) >= 1 else float("nan")


def metric_sd(sg, conf, corr):
    v = _per_group_ece(sg, conf, corr)
    return float(v.std(ddof=0)) if len(v) >= 2 else float("nan")


METRICS = {"gap (max-min, current)": metric_gap,
           "worst (max, Rawlsian)": metric_worst,
           "sd (dispersion)": metric_sd}


# --------------------------------------------------------------- engine
def run(set_name, conf, corr, fam, B=B_BOOT, seeds=SEEDS):
    per_split = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        perm = rng.permutation(len(conf))
        ncal = int(len(conf) * CAL_RATIO)
        cal, test = perm[:ncal], perm[ncal:]
        tc = conf[test]
        pg = group_iso(conf[cal], corr[cal], fam[cal], tc, fam[test])
        pp = pooled_iso(conf[cal], corr[cal], tc)
        per_split.append((pg, pp, corr[test], fam[test]))

    rng = np.random.default_rng(11)
    S = len(per_split)
    out = {}
    for mname, fn in METRICS.items():
        deltas = np.empty(B)
        ps_vals, g_vals, p_vals = [], [], []
        for (pg, pp, cy, fam_t) in per_split:
            ps_vals.append(fn(fam_t, pg, cy) - fn(fam_t, pp, cy))
            g_vals.append(fn(fam_t, pg, cy))
            p_vals.append(fn(fam_t, pp, cy))
        for b in range(B):
            s = rng.integers(0, S)
            pg, pp, cy, fam_t = per_split[s]
            idx = rng.integers(0, len(cy), len(cy))
            deltas[b] = fn(fam_t[idx], pg[idx], cy[idx]) - fn(fam_t[idx], pp[idx], cy[idx])
        res = boot_summary(deltas, ps_vals)
        res.update({
            "group_mean": float(np.mean(g_vals)),
            "pooled_mean": float(np.mean(p_vals)),
            "rel_reduction_pct": float(100 * (np.mean(p_vals) - np.mean(g_vals)) / np.mean(p_vals)),
            "n": int(len(conf)),
        })
        out[mname] = res
    return out


def load_qwen():
    """Qwen2.5-VL-3B/RAF-DB records; verify row order matches the eval manifest."""
    recs = [json.loads(l) for l in open(QWEN_RECORDS, encoding="utf-8")]
    mani = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    assert len(recs) == len(mani), f"length mismatch {len(recs)} vs {len(mani)}"
    mism = sum(1 for r, m in zip(recs, mani)
               if str(r.get("true", "")).strip().lower() != str(m.get("label", "")).strip().lower())
    conf = np.array([float(r["conf"]) for r in recs])
    corr = np.array([1 if str(r.get("pred_label", r.get("pred", ""))).strip().lower()
                     == str(r.get("true", "")).strip().lower() else 0 for r in recs])
    acc = float(corr.mean())
    return conf, corr, mism, len(recs), acc


if __name__ == "__main__":
    data = load()
    ll_conf, ll_corr, emo, gender, age, race = data
    keep = np.array([g in ("male", "female") for g in gender])

    sets = {"LLaVA/RAF-DB": (ll_conf[keep], ll_corr[keep], make_family([emo, gender])[keep])}

    if os.path.exists(QWEN_RECORDS):
        qc, qy, mism, n, acc = load_qwen()
        print(f"Qwen2.5-VL-3B records: n={n}, label-order mismatches vs manifest={mism}, "
              f"accuracy={acc:.3f}")
        if mism == 0:
            sets["Qwen2.5-VL-3B/RAF-DB"] = (qc[keep], qy[keep],
                                            make_family([emo, gender])[keep])
        else:
            print("  !! row order NOT aligned with the manifest; demographics unsafe, skipped")

    print("\n" + "=" * 96)
    print("Fairness-metric power comparison — Group-Iso vs Pooled-Iso, family = emo x gender")
    print("=" * 96)
    results = {}
    for sname, (c, y, fam) in sets.items():
        c = np.asarray(c, float); y = np.asarray(y, int)
        res = run(sname, c, y, fam)
        results[sname] = res
        print(f"\n### {sname}   (rows kept {len(c)}, accuracy {y.mean():.3f})")
        for mname, r in res.items():
            flag = "  <== SIGNIFICANT" if abs(r["z"]) >= 1.96 else ""
            print(f"  {mname:<26s} pooled {r['pooled_mean']:.4f} -> group {r['group_mean']:.4f}"
                  f"  d={r['delta']:+.4f}  z={r['z']:+.2f}  "
                  f"p_pct={r['p_percentile']:.3f}  p_norm={r['p_normal']:.3f}  "
                  f"rel={r['rel_reduction_pct']:.1f}%  {r['n_splits_same_dir']}/5{flag}")

    dst = os.path.join(HERE, "h3_metric_power.json")
    json.dump(results, open(dst, "w", encoding="utf-8"), indent=2)
    print(f"\n-> {dst}")
