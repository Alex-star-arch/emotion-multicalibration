#!/usr/bin/env python3
"""
Robustness of the worst-group fairness result before we claim it.

`h3_metric_power.py` found that replacing gap_pp (max-min) with the worst-group
(max) ECE turns H3 from null into significant on BOTH vision sets:
  LLaVA/RAF-DB        delta = -0.2582, z = -2.91, p = 0.004, 5/5
  Qwen2.5-VL-3B/RAF   delta = -0.2089, z = -2.55, p = 0.011, 5/5

Before writing that into the paper, rule out two ways it could be an artefact:

  (T1) The "worst" cell could simply be whichever cell holds fewest samples,
       making the statistic a thin-cell artefact rather than a fairness signal.
       => vary min_n over {1,5,10,20}; the result must survive cells that are
          genuinely populated.
  (T2) It could be a binning artefact. => vary bins over {10,15,20}.
  (T3) It should not be driven by one pathological split. => report per-split
       values and direction consistency, and identify which subgroup is worst.
"""

import json
import os
import collections

import numpy as np

from h3_power_scan import (load, group_iso, pooled_iso, boot_summary,
                           make_family, SEEDS, CAL_RATIO, B_BOOT)
from h3_metric_power import (metric_worst, metric_gap, _per_group_ece,
                             load_qwen)

HERE = os.path.dirname(os.path.abspath(__file__))


def run(set_name, conf, corr, emo, gender, bins_list=(10, 15, 20),
        minn_list=(1, 5, 10, 20), B=B_BOOT):
    print(f"\n########## {set_name}  (n={len(conf)}) ##########")
    fam_full = make_family([emo, gender])
    out = {}

    def make_worst(bins, min_n):
        def fn(sg, conf_, corr_):
            sg = np.asarray(sg, object)
            d = {}
            for u in np.unique(sg):
                m = sg == u
                if m.sum() >= min_n:
                    d[str(u)] = _per_group_ece_single(conf_[m], corr_[m], bins)
            v = np.array(list(d.values())) if d else np.array([])
            return float(v.max()) if len(v) >= 1 else float("nan")
        return fn

    grid = {}
    for bins in bins_list:
        for min_n in minn_list:
            fn = make_worst(bins, min_n)
            per_split, d_sets = [], []
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                perm = rng.permutation(len(conf))
                ncal = int(len(conf) * CAL_RATIO)
                cal, test = perm[:ncal], perm[ncal:]
                tc = conf[test]
                pg = group_iso(conf[cal], corr[cal], fam_full[cal], tc, fam_full[test])
                pp = pooled_iso(conf[cal], corr[cal], tc)
                per_split.append((pg, pp, corr[test], fam_full[test]))
                d_sets.append(fn(fam_full[test], pg, corr[test])
                              - fn(fam_full[test], pp, corr[test]))
            rng = np.random.default_rng(23)
            S = len(per_split)
            deltas = np.empty(B)
            for b in range(B):
                s = rng.integers(0, S)
                pg, pp, cy, fam_t = per_split[s]
                idx = rng.integers(0, len(cy), len(cy))
                deltas[b] = fn(fam_t[idx], pg[idx], cy[idx]) - fn(fam_t[idx], pp[idx], cy[idx])
            res = boot_summary(deltas, d_sets)
            grid[f"bins{bins}_minn{min_n}"] = res
            star = "  <== SIGNIFICANT" if abs(res["z"]) >= 1.96 else ""
            print(f"  bins={bins:2d} min_n={min_n:2d}  d={res['delta']:+.4f}  "
                  f"z={res['z']:+.2f}  p_pct={res['p_percentile']:.3f}  "
                  f"{res['n_splits_same_dir']}/5{star}")
    out["grid"] = grid

    # ---- which subgroup is actually worst, and how thin is it?
    print("\n  --- identity of the worst subgroup (pooled-iso vs group-iso, seed 42) ---")
    sd = SEEDS[0]
    rng = np.random.default_rng(sd)
    perm = rng.permutation(len(conf))
    ncal = int(len(conf) * CAL_RATIO)
    cal, test = perm[:ncal], perm[ncal:]
    tc = conf[test]
    pg = group_iso(conf[cal], corr[cal], fam_full[cal], tc, fam_full[test])
    pp = pooled_iso(conf[cal], corr[cal], tc)
    diam = []
    for u in np.unique(fam_full[test]):
        m = fam_full[test] == u
        if m.sum() < 5:
            continue
        eg = _per_group_ece_single(pg[m], corr[test][m], 10)
        ep = _per_group_ece_single(pp[m], corr[test][m], 10)
        diam.append((str(u), int(m.sum()), ep, eg, ep - eg))
    diam.sort(key=lambda r: -r[2])
    print(f"  {'subgroup':<28s}{'n_test':>7s}{'pooled':>9s}{'group':>9s}{'delta':>9s}")
    for g, n, ep, eg, d in diam[:6]:
        print(f"  {g:<28s}{n:>7d}{ep:>9.4f}{eg:>9.4f}{d:>9.4f}")
    worst = diam[0]
    print(f"  worst subgroup under POOLED is '{worst[0]}' with n_test={worst[1]} "
          f"(ECE {worst[2]:.4f} -> {worst[3]:.4f})")
    thin = [d for d in diam if d[1] < 10]
    print(f"  cells with n_test<10 among the top-6: {[tuple(d) for d in thin]}")
    out["worst_detail"] = [{"subgroup": d[0], "n_test": d[1], "pooled": d[2],
                            "group": d[3], "delta": d[4]} for d in diam[:6]]
    return out


def _per_group_ece_single(conf, corr, bins=10):
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


if __name__ == "__main__":
    data = load()
    ll_conf, ll_corr, emo, gender, age, race = data
    keep = np.array([g in ("male", "female") for g in gender])

    qc, qy, mism, n, acc = load_qwen()
    print(f"Qwen label-order mismatch check: {mism} (must be 0)")

    results = {}
    results["LLaVA/RAF-DB"] = run("LLaVA/RAF-DB",
                                  ll_conf[keep], np.asarray(ll_corr, int)[keep],
                                  emo[keep], gender[keep])
    results["Qwen2.5-VL-3B/RAF-DB"] = run("Qwen2.5-VL-3B/RAF-DB",
                                          qc[keep], np.asarray(qy, int)[keep],
                                          emo[keep], gender[keep])

    dst = os.path.join(HERE, "h3_worstgroup_robust.json")
    json.dump(results, open(dst, "w", encoding="utf-8"), indent=2)
    print(f"\n-> {dst}")
