#!/usr/bin/env python3
"""
H3 (fairness coupling) — why it was null, and which design actually has power.

Background
----------
`analysis_race_axis.py` showed the previous null result is a FLOOR EFFECT:
on RAF-DB the frozen LLaVA outputs barely differ across gender
(ECE gap 0.019), while age (0.0945) and race (0.0545) carry real disparity.
There was almost nothing for a group calibrator to remove.

This script tests each candidate design with the same nested bootstrap used for
the primary claims (draw a split, then resample within it), using only the
already-frozen per-sample outputs. It reports, per design:

  gap_cells : max-min ECE over ALL cells of the evaluation family
              (the metric used in the previous paper draft -- confounds
               EMOTION disparity into a "fairness" number)
  gap_demo  : max-min ECE over DEMOGRAPHIC groups, marginalising emotion
              (the cleaner fairness metric)

Both compare Group-Iso vs Pooled-Iso *within the same evaluation family*, which
keeps them binning-bias comparable.

No GPU is used: every quantity is derived from `rigor_samples_v2.json`.
"""

import json
import math
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "01-input", "external_data"))
RACE = {0: "caucasian", 1: "african-american", 2: "asian"}

SEEDS = [42, 43, 44, 45, 46]
CAL_RATIO = 0.70
B_BOOT = 2000


# ------------------------------------------------------------------ helpers
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


def group_iso(cc, cy, csg, tc, tsg):
    """Per-subgroup isotonic; rows whose subgroup is unseen default to the
    subgroup-conditional base rate (see the uninitialised-memory bug)."""
    out = np.asarray(tc, float).copy()
    for u in np.unique(csg):
        m = csg == u
        tm = tsg == u
        if tm.sum() == 0 or m.sum() == 0:
            continue
        if m.sum() < 2 or len(np.unique(cy[m])) < 2:
            out[tm] = float(cy[m].mean())
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit(cc[m], cy[m])
        out[tm] = ir.predict(tc[tm])
    return out


def pooled_iso(cc, cy, tc):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(cc, cy)
    return ir.predict(tc)


def gap_cells(sg, conf, corr, bins=10, min_n=1):
    """Previous paper metric: max-min ECE over every cell of the family."""
    sg = np.asarray(sg, object)
    vals = []
    for u in np.unique(sg):
        m = sg == u
        if m.sum() >= min_n:
            vals.append(_ece(conf[m], corr[m], bins))
    return (max(vals) - min(vals)) if len(vals) >= 2 else float("nan")


def gap_demo(dem, conf, corr, bins=10, min_n=1):
    """Cleaner fairness metric: max-min ECE over demographic groups only."""
    dem = np.asarray(dem, object)
    vals = []
    for u in np.unique(dem):
        m = (dem == u)
        if m.sum() >= min_n:
            vals.append(_ece(conf[m], corr[m], bins))
    return (max(vals) - min(vals)) if len(vals) >= 2 else float("nan")


def boot_summary(d, per_split=None):
    """Two conventions of the bootstrap p, both reported.

    The project standard (rigor_r5b_r8b.py) is the PERCENTILE form
        p = 2 * min(P(d <= 0), P(d >= 0))
    i.e. how far the bootstrap distribution straddles zero. A previous draft of
    this file instead used Hall's shift (d - mean), which tests H0: Delta = mean
    and therefore always returns p ~ 1. That is a tautology, not a test; do not
    reintroduce it.
    """
    d = np.asarray(d, float)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_pct = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
    sd = float(d.std(ddof=1))
    z = float(d.mean() / sd) if sd > 0 else float("nan")
    # two-sided normal p = 2*Phi(-|z|) = erfc(|z|/sqrt(2))
    p_norm = float(math.erfc(abs(z) / math.sqrt(2))) if sd > 0 else float("nan")
    out = {
        "delta": float(d.mean()),
        "ci": [float(lo), float(hi)],
        "boot_se": sd,
        "z": z,
        "p_percentile": p_pct,
        "p_normal": p_norm,
    }
    if per_split is not None:
        ps = np.asarray(per_split, float)
        out["per_split"] = [float(x) for x in ps]
        out["n_splits_same_dir"] = int(np.sum(ps < 0))
    return out


# ------------------------------------------------------------------ main run
def load():
    import csv
    samples = json.load(open(os.path.join(HERE, "rigor_samples_v2.json"), encoding="utf-8"))
    v = samples["LLaVA(RAF)|RAF-DB"]
    conf = np.asarray(v["conf"], float)
    corr = np.asarray(v["corr"], int)
    emo = np.asarray(v["emo"], object)
    gender = np.asarray(v["gender"], object)
    age = np.asarray(v["age"], object)

    side = [json.loads(l) for l in open(
        os.path.join(HERE, "raf_demographic_sidecar.jsonl"), encoding="utf-8")]
    aligned = {}
    with open(os.path.join(ROOT, "rafdb_aligned.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aligned[row["Image"].strip()] = row
    race = []
    for s in side:
        key = (s["official_name"] or "").replace(".jpg", "_aligned.jpg")
        r = aligned.get(key)
        race.append(RACE.get(int(r["race"]), "__NORACE__") if r else "__NORACE__")
    race = np.asarray(race, object)
    return conf, corr, emo, gender, age, race


def make_family(parts):
    return np.array(["__".join(str(p) for p in row) for row in zip(*parts)], object)


def _gap_bootstrap(c, y_all, demo_axis, fam, cell_mode,
                   seeds=SEEDS, B=B_BOOT, rng_seed=7):
    """Nested bootstrap of the gap contrast on an arbitrary corpus `c`.

    `c` = confidence array, `y_all` = correctness, both already restricted to the
    rows to analyse. Leaves no module-level globals, so it can be reused for
    subsampling experiments.

    Rows whose demographic annotation is unusable ('unsure' gender, '__NORACE__')
    must have been dropped by the caller; keeping them would create a spurious
    "group" whose ECE difference masquerades as demographic disparity.
    """
    c = np.asarray(c, float)
    y = np.asarray(y_all, int)
    fam = np.asarray(fam, object)
    dm = np.asarray(demo_axis, object)
    per_split = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        perm = rng.permutation(len(c))
        ncal = int(len(c) * CAL_RATIO)
        cal, test = perm[:ncal], perm[ncal:]
        tc = c[test]
        pg = group_iso(c[cal], y[cal], fam[cal], tc, fam[test])
        pp = pooled_iso(c[cal], y[cal], tc)
        per_split.append((pg, pp, y[test], fam[test], dm[test]))

    rng = np.random.default_rng(rng_seed)
    S = len(per_split)
    deltas = np.empty(B)
    base_g, base_p = [], []
    ps_vals = []
    for (pg, pp, cy, fam_t, dem_t) in per_split:
        if cell_mode:
            ps_vals.append(gap_cells(fam_t, pg, cy) - gap_cells(fam_t, pp, cy))
        else:
            ps_vals.append(gap_demo(dem_t, pg, cy) - gap_demo(dem_t, pp, cy))
    for b in range(B):
        s = rng.integers(0, S)
        pg, pp, cy, fam_t, dem_t = per_split[s]
        idx = rng.integers(0, len(cy), len(cy))
        if cell_mode:
            dg, dp = gap_cells(fam_t[idx], pg[idx], cy[idx]), gap_cells(fam_t[idx], pp[idx], cy[idx])
        else:
            dg, dp = gap_demo(dem_t[idx], pg[idx], cy[idx]), gap_demo(dem_t[idx], pp[idx], cy[idx])
        deltas[b] = dg - dp
        base_g.append(dg); base_p.append(dp)

    res = boot_summary(deltas, ps_vals)
    res.update({
        "group_gap_mean": float(np.mean(base_g)),
        "pooled_gap_mean": float(np.mean(base_p)),
        "rel_reduction_pct": float(100 * (np.mean(base_p) - np.mean(base_g)) / np.mean(base_p)),
        "n_corpus": int(len(c)),
    })
    return res


if __name__ == "__main__":
    DATA = load()
    conf, corr, emo, gender, age, race = DATA
    print(f"loaded {len(conf)} frozen LLaVA/RAF-DB samples")

    # rows whose demographic annotation is unusable are dropped per design
    keep_gender = np.array([g in ("male", "female") for g in gender])
    keep_race = np.array([r != "__NORACE__" for r in race])
    keep_all = keep_gender & keep_race      # rows usable under every design

    designs = [
        # name, family parts, demographic axis, rows to keep
        ("A: emo x GENDER (previous design)", [emo, gender], gender, keep_gender),
        ("B: emo x RACE (unused axis)",       [emo, race],   race,   keep_race),
        ("C: emo x AGE (unused axis)",        [emo, age],    age,    np.ones(len(conf), bool)),
        ("D: emo x AGE x GENDER (3-way)",     [emo, age, gender],
         np.array([f"{a}__{g}" for a, g in zip(age, gender)], object), keep_gender),
    ]

    out = {}
    print("\n" + "=" * 88)
    print("H3 fairness designs — Group-Iso vs Pooled-Iso, evaluated in the SAME family")
    print("=" * 88)
    for name, parts, demo, keep in designs:
        keep = np.asarray(keep, bool)
        fam = make_family(parts)
        rc = _gap_bootstrap(conf[keep], corr[keep], demo[keep], fam[keep], True)
        rd = _gap_bootstrap(conf[keep], corr[keep], demo[keep], fam[keep], False)
        out[name] = {"gap_cells": rc, "gap_demo": rd}
        print(f"\n### {name}   (rows kept: {int(np.sum(keep))}/{len(conf)})")
        for tag, r in (("gap_cells (previous metric)", rc),
                       ("gap_demo  (clean demographic)", rd)):
            print(f"  {tag}")
            print(f"     pooled {r['pooled_gap_mean']:.4f} -> group {r['group_gap_mean']:.4f}"
                  f"  |  delta {r['delta']:+.4f} {np.round(r['ci'], 4).tolist()}")
            print(f"     z={r['z']:+.2f}  p_percentile={r['p_percentile']:.3f}  "
                  f"p_normal={r['p_normal']:.3f}  |  rel {r['rel_reduction_pct']:.1f}%  "
                  f"|  {r['n_splits_same_dir']}/5 splits same direction")

    dst = os.path.join(HERE, "h3_design_power.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=2)
    print(f"\n-> {dst}")
