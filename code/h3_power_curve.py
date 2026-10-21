#!/usr/bin/env python3
"""
How big must the H3 corpus be?

Why this exists
---------------
`h3_power_scan.py` showed that on the current 1400-row RAF-DB corpus no
demographic design reaches significance: the strongest is emo x GENDER with
delta = -0.154, z = -1.69, p = 0.092-0.118 (5/5 splits consistent). The question
the paper needs answered is whether that is a *method* failure or a *power*
failure, and if the latter, how many rows fix it.

We answer it empirically rather than by assuming SE ~ 1/sqrt(n). The gap is a
max-min statistic, whose bootstrap SE need NOT follow the parametric 1/sqrt(n)
rate; assuming it would understate the required corpus. So we measure the SE at
several corpus sizes by emotion-stratified subsampling, fit

    log SE(N) = a + k * log N

directly, and solve for the N at which |delta| / SE(N) = 1.96.

Caveat stated up front: this extrapolates the SE only. The effect size itself is
held at its full-corpus value, so the answer is optimistic if the true effect
shrinks on a larger, naturally-distributed corpus.
"""

import json
import os

import numpy as np

from h3_power_scan import (load, make_family, _gap_bootstrap,
                           SEEDS, B_BOOT)

HERE = os.path.dirname(os.path.abspath(__file__))

SIZES = [1400, 1000, 700, 500]
REPEATS = 3
B_LOCAL = 800


def stratified_subsample(emo, n_total, rng):
    """Emotion-stratified subsample of size n_total (proportional allocation)."""
    classes = np.unique(emo)
    per = max(1, n_total // len(classes))
    idx = []
    for cl in classes:
        pool = np.where(emo == cl)[0]
        take = min(per, len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
    return np.concatenate(idx)


def curve(name, parts, demo, keep, cell_mode):
    conf, corr, emo, gender, age, race = DATA
    keep = np.asarray(keep, bool)
    fam = make_family(parts)
    c_all, y_all, f_all, d_all, e_all = (
        conf[keep], corr[keep], fam[keep], np.asarray(demo, object)[keep], emo[keep])

    print(f"\n### {name}  (rows available: {len(c_all)})")
    rows = []
    for N in SIZES:
        ses, dels = [], []
        for rep in range(REPEATS):
            rng = np.random.default_rng(1000 + rep)
            if N >= len(c_all):
                idx = np.arange(len(c_all))
            else:
                idx = stratified_subsample(e_all, N, rng)
            res = _gap_bootstrap(c_all[idx], y_all[idx], d_all[idx], f_all[idx],
                                 cell_mode, seeds=SEEDS, B=B_LOCAL,
                                 rng_seed=31 + rep)
            ses.append(res["boot_se"])
            dels.append(res["delta"])
        sd_m = float(np.mean(ses))
        dl_m = float(np.mean(dels))
        rows.append((N, sd_m, dl_m))
        print(f"   N={N:5d}   mean delta={dl_m:+.4f}   boot SE={sd_m:.4f}   "
              f"z={dl_m / sd_m:+.2f}")

    # full-corpus effect, used as the effect-size anchor
    full = _gap_bootstrap(c_all, y_all, d_all, f_all, cell_mode,
                          seeds=SEEDS, B=B_BOOT, rng_seed=7)
    delta_full = full["delta"]
    se_full = full["boot_se"]
    print(f"   full corpus N={len(c_all)}: delta={delta_full:+.4f} SE={se_full:.4f} "
          f"z={delta_full / se_full:+.2f}")

    # fit log SE = a + k log N
    logs_N = np.log([r[0] for r in rows])
    logs_SE = np.log([r[1] for r in rows])
    k, a = np.polyfit(logs_N, logs_SE, 1)
    print(f"   fitted scaling: SE ~ N^({k:.3f})   (1/sqrt would be -0.500)")

    target_se = abs(delta_full) / 1.96
    N_star = float(np.exp((np.log(target_se) - a) / k))
    print(f"   need SE <= {target_se:.4f} for |z|=1.96")
    print(f"   => projected corpus N* = {N_star:,.0f} rows "
          f"({N_star / len(c_all):.2f}x current)")
    return {
        "name": name, "cell_mode": cell_mode,
        "curve": [{"N": r[0], "se": r[1], "delta": r[2]} for r in rows],
        "full": {"N": len(c_all), "delta": delta_full, "se": se_full,
                 "z": delta_full / se_full,
                 "p_normal": full["p_normal"], "p_percentile": full["p_percentile"],
                 "rel_reduction_pct": full["rel_reduction_pct"]},
        "scaling_exponent": float(k), "intercept": float(a),
        "projected_N_for_p05": N_star,
    }


if __name__ == "__main__":
    DATA = load()
    conf, corr, emo, gender, age, race = DATA
    keep_gender = np.array([g in ("male", "female") for g in gender])
    keep_all = np.ones(len(conf), bool)

    out = {}
    print("=" * 84)
    print("H3 power curve — how much corpus does the fairness claim need?")
    print("=" * 84)

    out["A_gender_cells"] = curve(
        "A: emo x GENDER, gap_cells (the metric used in the draft)",
        [emo, gender], gender, keep_gender, True)
    out["C_age_demo"] = curve(
        "C: emo x AGE, gap_demo (clean demographic axis)",
        [emo, age], age, keep_all, False)

    dst = os.path.join(HERE, "h3_power_curve.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=2)
    print(f"\n-> {dst}")
