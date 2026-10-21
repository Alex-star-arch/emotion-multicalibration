#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 (pre-review fix): degenerate-solution diagnostics for the isotonic
calibrators used in tab:main / tab:ablation, at the SAME standard applied to
HKRR in rigor_r5b_r8b.py (n_unique_outputs = len(unique(round(pt, 4)))).

Motivation: the pre-review found that HKRR's degenerate output (<=10 distinct
values) was disclosed, but Group-Iso itself was never checked. With ~47 cal
samples / group, an isotonic fit can collapse to a constant and produce a
spurious ECE drop. We report per-group unique-outputs / slope / sharpness for
  - Pooled-Iso
  - Group-Iso(emotion)
  - Group-Iso[C_emo]
and, for direct comparison, the same per-group diagnostics for HKRR's output
(partitioned post-hoc by the emotion family).

Pre-registered degenerate criteria (per group, on the TEST split):
  - n_unique_outputs <= 2          (constant / two-level output)
  - slope < 0.05                   (near-flat map raw conf -> calibrated)
Method-level flag mirrors the HKRR note: overall n_unique_outputs <= 10.

Output: p1_degenerate.json (stdout progress mirrors rigor style).
"""
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression

from rigor_exp_v2 import (RECORDS, SEEDS, CODE, norm_label, load_jsonl,
                          split_mask, pw_isotonic, group_iso, ece_overall)

OUT = os.path.join(CODE, "p1_degenerate.json")


# --------------------------------------------------------------- helpers
def load_set(path, side_kind, sides):
    rows = load_jsonl(path)
    ids = np.array([int(r.get("id", i)) for i, r in enumerate(rows)])
    conf = np.array([float(r["conf"]) for r in rows])
    pred = np.array([norm_label(r.get("pred_label", r.get("pred"))) for r in rows])
    corr = np.array([1 if norm_label(r.get("true")) == p else 0
                     for r, p in zip(rows, pred)])
    emo = np.array([norm_label(r.get("true")) for r in rows])
    side = sides[side_kind]
    agree = np.array([float(side.get(i, {}).get("agree", np.nan)) for i in ids])
    gender = np.array([str(side.get(i, {}).get("gender", "NA")) for i in ids])
    gender = np.where(gender == "2", "unsure", gender)
    return dict(conf=conf, corr=corr, emo=emo, agree=agree, gender=gender, n=len(rows))


def vec_diag(pt, raw):
    """Vector-level diagnostics, HKRR-comparable standard."""
    n_unique = int(len(np.unique(np.round(pt, 4))))
    sharp = float(np.std(pt)) if len(pt) else float("nan")
    span = float(np.max(pt) - np.min(pt)) if len(pt) else float("nan")
    if len(pt) >= 2 and float(np.std(raw)) > 1e-12:
        slope = float(np.polyfit(np.asarray(raw, float), np.asarray(pt, float), 1)[0])
    else:
        slope = float("nan")
    return dict(n_unique_outputs=n_unique, sharpness_std=sharp, span=span, slope=slope)


def group_fit_diag(cc, cy, csg, tc, tsg):
    """Mirror rigor_exp_v2.group_iso exactly, but capture per-group diagnostics.

    Fallback rule kept identical: groups with <2 cal rows or a single class
    get the constant mean prediction -- and are counted as degenerate.
    """
    out = np.asarray(tc, float).copy()
    cc = np.asarray(cc, float); cy = np.asarray(cy, float)
    tc = np.asarray(tc, float)
    rows = []
    for u in np.unique(csg):
        m = csg == u
        tm = tsg == u
        if tm.sum() == 0 or m.sum() == 0:
            continue
        n_cal = int(m.sum())
        constant_fit = (m.sum() < 2) or (len(np.unique(cy[m])) < 2)
        if constant_fit:
            out[tm] = float(cy[m].mean())
            used_fallback = True
        else:
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            ir.fit(cc[m], cy[m])
            out[tm] = ir.predict(tc[tm])
            used_fallback = False
        d = vec_diag(out[tm], tc[tm])
        d.update(group=str(u), n_cal=n_cal, n_test=int(tm.sum()),
                 cal_class_balance=float(cy[m].mean()),
                 used_const_fallback=used_fallback,
                 degenerate=bool(d["n_unique_outputs"] <= 2 or
                                 (np.isfinite(d["slope"]) and d["slope"] < 0.05)))
        rows.append(d)
    return out, rows


def summarise(rows):
    if not rows:
        return {}
    nu = np.array([r["n_unique_outputs"] for r in rows], float)
    sl = np.array([r["slope"] for r in rows], float)
    sh = np.array([r["sharpness_std"] for r in rows], float)
    ncal = np.array([r["n_cal"] for r in rows], float)
    return dict(
        n_groups=len(rows),
        n_cal_median=float(np.median(ncal)),
        n_unique_median=float(np.median(nu)),
        n_unique_min=int(nu.min()),
        n_unique_p25=float(np.percentile(nu, 25)),
        frac_groups_uniq_le2=float(np.mean(nu <= 2)),
        slope_median=float(np.nanmedian(sl)),
        slope_frac_lt_005=float(np.nanmean(sl < 0.05)),
        sharpness_median=float(np.median(sh)),
        n_const_fallback=int(sum(r["used_const_fallback"] for r in rows)),
        n_degenerate=int(sum(r["degenerate"] for r in rows)),
    )


def main():
    goe = {r["id"]: r for r in load_jsonl(os.path.join(CODE, "goe_annotator_sidecar.jsonl"))}
    raf = {r["id"]: r for r in load_jsonl(os.path.join(CODE, "raf_demographic_sidecar.jsonl"))}
    sides = {"goe": goe, "raf": raf}

    out = {}
    for (model, ds, mod, path, side_kind) in RECORDS:
        if not os.path.exists(path):
            print(f"  !! missing {path}", flush=True)
            continue
        key = f"{model}|{ds}"
        D = load_set(path, side_kind, sides)
        n = D["n"]; conf, corr, emo, agree, gen = (
            D["conf"], D["corr"], D["emo"], D["agree"], D["gender"])
        has_ambig = not np.all(np.isnan(agree))
        res = {}

        for s in SEEDS:
            cal = split_mask(n, seed=s)
            # definition families identical to rigor_r5b_r8b.def_family
            if has_ambig:
                cuts = np.percentile(agree[cal][~np.isnan(agree[cal])], [33, 67])
                tt = np.where(agree <= cuts[0], "lo", np.where(agree <= cuts[1], "mid", "hi"))
            fams = {
                "Pooled-Iso": np.array(["ALL"] * n),
                "Group-Iso(emotion)": emo,
            }
            if mod == "text" and has_ambig:
                fams["Group-Iso[C_emo]"] = np.array([f"{a}__{b}" for a, b in zip(emo, tt)])
            else:
                fams["Group-Iso[C_emo]"] = np.array([f"{a}__{b}" for a, b in zip(emo, gen)])

            for mname, dfam in fams.items():
                if mname == "Pooled-Iso":
                    pt = pw_isotonic(conf[cal], corr[cal], conf[~cal])
                    # single pseudo-group diagnostics
                    d = vec_diag(pt, conf[~cal])
                    d.update(group="ALL", n_cal=int(cal.sum()),
                             n_test=int((~cal).sum()),
                             cal_class_balance=float(corr[cal].mean()),
                             used_const_fallback=bool(len(np.unique(corr[cal])) < 2),
                             degenerate=bool(d["n_unique_outputs"] <= 2))
                    rows = [d]
                else:
                    pt, rows = group_fit_diag(conf[cal], corr[cal], dfam[cal],
                                              conf[~cal], dfam[~cal])
                overall = vec_diag(pt, conf[~cal])
                entry = res.setdefault(mname, {"per_seed": [], "per_group_rows": []})
                entry["per_seed"].append(dict(
                    seed=s, overall=overall, summary=summarise(rows),
                    test_ece=float(ece_overall(pt, corr[~cal])),
                    uncalibrated_test_ece=float(ece_overall(conf[~cal], corr[~cal]))))
                entry["per_group_rows"].extend(rows)

        # HKRR at the same standard (best-effort, seed 42 as in rigor_r5b_r8b)
        try:
            from multicalibration import MulticalibrationPredictor
            s = 42
            cal = split_mask(n, seed=s)
            uniq = np.unique(np.concatenate([emo[cal], emo[~cal]]))
            S_c = (emo[cal][:, None] == uniq[None, :]).astype(int)
            S_t = (emo[~cal][:, None] == uniq[None, :]).astype(int)
            p = MulticalibrationPredictor("HKRR", verbose=False)
            p.fit(np.clip(conf[cal], 1e-3, 1 - 1e-3), corr[cal], S_c,
                  {"lambda": 0.1, "alpha": 0.05, "max_iter": 20,
                   "randomized": True, "use_oracle": True})
            pt = np.asarray(p.predict(np.clip(conf[~cal], 1e-3, 1 - 1e-3), S_t)).ravel()
            # POST-HOC partition of HKRR's own output by the emotion family
            # (no refit) so the diagnostics sit at the same standard as above.
            rows = []
            emo_t = emo[~cal]
            for u in np.unique(emo_t):
                m = emo_t == u
                if m.sum() == 0:
                    continue
                d = vec_diag(pt[m], conf[~cal][m])
                d.update(group=str(u), n_cal=-1, n_test=int(m.sum()),
                         cal_class_balance=float("nan"),
                         used_const_fallback=False,
                         degenerate=bool(d["n_unique_outputs"] <= 2 or
                                         (np.isfinite(d["slope"]) and d["slope"] < 0.05)))
                rows.append(d)
            overall = vec_diag(pt, conf[~cal])
            res["HKRR(official pkg)"] = {
                "per_seed": [dict(seed=42, overall=overall, summary=summarise(rows),
                                  test_ece=float(ece_overall(pt, corr[~cal])),
                                  uncalibrated_test_ece=float(ece_overall(conf[~cal], corr[~cal])))],
                "per_group_rows": rows,
                "note": "official package, same fit as rigor_r5b_r8b R5' HKRR row"}
            print(f"  {key:22s} HKRR overall uniq={overall['n_unique_outputs']}"
                  f" slope={overall['slope']:.3f} sharp={overall['sharpness_std']:.4f}",
                  flush=True)
        except Exception as e:
            print("   HKRR skipped:", type(e).__name__, str(e)[:90], flush=True)

        # collapse per-seed -> method level
        collapsed = {}
        for mname, entry in res.items():
            ps = entry["per_seed"]
            collapsed[mname] = dict(
                overall_n_unique_mean=float(np.mean([p["overall"]["n_unique_outputs"] for p in ps])),
                overall_n_unique_min=int(min(p["overall"]["n_unique_outputs"] for p in ps)),
                overall_slope_mean=float(np.nanmean([p["overall"]["slope"] for p in ps])),
                overall_sharpness_mean=float(np.mean([p["overall"]["sharpness_std"] for p in ps])),
                overall_span_mean=float(np.mean([p["overall"]["span"] for p in ps])),
                groups_summary_across_seeds=summarise(entry["per_group_rows"]),
                per_seed=ps,
                per_group_rows=entry["per_group_rows"],
            )
            s = collapsed[mname]["groups_summary_across_seeds"]
            print(f"  {key:22s} {mname:20s} overall_uniq="
                  f"{collapsed[mname]['overall_n_unique_mean']:.1f}"
                  f"  grp_uniq_med={s.get('n_unique_median', float('nan')):.1f}"
                  f"  deg_grps={s.get('n_degenerate', 0)}/{s.get('n_groups', 0)}"
                  f"  slope_med={s.get('slope_median', float('nan')):.3f}"
                  f"  sharp_med={s.get('sharpness_median', float('nan')):.4f}",
                  flush=True)
        out[key] = {"modality": mod, "n": n, "methods": collapsed}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
