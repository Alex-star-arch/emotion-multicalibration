#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rigor-hardening experiment for the emotion-multicalibration paper.

Motivation (statistical audit of formal_exp_v1b / phase2_full):
  P0-1  main inference rested on a SINGLE cal/test split (bootstrap_seed42)
  P0-2  seed-level Wilcoxon used 5 resplits of the SAME data -> observations
        are dependent, and n=5 gives a minimum two-sided p of 0.0625
  P0-3  no effect sizes were reported at all
  P1-4  BH-FDR pooled p-values across heterogeneous families and covered only
        one of the two primary contrasts
  P1-5  CW-ECE / gap_pp depend on the bin count -> no bin sensitivity check
  P1-6  3-level subgroup family produced 62 subgroups over n=1400 (mean 22.6)
        -> tiny cells can dominate the weighted metric (no min-n guard)
  P1-7  no generic-multicalibration baseline in Phase 2
  P1-8  RAF-DB gender value 2 ("unsure", n=81) was not mapped and thus silently
        entered the fairness analysis as a third gender group
  P2-9  H2 Spearman correlation was computed on cal+test pooled (leaky)
  P2-10 H2 overconfidence statistic used one split only
  P2-11 no split-to-split variability reported
  P2-12 H2 only compared the two endpoints (lambda=0,1) -> no trade-off curve
  P2-13 vision side has a single model

This script delivers:
  R1  nested (cluster-aware) bootstrap: 5 splits x B, capturing BOTH
      split-to-split variation and within-split sampling variation
  R2  effect sizes: bootstrap z = delta/SE, paired Cohen's d_z on per-sample
      Brier losses, and relative improvement %
  R3  split-to-split SD / range / direction-consistency (k of 5)
  R4  sensitivity: bins in {10,15,20}; min-subgroup-n in {1,5,20}
  R5  generic multicalibration baseline (HKRR, Devic 2024 pkg) fitted on a
      confidence-binned generic family AND on our own C_emo family
  R6  H2 trade-off frontier: lambda-sweep over the soft target
      y_lambda = (1-lambda)*correctness + lambda*human_agreement
  R7  per-sample export for reliability diagrams / distribution plots
  R8  fairness reconsidered with the correct "unsure" handling

Output: rigor_results_v2.json  (+ rigor_samples_v2.json)
"""
import json
import os
import time

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FER_MAP = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad',
           5: 'surprise', 6: 'neutral'}
SEEDS = [42, 43, 44, 45, 46]
B_BOOT = 2000
CAL = 0.7
CODE = "./mc_data/code"
R10 = "./trustbench_emo_data/results/r10"

RECORDS = [
    ("Gemma-2-9B",    "GoEmotions", "text",
     f"{R10}/GoE_Gemma2_9B_main/records.jsonl", "goe"),
    ("Llama-3-8B",    "GoEmotions", "text",
     f"{R10}/GoE_Llama3_8B_main/records.jsonl", "goe"),
    ("Qwen2.5-7B",    "GoEmotions", "text",
     f"{R10}/GoE_Qwen25_7B_main/records.jsonl", "goe"),
    ("Qwen2.5-3B",    "GoEmotions", "text",
     f"{R10}/GoE_Qwen25_3B_main/records.jsonl", "goe"),
    ("Qwen2.5-VL-3B", "GoEmotions", "text",
     f"{R10}/GoE_Qwen25VL3B_main/records.jsonl", "goe"),
    ("LLaVA(RAF)",    "RAF-DB",     "vision",
     "./trustbench_emo_data/results/r15_llava_rafdb/eval/records.jsonl", "raf"),
]


def norm_label(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return FER_MAP.get(v, str(v))
    s = str(v)
    if s.isdigit():
        return FER_MAP.get(int(s), s)
    return s


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


# ------------------------------- metrics -------------------------------
def _binned_rates(conf, corr, bins, min_n=1):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    num, den = 0.0, 0.0
    for i in range(bins):
        m = idx == i
        n = int(m.sum())
        if n >= min_n:
            num += abs(conf[m].mean() - corr[m].mean()) * n
            den += n
    return (num / den) if den > 0 else np.nan


def ece_overall(conf, corr, bins=10, min_n=1):
    return float(_binned_rates(np.asarray(conf, float), np.asarray(corr, float),
                               bins, min_n))


def soft_ece(conf, soft_target, bins=10, min_n=1):
    return float(_binned_rates(np.asarray(conf, float),
                               np.asarray(soft_target, float), bins, min_n))


def per_group_ece(conf, corr, sg, bins=10, min_n=1):
    d = {}
    for u in np.unique(sg):
        m = sg == u
        v = _binned_rates(conf[m], corr[m], bins, min_n=1)
        d[str(u)] = (float(v) if np.isfinite(v) else np.nan, int(m.sum()))
    return d


def cw_ece(conf, corr, sg, bins=10, min_n=1):
    """Class-weighted ECE; subgroups with n < min_n are dropped."""
    d = per_group_ece(conf, corr, sg, bins)
    ks = [k for k in d if d[k][1] >= min_n and np.isfinite(d[k][0])]
    if not ks:
        return np.nan
    cnt = np.array([d[k][1] for k in ks], float)
    val = np.array([d[k][0] for k in ks], float)
    return float(np.sum(cnt * val) / cnt.sum())


def gap_pp(conf, corr, sg, bins=10, min_n=1):
    d = per_group_ece(conf, corr, sg, bins)
    ks = [k for k in d if d[k][1] >= min_n and np.isfinite(d[k][0])]
    if len(ks) < 2:
        return np.nan
    v = np.array([d[k][0] for k in ks], float)
    return float(v.max() - v.min())


def subgroup_sizes(sg):
    u, c = np.unique(sg, return_counts=True)
    return {"n_groups": int(len(u)), "min": int(c.min()), "median": float(np.median(c)),
            "max": int(c.max())}


# --------------------------- calibration methods ---------------------------
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pw_isotonic(cc, cy, tc):
    ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    ir.fit(np.asarray(cc, float), np.asarray(cy, float))
    return ir.predict(np.asarray(tc, float))


def temp_scaling(cc, cy, tc):
    cc = np.asarray(cc, float); cy = np.asarray(cy, float)
    best_T, best = 1.0, float("inf")
    for T in np.linspace(0.05, 10, 200):
        p = np.clip(1.0 / (1.0 + np.exp(-T * logit(cc))), 1e-6, 1 - 1e-6)
        nll = -np.mean(cy * np.log(p) + (1 - cy) * np.log(1 - p))
        if nll < best:
            best, best_T = nll, T
    return 1.0 / (1.0 + np.exp(-best_T * logit(tc)))


def group_iso(cc, cy, csg, tc, tsg):
    """Subgroup-wise isotonic calibration.

    NOTE (2026-10-01 fix). `out` must be initialised from the *uncalibrated*
    confidences rather than `np.empty`. Subgroup labels that occur ONLY in the
    test split are never visited by `for u in np.unique(csg)`, so an uninitialised
    buffer left those rows holding arbitrary memory -- observed as astronomically
    large CW-ECE for large families (e.g. emo x confbin10 x ambig, 100-170 groups).
    Small families used in the earlier phases were checked by audit_groupiso.py
    and had zero uncovered rows, so their published numbers are unaffected.
    """
    out = np.asarray(tc, float).copy()
    cc = np.asarray(cc, float); cy = np.asarray(cy, float)
    for u in np.unique(csg):
        m = csg == u; tm = tsg == u
        if tm.sum() == 0:
            continue
        if m.sum() == 0:
            continue
        if m.sum() < 2 or len(np.unique(cy[m])) < 2:
            out[tm] = float(cy[m].mean()); continue
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        ir.fit(cc[m], cy[m])
        out[tm] = ir.predict(tc[tm])
    return out


def group_iso_soft(cc, cys, csg, tc, tsg):
    """Same as group_iso but fitted to a continuous target (human agreement)."""
    out = np.asarray(tc, float).copy()   # see group_iso() note on initialisation
    cc = np.asarray(cc, float); cys = np.asarray(cys, float)
    for u in np.unique(csg):
        m = csg == u; tm = tsg == u
        if tm.sum() == 0:
            continue
        if m.sum() < 2:
            out[tm] = float(cys[m].mean()) if m.sum() else tc[tm]; continue
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        ir.fit(cc[m], cys[m])
        out[tm] = ir.predict(tc[tm])
    return out


def split_mask(n, seed=42, cal_ratio=CAL):
    rng = np.random.RandomState(seed)
    ids = np.arange(n); rng.shuffle(ids)
    k = int(round(n * cal_ratio))
    cal = np.zeros(n, dtype=bool); cal[ids[:k]] = True
    return cal


# --------------------------- generic MC baseline ---------------------------
HKRR_PARAMS = {"lambda": 0.1, "alpha": 0.05, "max_iter": 20,
               "randomized": True, "use_oracle": False}


def make_indicator(sg_cal, sg_test):
    uniq = np.unique(np.concatenate([np.asarray(sg_cal), np.asarray(sg_test)]))
    S_cal = (np.asarray(sg_cal)[:, None] == uniq[None, :]).astype(int)
    S_test = (np.asarray(sg_test)[:, None] == uniq[None, :]).astype(int)
    return S_cal, S_test


def hkrr_calibrate(conf_cal, corr_cal, sg_cal, conf_test, sg_test):
    """Generic multicalibration (HKRR, Devic 2024). Returns None on failure."""
    try:
        from multicalibration import MulticalibrationPredictor
        S_cal, S_test = make_indicator(sg_cal, sg_test)
        p = MulticalibrationPredictor("HKRR", verbose=False)
        p.fit(np.clip(np.asarray(conf_cal, float), 1e-3, 1 - 1e-3),
              np.asarray(corr_cal, int), S_cal, dict(HKRR_PARAMS))
        out = np.asarray(p.predict(
            np.clip(np.asarray(conf_test, float), 1e-3, 1 - 1e-3), S_test)).ravel()
        return np.clip(out, 0.0, 1.0)
    except Exception as e:
        print("    [HKRR fail]", type(e).__name__, str(e)[:140], flush=True)
        return None


# --------------------------- statistics ---------------------------
def nested_boot(pairs, B=B_BOOT, seed=0):
    """
    Cluster-aware bootstrap: draw a split, then resample its test set.

    pairs[s] = {"fa": f(idx)->float, "fb": f(idx)->float, "n": int, "full_a": float,
                "full_b": float}
    Captures BOTH split-to-split and within-split sampling variation.
    """
    rng = np.random.RandomState(seed)
    S = len(pairs)
    d = np.empty(B)
    for b in range(B):
        s = rng.randint(S)
        P = pairs[s]
        idx = rng.randint(0, P["n"], P["n"])
        d[b] = P["fa"](idx) - P["fb"](idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d > 0)))
    sd = float(d.std(ddof=1))
    per_split = np.array([P["full_a"] - P["full_b"] for P in pairs], float)
    return {
        "mean": float(d.mean()),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "p_two_sided": float(min(p, 1.0)),
        "boot_sd": sd,
        "z_effect": float(d.mean() / sd) if sd > 0 else None,   # standardized effect
        "per_split": [float(x) for x in per_split],
        "split_mean": float(per_split.mean()),
        "split_sd": float(per_split.std(ddof=1)) if S > 1 else None,
        "n_splits_same_dir": int(np.sum(per_split < 0)),
        "n_splits": int(S),
    }


def paired_dz(loss_a, loss_b):
    """Paired Cohen's d_z on per-sample losses (descriptive effect size)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    sd = d.std(ddof=1)
    return {"dz": float(d.mean() / sd) if sd > 0 else None,
            "mean_diff": float(d.mean()), "sd_diff": float(sd), "n": int(len(d))}


def bh_fdr(pvals):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


# ==========================================================================
def main():
    t_start = time.time()
    res = {"meta": {"date": "2026-10-01", "platform": "Jinan A100",
                    "purpose": "statistical rigor hardening (R1-R8)",
                    "seeds": SEEDS, "B_bootstrap": B_BOOT, "cal_ratio": CAL,
                    "min_subgroup_n_main": 1, "hkrr_params": HKRR_PARAMS},
           "R1_main": {}, "R4_sensitivity": {}, "R5_generic_mc": {},
           "R6_lambda_tradeoff": {}, "R8_fairness": {}, "fdr": {}}

    goe_side = {r["id"]: r for r in load_jsonl(os.path.join(CODE, "goe_annotator_sidecar.jsonl"))}
    raf_side = {r["id"]: r for r in load_jsonl(os.path.join(CODE, "raf_demographic_sidecar.jsonl"))}

    def prepare(ds, path, side_kind):
        rows = load_jsonl(path)
        ids = np.array([int(r.get("id", i)) for i, r in enumerate(rows)])
        conf = np.array([float(r["conf"]) for r in rows])
        pred = np.array([norm_label(r.get("pred_label", r.get("pred"))) for r in rows])
        corr = np.array([1 if norm_label(r.get("true")) == p else 0
                         for r, p in zip(rows, pred)])
        emo = np.array([norm_label(r.get("true")) for r in rows])
        side = goe_side if side_kind == "goe" else raf_side
        agree = np.array([float(side.get(i, {}).get("agree", np.nan)) for i in ids])
        gender = np.array([str(side.get(i, {}).get("gender", "NA")) for i in ids])
        age = np.array([str(side.get(i, {}).get("age_group", "NA")) for i in ids])
        # R8: RAF gender value 2 == "unsure" (was previously left unmapped)
        gender = np.where(gender == "2", "unsure", gender)
        return dict(ids=ids, conf=conf, corr=corr, emo=emo, pred=pred,
                    agree=agree, gender=gender, age=age, n=len(rows))

    def family_of(D, fam):
        emo, gen, age = D["emo"], D["gender"], D["age"]
        if fam == "pooled":
            return np.array(["ALL"] * D["n"])
        if fam == "emo":
            return emo
        if fam == "emo_x_gender":
            return np.array([f"{a}__{b}" for a, b in zip(emo, gen)])
        if fam == "emo_x_ambig":
            return None  # needs tier cuts -> built per split
        if fam == "emo_x_gender_x_age":
            return np.array([f"{a}__{b}__{c}" for a, b, c in zip(emo, gen, age)])
        raise ValueError(fam)

    samples_export = {}

    # ================= R1 / R2 / R3 : main contrasts =================
    print("=== R1-R3: main contrasts with nested bootstrap ===", flush=True)
    PRIMARY = []          # (set_key, contrast, p) for FDR
    for (model, ds, mod, path, side_kind) in RECORDS:
        if not os.path.exists(path):
            print("  [skip missing]", path, flush=True)
            continue
        key = f"{model}|{ds}"
        D = prepare(ds, path, side_kind)
        n = D["n"]
        conf, corr, emo, agree = D["conf"], D["corr"], D["emo"], D["agree"]
        fams = ["emo_x_ambig"] if mod == "text" else ["emo_x_gender", "emo_x_gender_x_age"]
        entry = {"modality": mod, "n": n, "comparisons": {}}
        exp = {"n": n, "emo": emo.tolist(), "gender": D["gender"].tolist(),
               "age": D["age"].tolist(), "conf": conf.tolist(),
               "corr": corr.astype(int).tolist(),
               "agree": [None if np.isnan(x) else float(x) for x in agree],
               "splits": {}}

        for fam in fams:
            for base_name in ["Pooled-Iso", "Uncalibrated"]:
                pairs = []
                loss_grp, loss_base = [], []
                fam_sizes = []
                for seed in SEEDS:
                    cal = split_mask(n, seed=seed)
                    tc, ty = conf[~cal], corr[~cal]
                    cc, cy = conf[cal], corr[cal]
                    if fam == "emo_x_ambig":
                        cuts = np.percentile(agree[cal][~np.isnan(agree[cal])], [33, 67])
                        tt = np.where(agree <= cuts[0], "lo",
                                      np.where(agree <= cuts[1], "mid", "hi"))
                        sg = np.array([f"{a}__{b}" for a, b in zip(emo, tt)])
                    else:
                        sg = family_of(D, fam)
                    sg_cal, sg_test = sg[cal], sg[~cal]
                    m_grp = group_iso(cc, cy, sg_cal, tc, sg_test)
                    if base_name == "Pooled-Iso":
                        m_base = pw_isotonic(cc, cy, tc)
                    else:
                        m_base = tc
                    fa = (lambda mm, tt, ss: (lambda idx: cw_ece(mm[idx], tt[idx], ss[idx])))(
                        m_grp, ty, sg_test)
                    fb = (lambda mm, tt, ss: (lambda idx: cw_ece(mm[idx], tt[idx], ss[idx])))(
                        m_base, ty, sg_test)
                    full = np.arange(len(ty))
                    pairs.append({"fa": fa, "fb": fb, "n": len(ty),
                                  "full_a": fa(full), "full_b": fb(full)})
                    loss_grp.append((m_grp - ty) ** 2)
                    loss_base.append((m_base - ty) ** 2)
                    fam_sizes.append(subgroup_sizes(sg_test))
                    if seed == 42:
                        # NOTE: merge instead of overwrite -- this block runs once
                        # per baseline, so a plain assignment would leave only the
                        # LAST baseline (Uncalibrated) in the slot and null out
                        # pred_group, which the reliability figure needs.
                        slot = exp["splits"].setdefault("42", {})
                        slot["test_idx"] = np.where(~cal)[0].tolist()
                        slot["sg_test"] = sg_test.tolist()
                        if base_name == "Pooled-Iso":
                            slot["pred_group"] = m_grp.tolist()
                            slot["pred_pooled"] = m_base.tolist()
                        else:
                            slot["pred_uncal"] = m_base.tolist()
                nb = nested_boot(pairs, B=B_BOOT, seed=0)
                dz = paired_dz(np.concatenate(loss_grp), np.concatenate(loss_base))
                pooled_base = float(np.mean([P["full_b"] for P in pairs]))
                ours_mean = pooled_base + nb["split_mean"]
                nb["relative_improvement_pct"] = (
                    float(100.0 * (pooled_base - ours_mean) / pooled_base)
                    if pooled_base else None)
                nb["paired_dz_on_brier"] = dz
                nb["eval_family_sizes"] = fam_sizes[0]
                entry["comparisons"][f"Group-Iso[{fam}] vs {base_name}"] = nb
                if base_name == "Pooled-Iso":
                    PRIMARY.append((key, f"Group-Iso[{fam}] vs Pooled-Iso",
                                    nb["p_two_sided"]))
                print(f"  {key:20s} {fam:19s} vs {base_name:13s} "
                      f"d={nb['mean']:+.4f} CI[{nb['ci_lo']:+.4f},{nb['ci_hi']:+.4f}] "
                      f"p={nb['p_two_sided']:.4f} z={nb['z_effect']:+.2f} "
                      f"dz={dz['dz']:+.2f} k/5={nb['n_splits_same_dir']} "
                      f"splitSD={nb['split_sd']:.4f}", flush=True)

        # ---- R4 sensitivity on the primary contrast of this set ----
        fam0 = fams[0]
        sens = {}
        for mm_n in [1, 5, 20]:
            ds_ = []
            for seed in SEEDS:
                cal = split_mask(n, seed=seed)
                tc, ty = conf[~cal], corr[~cal]
                if fam0 == "emo_x_ambig":
                    cuts = np.percentile(agree[cal][~np.isnan(agree[cal])], [33, 67])
                    tt = np.where(agree <= cuts[0], "lo",
                                  np.where(agree <= cuts[1], "mid", "hi"))
                    sg = np.array([f"{a}__{b}" for a, b in zip(emo, tt)])
                else:
                    sg = family_of(D, fam0)
                g = group_iso(conf[cal], corr[cal], sg[cal], tc, sg[~cal])
                p = pw_isotonic(conf[cal], corr[cal], tc)
                ds_.append(cw_ece(g, ty, sg[~cal], min_n=mm_n)
                           - cw_ece(p, ty, sg[~cal], min_n=mm_n))
            sens[f"min_subgroup_n={mm_n}"] = {"mean_delta": float(np.mean(ds_)),
                                              "per_split": [float(x) for x in ds_]}
        for bins in [10, 15, 20]:
            ds_ = []
            for seed in SEEDS:
                cal = split_mask(n, seed=seed)
                tc, ty = conf[~cal], corr[~cal]
                if fam0 == "emo_x_ambig":
                    cuts = np.percentile(agree[cal][~np.isnan(agree[cal])], [33, 67])
                    tt = np.where(agree <= cuts[0], "lo",
                                  np.where(agree <= cuts[1], "mid", "hi"))
                    sg = np.array([f"{a}__{b}" for a, b in zip(emo, tt)])
                else:
                    sg = family_of(D, fam0)
                g = group_iso(conf[cal], corr[cal], sg[cal], tc, sg[~cal])
                p = pw_isotonic(conf[cal], corr[cal], tc)
                ds_.append(cw_ece(g, ty, sg[~cal], bins=bins)
                           - cw_ece(p, ty, sg[~cal], bins=bins))
            sens[f"bins={bins}"] = {"mean_delta": float(np.mean(ds_)),
                                    "per_split": [float(x) for x in ds_]}
        res["R4_sensitivity"][key] = {"family": fam0, **sens}

        res["R1_main"][key] = entry
        samples_export[key] = exp

    # ================= R6 : H2 trade-off frontier =================
    print("=== R6: H2 lambda trade-off frontier ===", flush=True)
    LAMS = [round(x, 1) for x in np.arange(0, 1.0001, 0.1)]
    for (model, ds, mod, path, side_kind) in RECORDS:
        if side_kind != "goe" or not os.path.exists(path):
            continue
        key = f"{model}|{ds}"
        D = prepare(ds, path, side_kind)
        n = D["n"]; conf, corr, emo, agree = D["conf"], D["corr"], D["emo"], D["agree"]
        curves = {"pooled": {}, "grouped": {}}
        for lam in LAMS:
            y = (1 - lam) * corr + lam * agree
            ce_p, hu_p, ce_g, hu_g = [], [], [], []
            for seed in SEEDS:
                cal = split_mask(n, seed=seed)
                cc, cy, cemo = conf[cal], y[cal], emo[cal]
                tc, ty, tagree, temo = conf[~cal], corr[~cal], agree[~cal], emo[~cal]
                mp = pw_isotonic(cc, cy, tc)
                mg = group_iso_soft(cc, cy, cemo, tc, temo)
                ce_p.append(ece_overall(mp, ty))
                hu_p.append(soft_ece(mp, tagree))
                ce_g.append(ece_overall(mg, ty))
                hu_g.append(soft_ece(mg, tagree))
            curves["pooled"][f"{lam:.1f}"] = {"corr_ece": float(np.mean(ce_p)),
                                             "hu_ece": float(np.mean(hu_p))}
            curves["grouped"][f"{lam:.1f}"] = {"corr_ece": float(np.mean(ce_g)),
                                              "hu_ece": float(np.mean(hu_g))}
        res["R6_lambda_tradeoff"][key] = {"lambdas": LAMS, "curves": curves,
                                         "corr_target_note":
                                         "lambda=0 -> calibrated to correctness; "
                                         "lambda=1 -> calibrated to human agreement"}
        p0 = curves["grouped"]["0.0"]; p1 = curves["grouped"]["1.0"]
        print(f"  {key:20s} grouped: lam0 corrECE={p0['corr_ece']:.4f} huECE={p0['hu_ece']:.4f} "
              f"| lam1 corrECE={p1['corr_ece']:.4f} huECE={p1['hu_ece']:.4f}", flush=True)
        # Spearman on TEST only (fixes P2-9)
        rhos = []
        for seed in SEEDS:
            cal = split_mask(n, seed=seed)
            rhos.append(float(stats.spearmanr(conf[~cal], agree[~cal]).statistic))
        res["R6_lambda_tradeoff"][key]["spearman_conf_agree_test_only"] = {
            "mean": float(np.mean(rhos)), "per_split": rhos}
        # overconfidence on human-disagreed items, all splits (fixes P2-10)
        oc = []
        for seed in SEEDS:
            cal = split_mask(n, seed=seed)
            tc, ty, tagree = conf[~cal], corr[~cal], agree[~cal]
            m = pw_isotonic(conf[cal], corr[cal], tc)
            amb = tagree < 0.5
            if amb.sum() > 0:
                oc.append(float(np.mean(m[amb]) - np.mean(ty[amb])))
        res["R6_lambda_tradeoff"][key]["overconf_on_disagreed"] = {
            "mean": float(np.mean(oc)) if oc else None, "per_split": oc,
            "n_test_disagreed_avg": float(np.mean([ (agree[~split_mask(n, s)] < 0.5).sum()
                                                    for s in SEEDS]))}

    # ================= R5 : generic MC baseline =================
    print("=== R5: generic multicalibration (HKRR) baseline ===", flush=True)
    for (model, ds, mod, path, side_kind) in RECORDS:
        if not os.path.exists(path):
            continue
        key = f"{model}|{ds}"
        D = prepare(ds, path, side_kind)
        n = D["n"]; conf, corr, emo = D["conf"], D["corr"], D["emo"]
        fam_eval = "emo_x_ambig" if mod == "text" else "emo_x_gender"
        out = {}
        for seed in [42, 43]:
            cal = split_mask(n, seed=seed)
            tc, ty = conf[~cal], corr[~cal]
            if fam_eval == "emo_x_ambig":
                ag = D["agree"]
                cuts = np.percentile(ag[cal][~np.isnan(ag[cal])], [33, 67])
                tt = np.where(ag <= cuts[0], "lo", np.where(ag <= cuts[1], "mid", "hi"))
                sg_eval = np.array([f"{a}__{b}" for a, b in zip(emo, tt)])
            else:
                sg_eval = family_of(D, fam_eval)
            # generic family: emotion x 5 confidence bins
            cb = np.clip(np.digitize(conf, np.linspace(0, 1, 6)) - 1, 0, 4)
            sg_gen = np.array([f"{a}__c{b}" for a, b in zip(emo, cb)])
            m_our = group_iso(conf[cal], corr[cal], sg_eval[cal], tc, sg_eval[~cal])
            m_pool = pw_isotonic(conf[cal], corr[cal], tc)
            m_hk_gen = hkrr_calibrate(conf[cal], corr[cal], sg_gen[cal], tc, sg_gen[~cal])
            t0 = time.time()
            m_hk_our = hkrr_calibrate(conf[cal], corr[cal], sg_eval[cal], tc, sg_eval[~cal])
            dt = time.time() - t0
            e = lambda m: (None if m is None else float(cw_ece(m, ty, sg_eval[~cal])))
            out[f"seed{seed}"] = {
                "Group-Iso(C_emo)": e(m_our), "Pooled-Iso": e(m_pool),
                "HKRR(generic emo x confbin)": e(m_hk_gen),
                "HKRR(our C_emo family)": e(m_hk_our),
                "hkrr_sec": dt}
            print(f"  {key:20s} s{seed} our={out[f'seed{seed}']['Group-Iso(C_emo)']:.4f} "
                  f"pool={out[f'seed{seed}']['Pooled-Iso']:.4f} "
                  f"hkr_gen={out[f'seed{seed}']['HKRR(generic emo x confbin)']} "
                  f"hkr_our={out[f'seed{seed}']['HKRR(our C_emo family)']} "
                  f"({dt:.1f}s)", flush=True)
        res["R5_generic_mc"][key] = out

    # ================= R8 : fairness with correct unsure handling =================
    print("=== R8: fairness (unsure handled explicitly) ===", flush=True)
    for (model, ds, mod, path, side_kind) in RECORDS:
        if side_kind != "raf" or not os.path.exists(path):
            continue
        key = f"{model}|{ds}"
        D = prepare(ds, path, side_kind)
        n = D["n"]; conf, corr, emo, gen = D["conf"], D["corr"], D["emo"], D["gender"]
        out = {"dist": {g: int((gen == g).sum()) for g in np.unique(gen)}}
        for policy, mask_keep in [("exclude_unsure", gen != "unsure"),
                                  ("include_unsure", np.ones(n, bool))]:
            sg = np.array([f"{a}__{b}" for a, b in zip(emo, gen)])
            gp, gg, per_g = [], [], {}
            for seed in SEEDS:
                cal = split_mask(n, seed=seed) & mask_keep
                test = (~split_mask(n, seed=seed)) & mask_keep
                if cal.sum() < 20 or test.sum() < 20:
                    continue
                S = np.array([f"{a}__{b}" for a, b in zip(emo, gen)])
                mp = pw_isotonic(conf[cal], corr[cal], conf[test])
                mg = group_iso(conf[cal], corr[cal], S[cal], conf[test], S[test])
                gp.append(gap_pp(mp, corr[test], S[test]))
                gg.append(gap_pp(mg, corr[test], S[test]))
                for g in ["male", "female"]:
                    gm = gen[test] == g
                    if gm.sum() < 10:
                        continue
                    per_g.setdefault(g, []).append(
                        cw_ece(mp[gm], corr[test][gm], emo[test][gm]))
            d = np.array(gg) - np.array(gp)
            out[policy] = {
                "gap_pp_pooled": float(np.mean(gp)), "gap_pp_groupiso": float(np.mean(gg)),
                "delta_gap_per_split": [float(x) for x in d],
                "delta_gap_mean": float(d.mean()),
                "wilcoxon_p_less": float(stats.wilcoxon(d, alternative="less").pvalue)
                if len(d) > 0 else None,
                "per_gender_cw_ece": {g: float(np.mean(v)) for g, v in per_g.items()}}
            print(f"  {key} [{policy}] gap pooled={np.mean(gp):.4f} "
                  f"groupiso={np.mean(gg):.4f} d={d.mean():+.4f} "
                  f"p_less={out[policy]['wilcoxon_p_less']}", flush=True)
        res["R8_fairness"][key] = out

    # ================= FDR over the declared family =================
    if PRIMARY:
        pv = [p for _, _, p in PRIMARY]
        adj = bh_fdr(pv)
        res["fdr"] = {"family": "all primary contrasts (Group-Iso vs Pooled-Iso), "
                                "one per set x family",
                      "labels": [f"{k} :: {c}" for k, c, _ in PRIMARY],
                      "raw": pv, "bh_adj": [float(x) for x in adj]}

    out1 = os.path.join(CODE, "rigor_results_v2.json")
    with open(out1, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    out2 = os.path.join(CODE, "rigor_samples_v2.json")
    with open(out2, "w") as f:
        json.dump(samples_export, f, ensure_ascii=False)
    print(f"WROTE {out1}")
    print(f"WROTE {out2}")
    print(f"TOTAL {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
