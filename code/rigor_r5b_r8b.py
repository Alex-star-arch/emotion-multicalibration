#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R5' : controllable generic-multicalibration comparison (replaces the HKRR
      baseline, which was shown to degenerate -- see diag output: only 10
      distinct outputs, invariant to max_iter / use_oracle / subgroup family,
      TEST-ECE 0.0801 vs uncalibrated 0.0800).

      Design: hold the EVALUATION family fixed (= our C_emo family) and vary
      the DEFINITION family only, so the comparison isolates the question
      "how much subgroup knowledge does the calibrator need?":

        Pooled-Iso                     : no family at all
        Group-Iso(emotion)             : 7 groups       (coarse semantic)
        Group-Iso(C_emo = emo x ambig) : ~21 groups     (ours, compact+informed)
        GenericMC(emo x confbin10)     : ~70 groups     (semantic, blind to ambiguity)
        GenericMC(confbin10 x ambig)   : ~30 groups     (no semantic labels)
        GenericMC(emo x confbin10 x ambig): ~210 groups (brute force)
        HKRR(official pkg)             : reported for transparency

R8b : nested bootstrap for the fairness contrast (fixes the closure bug and
      the under-powered seed-level Wilcoxon in the earlier run).
"""
import json
import os

import numpy as np

from rigor_exp_v2 import (RECORDS, SEEDS, CODE, norm_label, load_jsonl,
                          split_mask, cw_ece, ece_overall, pw_isotonic, group_iso)

B = 2000


# ------------------------------------------------------------------ helpers
def load_set(path, side_kind, sides):
    rows = load_jsonl(path)
    ids = np.array([int(r.get("id", i)) for i, r in enumerate(rows)])
    conf = np.array([float(r["conf"]) for r in rows])
    pred = np.array([norm_label(r.get("pred_label", r.get("pred"))) for r in rows])
    corr = np.array([1 if norm_label(r.get("true")) == p else 0 for r, p in zip(rows, pred)])
    emo = np.array([norm_label(r.get("true")) for r in rows])
    side = sides[side_kind]
    agree = np.array([float(side.get(i, {}).get("agree", np.nan)) for i in ids])
    gender = np.array([str(side.get(i, {}).get("gender", "NA")) for i in ids])
    gender = np.where(gender == "2", "unsure", gender)
    return dict(conf=conf, corr=corr, emo=emo, agree=agree, gender=gender, n=len(rows))


def gap_pp_local(pred, corr, sg, bins=10):
    d = {}
    edges = np.linspace(0, 1, bins + 1)
    for u in np.unique(sg):
        m = sg == u
        if m.sum() == 0:
            continue
        bi = np.clip(np.digitize(pred[m], edges) - 1, 0, bins - 1)
        num = den = 0.0
        for i in range(bins):
            mm = bi == i
            if mm.sum() > 0:
                num += abs(pred[m][mm].mean() - corr[m][mm].mean()) * mm.sum()
                den += mm.sum()
        if den > 0:
            d[str(u)] = num / den
    if len(d) < 2:
        return np.nan
    v = np.array(list(d.values()))
    return float(v.max() - v.min())


def nested_boot(pairs, B=B, seed=0):
    """Cluster-aware bootstrap with NaN guard (draw a split, resample inside)."""
    rng = np.random.RandomState(seed)
    S = len(pairs)
    d = []
    while len(d) < B:
        s = rng.randint(S)
        P = pairs[s]
        idx = rng.randint(0, P["n"], P["n"])
        v = P["fa"](idx) - P["fb"](idx)
        if np.isfinite(v):
            d.append(v)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d > 0)))
    per_split = np.array([P["full_a"] - P["full_b"] for P in pairs], float)
    return {"mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": float(min(p, 1.0)),
            "z_effect": float(d.mean() / d.std(ddof=1)),
            "per_split": [float(x) for x in per_split],
            "split_sd": float(per_split.std(ddof=1)) if S > 1 else None,
            "n_splits_same_dir": int(np.sum(per_split < 0)), "n_splits": int(S)}


# ------------------------------------------------------------------ R5'
def run_r5(sides, out):
    print("################ R5': definition-family ladder ################", flush=True)
    for (model, ds, mod, path, side_kind) in RECORDS:
        if not os.path.exists(path):
            continue
        key = f"{model}|{ds}"
        D = load_set(path, side_kind, sides)
        n = D["n"]; conf, corr, emo, agree, gen = (
            D["conf"], D["corr"], D["emo"], D["agree"], D["gender"])
        cb10 = np.clip(np.digitize(conf, np.linspace(0, 1, 11)) - 1, 0, 9)

        if mod == "text":
            EVAL = "emo_x_ambig"
        else:
            EVAL = "emo_x_gender"

        def eval_family(cal):
            if EVAL == "emo_x_ambig":
                cuts = np.percentile(agree[cal][~np.isnan(agree[cal])], [33, 67])
                tt = np.where(agree <= cuts[0], "lo", np.where(agree <= cuts[1], "mid", "hi"))
                return np.array([f"{a}__{b}" for a, b in zip(emo, tt)])
            return np.array([f"{a}__{b}" for a, b in zip(emo, gen)])

        def def_family(name, cal):
            if name == "Pooled-Iso":
                return np.array(["ALL"] * n)
            if name == "Group-Iso(emotion)":
                return emo
            if name == "Group-Iso[C_emo]":
                return eval_family(cal)
            if name == "GenericMC(emo x confbin10)":
                return np.array([f"{a}__c{b}" for a, b in zip(emo, cb10)])
            if name == "GenericMC(confbin10 x ambig)":
                if mod == "text":
                    cuts = np.percentile(agree[cal][~np.isnan(agree[cal])], [33, 67])
                    tt = np.where(agree <= cuts[0], "lo", np.where(agree <= cuts[1], "mid", "hi"))
                    return np.array([f"c{b}__{t}" for b, t in zip(cb10, tt)])
                return np.array([f"c{b}__{g}" for b, g in zip(cb10, gen)])
            if name == "GenericMC(emo x confbin10 x ambig)":
                if mod == "text":
                    cuts = np.percentile(agree[cal][~np.isnan(agree[cal])], [33, 67])
                    tt = np.where(agree <= cuts[0], "lo", np.where(agree <= cuts[1], "mid", "hi"))
                    return np.array([f"{a}__c{b}__{t}" for a, b, t in zip(emo, cb10, tt)])
                return np.array([f"{a}__c{b}__{g}" for a, b, g in zip(emo, cb10, gen)])
            raise ValueError(name)

        METHODS = ["Pooled-Iso", "Group-Iso(emotion)", "Group-Iso[C_emo]",
                   "GenericMC(emo x confbin10)", "GenericMC(confbin10 x ambig)",
                   "GenericMC(emo x confbin10 x ambig)"]
        res = {}
        for mname in METHODS:
            cw, gp, ng = [], [], []
            for s in SEEDS:
                cal = split_mask(n, seed=s)
                ev = eval_family(cal)
                dfam = def_family(mname, cal)
                if mname == "Pooled-Iso":
                    p = pw_isotonic(conf[cal], corr[cal], conf[~cal])
                else:
                    p = group_iso(conf[cal], corr[cal], dfam[cal], conf[~cal], dfam[~cal])
                cw.append(cw_ece(p, corr[~cal], ev[~cal]))
                gp.append(gap_pp_local(p, corr[~cal], ev[~cal]))
                ng.append(len(np.unique(dfam[cal])))
            res[mname] = {"cw_ece_mean": float(np.mean(cw)),
                          "cw_ece_sd": float(np.std(cw, ddof=1)),
                          "gap_pp_mean": float(np.mean(gp)),
                          "n_def_groups_cal": float(np.mean(ng)),
                          "cal_samples_per_group": float(980.0 / np.mean(ng))}
            print(f"  {key:20s} {mname:34s} CW-ECE={res[mname]['cw_ece_mean']:.4f}"
                  f"±{res[mname]['cw_ece_sd']:.4f}  gap={res[mname]['gap_pp_mean']:.4f}"
                  f"  #grp={res[mname]['n_def_groups_cal']:.0f}"
                  f"  n/grp={res[mname]['cal_samples_per_group']:.1f}", flush=True)

        # HKRR transparency note
        try:
            from multicalibration import MulticalibrationPredictor
            cal = split_mask(n, seed=42)
            uniq = np.unique(np.concatenate([emo[cal], emo[~cal]]))
            S_c = (emo[cal][:, None] == uniq[None, :]).astype(int)
            S_t = (emo[~cal][:, None] == uniq[None, :]).astype(int)
            p = MulticalibrationPredictor("HKRR", verbose=False)
            p.fit(np.clip(conf[cal], 1e-3, 1 - 1e-3), corr[cal], S_c,
                  {"lambda": 0.1, "alpha": 0.05, "max_iter": 20,
                   "randomized": True, "use_oracle": True})
            pt = np.asarray(p.predict(np.clip(conf[~cal], 1e-3, 1 - 1e-3), S_t)).ravel()
            res["HKRR(official pkg)"] = {
                "cw_ece_mean": float(cw_ece(pt, corr[~cal], eval_family(cal)[~cal])),
                "n_unique_outputs": int(len(np.unique(np.round(pt, 4)))),
                "test_ece": float(ece_overall(pt, corr[~cal])),
                "uncalibrated_test_ece": float(ece_overall(conf[~cal], corr[~cal])),
                "note": "degenerate: invariant to max_iter/use_oracle/subgroup family; "
                        "outputs only 10 distinct values (one per confidence bin)"}
            print(f"  {key:20s} HKRR(official pkg)               CW-ECE={res['HKRR(official pkg)']['cw_ece_mean']:.4f}"
                  f"  uniq_out={res['HKRR(official pkg)']['n_unique_outputs']}"
                  f"  [DEGENERATE]", flush=True)
        except Exception as e:
            print("   HKRR skipped:", type(e).__name__, str(e)[:90], flush=True)

        out[key] = {"modality": mod, "n": n, "eval_family": EVAL, "methods": res}


# ------------------------------------------------------------------ R8b
def run_r8b(sides, out):
    print("\n################ R8b: fairness nested bootstrap ################", flush=True)
    for (model, ds, mod, path, side_kind) in RECORDS:
        if side_kind != "raf" or not os.path.exists(path):
            continue
        key = f"{model}|{ds}"
        D = load_set(path, side_kind, sides)
        n = D["n"]; conf, corr, emo, gen = D["conf"], D["corr"], D["emo"], D["gender"]
        sg = np.array([f"{a}__{b}" for a, b in zip(emo, gen)])
        res = {"dist": {g: int((gen == g).sum()) for g in np.unique(gen)}}
        for policy, keep in [("exclude_unsure", gen != "unsure"),
                             ("include_unsure", np.ones(n, bool))]:
            pairs, lg, lp = [], [], []
            for s in SEEDS:
                m = split_mask(n, seed=s)
                calm, testm = m & keep, (~m) & keep
                if calm.sum() < 20 or testm.sum() < 20:
                    continue
                idx = np.where(testm)[0]
                corr_t, sg_t = corr[idx], sg[idx]
                mp = pw_isotonic(conf[calm], corr[calm], conf[idx])
                mg = group_iso(conf[calm], corr[calm], sg[calm], conf[idx], sg[idx])

                def mk(pred, corr_t=corr_t, sg_t=sg_t):
                    return lambda ii: gap_pp_local(pred[ii], corr_t[ii], sg_t[ii])

                pairs.append({"fa": mk(mg), "fb": mk(mp), "n": len(idx),
                              "full_a": gap_pp_local(mg, corr_t, sg_t),
                              "full_b": gap_pp_local(mp, corr_t, sg_t)})
                lg.append((mg - corr_t) ** 2)
                lp.append((mp - corr_t) ** 2)
            nb = nested_boot(pairs, B=B, seed=0)
            da = np.concatenate(lg) - np.concatenate(lp)
            nb["paired_dz_on_brier"] = float(da.mean() / da.std(ddof=1)) if da.std(ddof=1) > 0 else None
            nb["gap_pp_pooled"] = float(np.mean([P["full_b"] for P in pairs]))
            nb["gap_pp_groupiso"] = float(np.mean([P["full_a"] for P in pairs]))
            res[policy] = nb
            print(f"  {key} [{policy:15s}] gap pooled={nb['gap_pp_pooled']:.4f} "
                  f"groupiso={nb['gap_pp_groupiso']:.4f} d={nb['mean']:+.4f} "
                  f"CI[{nb['ci_lo']:+.4f},{nb['ci_hi']:+.4f}] p={nb['p_two_sided']:.4f} "
                  f"dz={nb['paired_dz_on_brier']:+.2f} k/5={nb['n_splits_same_dir']}", flush=True)
        out[key] = res


def main():
    goe = {r["id"]: r for r in load_jsonl(os.path.join(CODE, "goe_annotator_sidecar.jsonl"))}
    raf = {r["id"]: r for r in load_jsonl(os.path.join(CODE, "raf_demographic_sidecar.jsonl"))}
    sides = {"goe": goe, "raf": raf}

    r5, r8b = {}, {}
    run_r5(sides, r5)
    run_r8b(sides, r8b)

    with open(os.path.join(CODE, "rigor_r5b_family_ladder.json"), "w") as f:
        json.dump(r5, f, indent=1, ensure_ascii=False)
    with open(os.path.join(CODE, "rigor_r8b_fairness.json"), "w") as f:
        json.dump(r8b, f, indent=1, ensure_ascii=False)
    print("WROTE rigor_r5b_family_ladder.json, rigor_r8b_fairness.json")


if __name__ == "__main__":
    main()
