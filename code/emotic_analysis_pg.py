#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emotic_analysis_pg.py — P-G 实验：显式纳入「年龄×性别」校准子群族。

设计 (配对、与 H5b 公平比较):
  校准子群族三候选:
    - pooled : 全局 isotonic (reference)
    - core   : emotion × gender × ambiguity_tier               (H5b 基线, 原 𝒞_emo 3 路)
    - pg     : emotion × gender × age_group × ambiguity_tier   (P-G 提案, 显式加 age 轴)
  worst-group ECE 度量轴 = age × gender   (与 H5b 一致)
  CW-ECE 度量轴        = emotion × gender × ambiguity_tier (论文主指标, 同族比避免退化)

统计: 嵌套 bootstrap B=2000, seeds={42,7,13,21,101}, 校准在 held-out cal 集拟合,
      test 集重采样得 Δ 分布; Stouffer 跨 seed 聚合, k/5 方向一致计数。

输出 JSON 含三种校准两两对比的 bootstrap 统计量 + 聚合 + PG_verdict。
"""
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_core import (build_subgroup_family, cw_ece, ece_per_subgroup)

SEEDS = [42, 7, 13, 21, 101]


def safe_group_isotonic(confs, correct, sg_id, test_confs, test_sg_id):
    """逐子群 isotonic; cal 子群样本 < min_n 时退化为 pooled isotonic。"""
    from sklearn.isotonic import IsotonicRegression
    min_n = 10
    pooled = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    pooled.fit(confs, correct)
    out = np.empty_like(test_confs, dtype=float)
    for u in np.unique(sg_id):
        m = sg_id == u
        tm = test_sg_id == u
        if tm.sum() == 0:
            continue
        if m.sum() >= min_n and len(np.unique(correct[m])) > 1:
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(confs[m], correct[m])
            out[tm] = ir.predict(test_confs[tm])
        else:
            out[tm] = pooled.predict(test_confs[tm])
    return out


def safe_pooled_isotonic(confs, correct, test_confs):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(confs, correct)
    return ir.predict(test_confs)


def max_subgroup_ece(confs, correct, sg_id, bins=10):
    """Rawlsian worst-group ECE = max 子群 ECE。"""
    d = ece_per_subgroup(confs, correct, sg_id, bins)
    vals = [v for v in d.values() if not np.isnan(v)]
    return float(np.max(vals)) if vals else float("nan")


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def bootstrap_delta(cal_confs, cal_correct, cal_sg_core, cal_sg_pg, cal_sg_age,
                    test_confs, test_correct, test_sg_core, test_sg_pg, test_sg_age,
                    B, seed):
    rng = np.random.default_rng(seed)
    n = len(test_confs)
    # held-out 拟合 (三种校准都基于原始 cal 集)
    rec_core = safe_group_isotonic(cal_confs, cal_correct, cal_sg_core, test_confs, test_sg_core)
    rec_pg   = safe_group_isotonic(cal_confs, cal_correct, cal_sg_pg,   test_confs, test_sg_pg)
    rec_pool = safe_pooled_isotonic(cal_confs, cal_correct, test_confs)

    # 观测值 (原始 test)
    def cw(rec):  return cw_ece(rec, test_correct, test_sg_core)
    def wg(rec):  return max_subgroup_ece(rec, test_correct, test_sg_age)
    obs = {
        "cw":  {"pooled": cw(rec_pool), "core": cw(rec_core), "pg": cw(rec_pg)},
        "wg":  {"pooled": wg(rec_pool), "core": wg(rec_core), "pg": wg(rec_pg)},
    }

    # bootstrap (重采样 test)
    dcw_cp, dcw_pp, dcw_pc = [], [], []
    dwg_cp, dwg_pp, dwg_pc = [], [], []
    idx = np.arange(n)
    for _ in range(B):
        s = rng.choice(idx, size=n, replace=True)
        tc, tr = test_confs[s], test_correct[s]
        rc = safe_group_isotonic(cal_confs, cal_correct, cal_sg_core, tc, test_sg_core[s])
        rp = safe_group_isotonic(cal_confs, cal_correct, cal_sg_pg,   tc, test_sg_pg[s])
        pl = safe_pooled_isotonic(cal_confs, cal_correct, tc)
        # CW-ECE 都在 core 族度量
        cw_c, cw_p, cw_pl = cw_ece(rc, tr, test_sg_core[s]), cw_ece(rp, tr, test_sg_core[s]), cw_ece(pl, tr, test_sg_core[s])
        dcw_cp.append(cw_c - cw_pl)   # core vs pooled
        dcw_pp.append(cw_p - cw_pl)   # pg vs pooled
        dcw_pc.append(cw_p - cw_c)    # pg vs core
        # worst-group 在 age×gender 度量
        wg_c, wg_p, wg_pl = max_subgroup_ece(rc, tr, test_sg_age[s]), max_subgroup_ece(rp, tr, test_sg_age[s]), max_subgroup_ece(pl, tr, test_sg_age[s])
        dwg_cp.append(wg_c - wg_pl)
        dwg_pp.append(wg_p - wg_pl)
        dwg_pc.append(wg_p - wg_c)

    def stats(d):
        d = np.array(d); se = d.std(ddof=1)
        p = min(2.0 * min(float((d <= 0).mean()), float((d >= 0).mean())), 1.0)
        z = (d.mean() / se) if se > 0 else float("nan")
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"mean": float(d.mean()), "se": float(se), "z": float(z),
                "p": float(p), "ci_lo": float(lo), "ci_hi": float(hi)}

    return {
        "obs": obs,
        "cw_core_minus_pooled": stats(dcw_cp),
        "cw_pg_minus_pooled":   stats(dcw_pp),
        "cw_pg_minus_core":     stats(dcw_pc),
        "wg_core_minus_pooled": stats(dwg_cp),
        "wg_pg_minus_pooled":   stats(dwg_pp),   # P-G 主检验: pg 校准是否改善 worst-group
        "wg_pg_minus_core":     stats(dwg_pc),   # P-G vs 基线校准
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--cal-frac", type=float, default=0.6)
    args = ap.parse_args()

    recs = load_records(args.records)
    cats = [s.strip() for s in open(args.categories, encoding="utf-8") if s.strip()]
    cat2idx = {c: i for i, c in enumerate(cats)}

    clean, n_dropped = [], 0
    for r in recs:
        try:
            c = float(r["conf"]); co = int(r["correct"])
        except (TypeError, ValueError):
            n_dropped += 1; continue
        if not (np.isfinite(c) and np.isfinite(co)):
            n_dropped += 1; continue
        clean.append((c, co, r))
    if n_dropped:
        print(f"  [clean] dropped {n_dropped} non-finite records (of {len(recs)})", flush=True)
    recs = [x[2] for x in clean]
    confs = np.array([x[0] for x in clean])
    correct = np.array([x[1] for x in clean])
    pred_i = np.array([cat2idx[r["pred_emotion"]] for r in recs])
    gender_i = np.array([0 if r["gender"] == "Male" else 1 for r in recs])
    age_i = np.array([{"adult": 0, "teen": 1, "child": 2}.get(r["age_group"], 0) for r in recs])
    ambig = np.array([0 if int(r.get("n_true", 1)) == 1 else 1 for r in recs])

    sg_core = build_subgroup_family(pred_i, gender_i, ambig)[0]
    # P-G 校准子群族: emotion × gender × age_group × ambiguity (4 路, 显式加 age)
    sg_pg_str = np.array([f"{int(pred_i[j])}|{int(gender_i[j])}|{int(age_i[j])}|{int(ambig[j])}"
                          for j in range(len(pred_i))], dtype=object)
    _, sg_pg_id = np.unique(sg_pg_str, return_inverse=True)
    sg_age = np.array([f"{int(a)}|{int(g)}" for a, g in zip(age_i, gender_i)], dtype=object)
    _, sg_age_id = np.unique(sg_age, return_inverse=True)

    per_seed = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        n = len(confs); perm = rng.permutation(n); k = int(n * args.cal_frac)
        cal, test = perm[:k], perm[k:]
        res = bootstrap_delta(
            confs[cal], correct[cal], sg_core[cal], sg_pg_id[cal], sg_age_id[cal],
            confs[test], correct[test], sg_core[test], sg_pg_id[test], sg_age_id[test],
            args.b, s)
        res["seed"], res["n_cal"], res["n_test"] = s, len(cal), len(test)
        per_seed.append(res)
        print(f"  seed={s}  cw_core-pool={res['cw_core_minus_pooled']['mean']:+.4f}(p={res['cw_core_minus_pooled']['p']:.2e})  "
              f"cw_pg-pool={res['cw_pg_minus_pooled']['mean']:+.4f}(p={res['cw_pg_minus_pooled']['p']:.2e})  "
              f"wg_core-pool={res['wg_core_minus_pooled']['mean']:+.4f}(p={res['wg_core_minus_pooled']['p']:.3f})  "
              f"wg_pg-pool={res['wg_pg_minus_pooled']['mean']:+.4f}(p={res['wg_pg_minus_pooled']['p']:.3f})  "
              f"wg_pg-core={res['wg_pg_minus_core']['mean']:+.4f}(p={res['wg_pg_minus_core']['p']:.3f})", flush=True)

    from scipy.stats import norm
    def stouffer(means, ses):
        z = np.array(means) / np.array(ses)
        zs = float(np.sum(z) / np.sqrt(len(z)))
        p = 2 * min(norm.cdf(zs), norm.cdf(-zs))
        return zs, float(p)
    cw_cp = [p["cw_core_minus_pooled"] for p in per_seed]
    cw_pp = [p["cw_pg_minus_pooled"] for p in per_seed]
    cw_pc = [p["cw_pg_minus_core"] for p in per_seed]
    wg_cp = [p["wg_core_minus_pooled"] for p in per_seed]
    wg_pp = [p["wg_pg_minus_pooled"] for p in per_seed]
    wg_pc = [p["wg_pg_minus_core"] for p in per_seed]
    z_cw_cp, p_cw_cp = stouffer([x["mean"] for x in cw_cp], [x["se"] for x in cw_cp])
    z_cw_pp, p_cw_pp = stouffer([x["mean"] for x in cw_pp], [x["se"] for x in cw_pp])
    z_cw_pc, p_cw_pc = stouffer([x["mean"] for x in cw_pc], [x["se"] for x in cw_pc])
    z_wg_cp, p_wg_cp = stouffer([x["mean"] for x in wg_cp], [x["se"] for x in wg_cp])
    z_wg_pp, p_wg_pp = stouffer([x["mean"] for x in wg_pp], [x["se"] for x in wg_pp])
    z_wg_pc, p_wg_pc = stouffer([x["mean"] for x in wg_pc], [x["se"] for x in wg_pc])
    k = lambda arr: int((np.array(arr) < 0).sum())

    summary = {
        "dataset": "EMOTIC-PG",
        "variant": "calibrate_on_emotion_x_gender_x_age_x_ambiguity",
        "worstgroup_axis": "age_x_gender",
        "cw_axis": "emotion_x_gender_x_ambiguity",
        "n_records": len(recs),
        "n_dropped": n_dropped,
        "b": args.b, "seeds": SEEDS,
        "per_seed": per_seed,
        "aggregate": {
            "cw_core_minus_pooled": {"stouffer_z": z_cw_cp, "stouffer_p": p_cw_cp, "k_of_5_neg": k([x["mean"] for x in cw_cp])},
            "cw_pg_minus_pooled":   {"stouffer_z": z_cw_pp, "stouffer_p": p_cw_pp, "k_of_5_neg": k([x["mean"] for x in cw_pp])},
            "cw_pg_minus_core":     {"stouffer_z": z_cw_pc, "stouffer_p": p_cw_pc, "k_of_5_neg": k([x["mean"] for x in cw_pc])},
            "wg_core_minus_pooled": {"stouffer_z": z_wg_cp, "stouffer_p": p_wg_cp, "k_of_5_neg": k([x["mean"] for x in wg_cp])},
            "wg_pg_minus_pooled":   {"stouffer_z": z_wg_pp, "stouffer_p": p_wg_pp, "k_of_5_neg": k([x["mean"] for x in wg_pp])},
            "wg_pg_minus_core":     {"stouffer_z": z_wg_pc, "stouffer_p": p_wg_pc, "k_of_5_neg": k([x["mean"] for x in wg_pc])},
        },
        "PG_verdict": {
            "H5a_reconfirm_cw_core_sig": bool(p_cw_cp < 0.05 and k([x["mean"] for x in cw_cp]) >= 4),
            "H5a_cw_pg_sig":            bool(p_cw_pp < 0.05 and k([x["mean"] for x in cw_pp]) >= 4),
            "cw_not_degraded_by_pg":    bool(p_cw_pc >= 0.05),   # pg 不应显著劣于 core (CW 主指标)
            "H5b_PG_vs_pooled_improved": bool(p_wg_pp < 0.05 and k([x["mean"] for x in wg_pp]) >= 4),
            "H5b_PG_vs_core_improved":   bool(p_wg_pc < 0.05 and k([x["mean"] for x in wg_pc]) >= 4),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    csv_path = args.out.replace(".json", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for key in ["cw_core_minus_pooled", "cw_pg_minus_pooled", "cw_pg_minus_core",
                    "wg_core_minus_pooled", "wg_pg_minus_pooled", "wg_pg_minus_core"]:
            a = summary["aggregate"][key]
            w.writerow([key + "_z", a["stouffer_z"]]); w.writerow([key + "_p", a["stouffer_p"]]); w.writerow([key + "_k5", a["k_of_5_neg"]])
        for key, val in summary["PG_verdict"].items():
            w.writerow([key, val])
    print(f"DONE -> {args.out}")
    print(json.dumps(summary["PG_verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
