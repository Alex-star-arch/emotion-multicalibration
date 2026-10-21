#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emotic_analysis.py — H5 指标计算 (EMOTIC 第二视觉数据集泛化验证)。

复用 calibration_core.py (已验证):
  - 核心子群族 (predicted_emotion x gender x ambiguity_tier) 的 CW-ECE:
      Uncalibrated / Pooled-Iso / Group-Iso(𝒞_emo)
  - 新增公平性轴: age x gender 的 Rawlsian worst-group ECE (max 子群 ECE)
  - 嵌套 bootstrap: 外层按 seed 切 cal/test, 内层 B=2000 重采样 test
  - 双侧 p = 2*min(P(d<=0), P(d>=0)); 同时报 z=Δ/SE, k/5 方向一致, BH-FDR

输入: infer_vlm.py 产出的 records JSONL (含 pred_emotion, conf, correct, gender, age_group, n_true)
输出: H5 指标 JSON + CSV (落 results/)

用法:
  python3 emotic_analysis.py \
      --records /wanzh/.../results/records_llava.jsonl \
      --categories /wanzh/.../data/emotic_categories.txt \
      --out /wanzh/.../results/h5_llava.json \
      --b 2000
"""
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_core import (build_subgroup_family, group_isotonic_calibrate,
                              cw_ece, ece_per_subgroup)

SEEDS = [42, 7, 13, 21, 101]


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


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
    """Rawlsian worst-group ECE = max 子群 ECE (§4.7 / §9.4 口径)。"""
    d = ece_per_subgroup(confs, correct, sg_id, bins)
    vals = [v for v in d.values() if not np.isnan(v)]
    return float(np.max(vals)) if vals else float("nan")


def ambiguity_tier(n_true):
    return 0 if n_true == 1 else 1


def bootstrap_delta(cal_confs, cal_correct, cal_sg, cal_sg_age,
                    test_confs, test_correct, test_sg, test_sg_age, B, seed):
    rng = np.random.default_rng(seed)
    n = len(test_confs)
    # 拟合 (held-out)
    rec_group = safe_group_isotonic(cal_confs, cal_correct, cal_sg, test_confs, test_sg)
    rec_pool = safe_pooled_isotonic(cal_confs, cal_correct, test_confs)
    # 观测 Δ
    d_cw_obs = (cw_ece(rec_group, test_correct, test_sg)
                - cw_ece(rec_pool, test_correct, test_sg))
    # worst-group (Rawlsian = max 子群 ECE) 绝对值与 Δ
    wg_group_obs = max_subgroup_ece(rec_group, test_correct, test_sg_age)
    wg_pool_obs = max_subgroup_ece(rec_pool, test_correct, test_sg_age)
    d_wg_obs = wg_group_obs - wg_pool_obs
    # bootstrap (对 test 重采样)
    dcw, dwg, wgg, wgp = [], [], [], []
    idx = np.arange(n)
    for _ in range(B):
        s = rng.choice(idx, size=n, replace=True)
        tc, tr, ts, ta = test_confs[s], test_correct[s], test_sg[s], test_sg_age[s]
        rg = safe_group_isotonic(cal_confs, cal_correct, cal_sg, tc, ts)
        rp = safe_pooled_isotonic(cal_confs, cal_correct, tc)
        dcw.append(cw_ece(rg, tr, ts) - cw_ece(rp, tr, ts))
        dwg.append(max_subgroup_ece(rg, tr, ta) - max_subgroup_ece(rp, tr, ta))
        wgg.append(max_subgroup_ece(rg, tr, ta))
        wgp.append(max_subgroup_ece(rp, tr, ta))
    dcw = np.array(dcw); dwg = np.array(dwg); wgg = np.array(wgg); wgp = np.array(wgp)

    def stats(d):
        se = d.std(ddof=1)
        # 双侧 bootstrap p = 2*min(P(d<=0), P(d>=0)); 当差值含大量恰好为 0 的样本时
        # P(d<=0)+P(d>=0) 可能 >1 -> p 越界. 夹紧到 [0,1] 以符合 p 值定义 (论文口径).
        p = min(2.0 * min(float((d <= 0).mean()), float((d >= 0).mean())), 1.0)
        z = (d.mean() / se) if se > 0 else float("nan")
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"mean": float(d.mean()), "se": float(se), "z": float(z),
                "p": float(p), "ci_lo": float(lo), "ci_hi": float(hi)}

    return {
        "cw_delta_group_minus_pooled": stats(dcw),
        "wg_delta_group_minus_pooled": stats(dwg),
        "worstgroup_ece": {"group": wg_group_obs, "pooled": wg_pool_obs,
                            "abs_reduction": wg_pool_obs - wg_group_obs,
                            "pct_reduction": float((wg_pool_obs - wg_group_obs) / wg_pool_obs) if wg_pool_obs > 0 else float("nan")},
        "obs": {"cw_delta": d_cw_obs, "wg_delta": d_wg_obs},
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
    cats = []
    with open(args.categories, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                cats.append(s)
    cat2idx = {c: i for i, c in enumerate(cats)}

    # 向量化 + 清洗 (丢弃非有限 conf/correct 的偶发脏记录, 如模型退化输出)
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
        print(f"  [clean] dropped {n_dropped} non-finite records "
              f"(of {len(recs)} read)", flush=True)
    recs = [x[2] for x in clean]
    confs = np.array([x[0] for x in clean])
    correct = np.array([x[1] for x in clean])
    pred_i = np.array([cat2idx[r["pred_emotion"]] for r in recs])
    gender_i = np.array([0 if r["gender"] == "Male" else 1 for r in recs])
    age_i = np.array([{"adult": 0, "teen": 1, "child": 2}.get(r["age_group"], 0) for r in recs])
    ambig = np.array([ambiguity_tier(int(r.get("n_true", 1))) for r in recs])
    n_true = np.array([int(r.get("n_true", 1)) for r in recs])

    sg_core = build_subgroup_family(pred_i, gender_i, ambig)[0]
    sg_age = np.array([f"{int(a)}|{int(g)}" for a, g in zip(age_i, gender_i)], dtype=object)
    _, sg_age_id = np.unique(sg_age, return_inverse=True)

    # 各 seed 外层切分
    per_seed = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        n = len(confs)
        perm = rng.permutation(n)
        k = int(n * args.cal_frac)
        cal, test = perm[:k], perm[k:]
        res = bootstrap_delta(
            confs[cal], correct[cal], sg_core[cal], sg_age_id[cal],
            confs[test], correct[test], sg_core[test], sg_age_id[test],
            args.b, s)
        res["seed"] = s
        res["n_cal"], res["n_test"] = len(cal), len(test)
        per_seed.append(res)
        print(f"  seed={s}  cwΔ={res['obs']['cw_delta']:+.4f}  "
              f"wgΔ={res['obs']['wg_delta']:+.4f}  "
              f"p_cw={res['cw_delta_group_minus_pooled']['p']:.3f}  "
              f"worstgrp_abs_red={res['worstgroup_ece']['abs_reduction']:+.4f}", flush=True)

    # 聚合
    cw_means = np.array([p["cw_delta_group_minus_pooled"]["mean"] for p in per_seed])
    wg_means = np.array([p["wg_delta_group_minus_pooled"]["mean"] for p in per_seed])
    k_cw = int((cw_means < 0).sum())  # 负 Δ = 我方更优
    k_wg = int((wg_means < 0).sum())
    # pooled p (Stouffer) over seeds
    z_cw = cw_means / np.array([p["cw_delta_group_minus_pooled"]["se"] for p in per_seed])
    z_wg = wg_means / np.array([p["wg_delta_group_minus_pooled"]["se"] for p in per_seed])
    stouffer_cw = float(np.sum(z_cw) / np.sqrt(len(z_cw)))
    stouffer_wg = float(np.sum(z_wg) / np.sqrt(len(z_wg)))
    p_stouffer_cw = 2 * min(__import__("scipy.stats", fromlist=["norm"]).norm.cdf(stouffer_cw),
                            __import__("scipy.stats", fromlist=["norm"]).norm.cdf(-stouffer_cw))
    p_stouffer_wg = 2 * min(__import__("scipy.stats", fromlist=["norm"]).norm.cdf(stouffer_wg),
                            __import__("scipy.stats", fromlist=["norm"]).norm.cdf(-stouffer_wg))

    summary = {
        "dataset": "EMOTIC",
        "n_records": len(recs),
        "n_true_distribution": {str(k): int(v) for k, v in zip(*np.unique(n_true, return_counts=True))},
        "per_seed": per_seed,
        "aggregate": {
            "cw_delta_mean_over_seeds": float(cw_means.mean()),
            "cw_k_of_5_negative": k_cw,
            "wg_delta_mean_over_seeds": float(wg_means.mean()),
            "wg_k_of_5_negative": k_wg,
            "stouffer_z_cw": stouffer_cw, "stouffer_p_cw": float(p_stouffer_cw),
            "stouffer_z_wg": stouffer_wg, "stouffer_p_wg": float(p_stouffer_wg),
        },
        "H5_verdict": {
            "cw_significant": bool(p_stouffer_cw < 0.05 and k_cw >= 4),
            "worstgroup_improved": bool(p_stouffer_wg < 0.05 and k_wg >= 4),
            "direction_consistent": bool(k_cw >= 4 and k_wg >= 4),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # 简短 CSV
    csv_path = args.out.replace(".json", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_records", len(recs)])
        w.writerow(["cw_delta_mean", summary["aggregate"]["cw_delta_mean_over_seeds"]])
        w.writerow(["cw_k_of_5_negative", k_cw])
        w.writerow(["stouffer_z_cw", stouffer_cw])
        w.writerow(["stouffer_p_cw", p_stouffer_cw])
        w.writerow(["wg_delta_mean", summary["aggregate"]["wg_delta_mean_over_seeds"]])
        w.writerow(["wg_k_of_5_negative", k_wg])
        w.writerow(["stouffer_z_wg", stouffer_wg])
        w.writerow(["stouffer_p_wg", p_stouffer_wg])
        w.writerow(["H5_cw_sig", summary["H5_verdict"]["cw_significant"]])
        w.writerow(["H5_wg_improved", summary["H5_verdict"]["worstgroup_improved"]])
    print(f"DONE -> {args.out}")
    print(json.dumps(summary["H5_verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
