#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formal_exp_large.py — 大规模数据集扩展实验 (H6 规模鲁棒 + H2 歧义层验证)
执行于济南超算 A100。消费 VLM verbalization 记录(jsonl), 计算:
  - 子群族 F0: emotion-only (true label)            —— 对齐 formal_exp_v1
  - 子群族 F1: emotion × ambiguity-tier(熵代理)     —— 完整 𝒞_emo 基础层
  - F2: emotion × ambiguity × gender(退化) = F1
  - 校准方法: Uncalibrated / Pooled-Iso / Group-Iso / Group-Platt
  - ΔCW-ECE_full 的逐样本 paired bootstrap(B=2000) 95%CI + 双侧 p (benjamini_hochberg)
  - 种子级 Wilcoxon(双侧, n=5 seeds) + 多重比较 BH-FDR(m=2 比较)
输出: formal_exp_large_results.json

注意: FERPlus/AffectNet 无人口学标注, gender 退化为常数; 歧义层用模型 full_conf 熵代理
      (诚实披露: 非真人多标注). FER2013 记录源于 TrustBench-Emo 项目, 此处作为方法输入数据复用.
"""
import json, os, sys, time, ast
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats

SEEDS = [42, 7, 13, 21, 101]
B_BOOT = 2000

# (model_label, dataset, modality, path, has_demo)
# 仅纳入真正大规模且新 verbalize 的数据集 (H6 规模鲁棒):
#  - FERPlus: 本论文新 verbalize (35K, 7类, 无人口学; 10标注者→真实 p_human 留待主观标定扩展)
#  - AffectNet: 本论文新下载+verbalize (287K, 7类去contempt, 无人口学)
#  FER2013 (TrustBench-Emo 既有记录) 仅 140 条/模型 且缺 full_conf → 非"大规模"、缺歧义层输入, 不纳入本扩展; 其 140 规模已在主稿分析.
DATASETS = [
    ("Qwen2.5-VL-3B", "FERPlus", "vision", "./mc_data/large_exp/ferplus/qwen25vl3b_records.jsonl", False),
    ("LLaVA-1.5-7B",  "FERPlus", "vision", "./mc_data/large_exp/ferplus/llava15_7b_records.jsonl", False),
    ("Qwen2.5-VL-3B", "AffectNet", "vision", "./mc_data/large_exp/affectnet/qwen25vl3b_records.jsonl", False),
    ("LLaVA-1.5-7B",  "AffectNet", "vision", "./mc_data/large_exp/affectnet/llava15_7b_records.jsonl", False),
]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def norm_emotion(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return str(v).strip().lower()


def parse_list(v):
    """records 中 true_emotions 常被序列化为字符串(e.g. "['happiness']")."""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return []
    return []


def parse_dict(v):
    """records 中 full_conf 常被序列化为字符串(e.g. "{'angry':0.5,...}")."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return {}
    return {}


def record_correct(r, pred_emo):
    """1 if pred_emo matches the record's ground-truth emotion(s), else 0.
    Handles both multi-label (true_emotions list) and single-label (true / emotion_label) schemas."""
    te = parse_list(r.get("true_emotions"))
    if te:
        return 1 if norm_emotion(pred_emo) in [norm_emotion(x) for x in te] else 0
    true = r.get("true", r.get("emotion_label", r.get("true_emotions_str", "na")))
    return 1 if norm_emotion(true) == norm_emotion(pred_emo) else 0


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def fc_entropy(r):
    """full_conf 熵(歧义层代理); 缺字段时回退均匀 7 类."""
    d = parse_dict(r.get("full_conf"))
    if not d:
        return entropy(np.array([1.0/7]*7, float))
    return entropy(np.array(list(d.values()), float))


def ambiguity_tier(full_conf, tertiles):
    """full_conf: dict 或 list; 返回 0/1/2 歧义层 (按训练集三分位)."""
    if isinstance(full_conf, dict):
        vals = np.array(list(full_conf.values()), float)
    else:
        vals = np.array(full_conf, float)
    vals = vals / (vals.sum() + 1e-12)
    h = entropy(vals)
    if h <= tertiles[0]:
        return 0
    if h <= tertiles[1]:
        return 1
    return 2


# ---------- metrics ----------
def ece_overall(conf, corr, bins=10):
    conf = np.asarray(conf, float); corr = np.asarray(corr, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    tot = 0.0
    for i in range(bins):
        m = idx == i
        if m.sum() > 0:
            tot += abs(conf[m].mean() - corr[m].mean()) * m.sum()
    return float(tot / len(conf)) if len(conf) else float("nan")


def per_sg_ece(conf, corr, sg, bins=10):
    d = {}
    for u in np.unique(sg):
        m = sg == u
        d[str(u)] = ece_overall(conf[m], corr[m], bins)
    return d


def cw_ece(conf, corr, sg, bins=10):
    d = per_sg_ece(conf, corr, sg, bins)
    counts = np.array([(sg == u).sum() for u in d])
    vals = np.array(list(d.values()))
    return float(np.sum(counts * vals) / counts.sum()) if counts.sum() else float("nan")


def gap_pp(conf, corr, sg, bins=10):
    d = per_sg_ece(conf, corr, sg, bins)
    vals = np.array(list(d.values()))
    return float(vals.max() - vals.min()) if len(vals) else float("nan")


# ---------- calibration ----------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pw_iso(cc, cy, tc):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1); ir.fit(cc, cy)
    return ir.predict(tc)


def pw_platt(cc, cy, tc):
    lr = LogisticRegression(C=1e6); lr.fit(logit(cc).reshape(-1, 1), cy)
    return lr.predict_proba(logit(tc).reshape(-1, 1))[:, 1]


def group_iso(cc, cy, csg, tc, tsg):
    out = np.empty_like(tc, dtype=float)
    for u in np.unique(csg):
        m = csg == u; tm = tsg == u
        if m.sum() == 0:
            out[tm] = tc[tm]; continue
        if m.sum() < 2 or len(np.unique(cy[m])) < 2:
            out[tm] = cy[m].mean(); continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1); ir.fit(cc[m], cy[m])
        out[tm] = ir.predict(tc[tm])
    return out


def group_platt(cc, cy, csg, tc, tsg):
    out = np.empty_like(tc, dtype=float)
    for u in np.unique(csg):
        m = csg == u; tm = tsg == u
        if m.sum() == 0:
            out[tm] = tc[tm]; continue
        if m.sum() < 2 or len(np.unique(cy[m])) < 2:
            out[tm] = cy[m].mean(); continue
        lr = LogisticRegression(C=1e6); lr.fit(logit(cc[m]).reshape(-1, 1), cy[m])
        out[tm] = lr.predict_proba(logit(tc[tm]).reshape(-1, 1))[:, 1]
    return out


def split_mask(n, seed=42, cal_ratio=0.7):
    rng = np.random.RandomState(seed)
    ids = np.arange(n); rng.shuffle(ids)
    k = int(round(n * cal_ratio))
    cal = np.zeros(n, dtype=bool); cal[ids[:k]] = True
    return cal


def benjamini_hochberg(pvals, alpha=0.05):
    """BH-FDR 校正. 返回 (reject_bool_list, p_adj_list) 与原顺序对齐."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); ranked = p[order]
    thr = (np.arange(1, m + 1) / m) * alpha
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    p_adj = np.empty(m); p_adj[order] = adj
    reject = np.zeros(m, dtype=bool)
    k = 0
    for i in range(m):
        if ranked[i] <= thr[i]:
            k = i + 1
    if k > 0:
        reject[order[:k]] = True
    return reject.tolist(), p_adj.tolist()


def paired_delta_bootstrap(A, B, ty, tsg, B_boot=B_BOOT, seed=0):
    """逐样本 paired bootstrap: Δ CW-ECE_full(A - B) 在测试集上的经验分布.
    返回 (deltas_array, mean, ci_lo, ci_hi, p_two_sided)."""
    n = len(ty); idx = np.arange(n); rng = np.random.RandomState(seed)
    deltas = np.empty(B_boot)
    for b in range(B_boot):
        s = rng.choice(idx, size=n, replace=True)
        deltas[b] = cw_ece(A[s], ty[s], tsg[s]) - cw_ece(B[s], ty[s], tsg[s])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return deltas, float(deltas.mean()), float(lo), float(hi), float(min(p, 1.0))


def build_sg_family(rows, fam, tertiles, cal_mask):
    """返回 (emotion_arr, ambig_arr, gender_arr) 同长度."""
    n = len(rows)
    emo = np.empty(n, dtype=object)
    amb = np.zeros(n, dtype=int)
    gen = np.array(["na"] * n, dtype=object)
    for i, r in enumerate(rows):
        te = parse_list(r.get("true_emotions"))
        if te:
            emo[i] = norm_emotion(te[0])
        else:
            emo[i] = norm_emotion(r.get("true", r.get("emotion_label", "na")))
        amb[i] = ambiguity_tier(parse_dict(r.get("full_conf", {})), tertiles)
        g = r.get("gender") or r.get("age_group")
        if g:
            gen[i] = str(g)
    return emo, amb, gen


def main():
    results = {"meta": {"date": "2026-10-13", "phase": "Large-scale extension (H6/H2)",
                        "seeds": SEEDS, "B_bootstrap": B_BOOT, "cal_ratio": 0.7,
                        "note": "gender degenerate on FERPlus/AffectNet; ambiguity=entropy proxy"},
               "per_set": {}, "bootstrap_primary": {}, "fdr": {}}
    for (model, ds, mod, path, has_demo) in DATASETS:
        if not os.path.exists(path):
            print(f"[skip] missing {path}"); continue
        rows = load(path)
        n = len(rows)
        # 预计算情绪/歧义层 (全量三分位)
        tert = np.percentile([fc_entropy(r) for r in rows], [33.3, 66.7])
        emo, amb, gen = build_sg_family(rows, "F1", tert, None)
        # conf/pred/corr (emotion-only 子群用 emo; 完整用 emo×amb)
        conf = np.array([float(r["conf"]) for r in rows])
        pred = np.array([norm_emotion(r.get("pred_emotion", r.get("pred", "na"))) for r in rows])
        corr = np.array([int(r["correct"]) if r.get("correct") not in (None, "") else record_correct(r, pred[i]) for i, r in enumerate(rows)])
        sg_emo = np.array([f"E|{e}" for e in emo])
        sg_full = np.array([f"E|{e}|A{a}|G{g}" for e, a, g in zip(emo, amb, gen)])
        per_seed = []
        test_data = []
        for seed in SEEDS:
            cal = split_mask(n, seed=seed, cal_ratio=0.7)
            cc, cy = conf[cal], corr[cal]
            tc, ty = conf[~cal], corr[~cal]
            tsg_emo = sg_emo[~cal]; csg_emo = sg_emo[cal]
            tsg_full = sg_full[~cal]; csg_full = sg_full[cal]
            def mk(base, sg_test):
                return {"CW_ECE_emo": cw_ece(base, ty, sg_test),
                        "CW_ECE_full": cw_ece(base, ty, sg_full[~cal]),
                        "gap_pp_full": gap_pp(base, ty, sg_full[~cal]),
                        "ECE": ece_overall(base, ty)}
            m_un = tc
            m_piso = pw_iso(cc, cy, tc)
            m_giso_emo = group_iso(cc, cy, csg_emo, tc, tsg_emo)
            m_giso_full = group_iso(cc, cy, csg_full, tc, tsg_full)
            m_gpl = group_platt(cc, cy, csg_full, tc, tsg_full)
            rec = {"seed": seed,
                   "Uncalibrated": mk(m_un, tsg_emo),
                   "Pooled-Iso": mk(m_piso, tsg_emo),
                   "Group-Iso-emo": mk(m_giso_emo, tsg_emo),
                   "Group-Iso-full": mk(m_giso_full, tsg_full),
                   "Group-Platt-full": mk(m_gpl, tsg_full),
                   "n_test": int((~cal).sum())}
            per_seed.append(rec)
            test_data.append({"Uncalibrated": m_un, "Pooled-Iso": m_piso,
                              "Group-Iso-emo": m_giso_emo, "Group-Iso-full": m_giso_full,
                              "Group-Platt-full": m_gpl, "ty": ty, "tsg": tsg_full})
        methods = ["Uncalibrated", "Pooled-Iso", "Group-Iso-emo", "Group-Iso-full", "Group-Platt-full"]
        mean_metrics = {m: {k: float(np.mean([s[m][k] for s in per_seed])) for k in per_seed[0][m]} for m in methods}
        std_metrics = {m: {k: float(np.std([s[m][k] for s in per_seed])) for k in per_seed[0][m]} for m in methods}
        # 统计: 逐样本 paired bootstrap(ΔCW-ECE_full) + 种子级 Wilcoxon(双侧) + BH-FDR
        comparisons = [("Group-Iso-full", "Uncalibrated"), ("Group-Iso-full", "Pooled-Iso")]
        boot = {}
        pvals, pkeys = [], []
        for a, b in comparisons:
            seed_deltas = []
            all_deltas = []
            for rnd, td in enumerate(test_data):
                A, Bm = td[a], td[b]; ty = td["ty"]; tsg = td["tsg"]
                seed_deltas.append(cw_ece(A, ty, tsg) - cw_ece(Bm, ty, tsg))
                d, _, _, _, _ = paired_delta_bootstrap(A, Bm, ty, tsg, seed=rnd)
                all_deltas.append(d)
            pooled = np.concatenate(all_deltas)
            pooled_mean = float(pooled.mean())
            ci_lo, ci_hi = [float(x) for x in np.percentile(pooled, [2.5, 97.5])]
            pool_p2 = float(2.0 * min(np.mean(pooled <= 0), np.mean(pooled >= 0)))
            try:
                w = stats.wilcoxon(seed_deltas, alternative="two-sided")
                w_p = float(w.pvalue)
            except Exception:
                w_p = None
            boot[f"{a}_vs_{b}"] = {"mean_seed_delta": float(np.mean(seed_deltas)),
                                   "bootstrap_mean_delta": pooled_mean,
                                   "ci_lo": ci_lo, "ci_hi": ci_hi,
                                   "per_sample_bootstrap_p_two_sided": pool_p2,
                                   "seed_wilcoxon_p_two_sided": w_p,
                                   "n_seeds": len(seed_deltas)}
            pvals.append(w_p if w_p is not None else 1.0)
            pkeys.append(f"{a}_vs_{b}")
        if pvals:
            rej, p_adj = benjamini_hochberg(pvals)
            for k, ra, pa in zip(pkeys, rej, p_adj):
                boot[k]["bh_fdr_p_adj"] = pa
                boot[k]["bh_fdr_reject"] = ra
        results["per_set"][f"{model}|{ds}"] = {"n": n, "mean_metrics": mean_metrics,
                                               "std_metrics": std_metrics,
                                               "delta_CW_ECE_full_vs": boot}
        print(f"[done] {model}|{ds} n={n} CW-ECE full: Group-Iso={mean_metrics['Group-Iso-full']['CW_ECE_full']:.3f} "
              f"vs Pooled={mean_metrics['Pooled-Iso']['CW_ECE_full']:.3f} vs Uncal={mean_metrics['Uncalibrated']['CW_ECE_full']:.3f}")
    out = "./mc_data/large_exp/formal_exp_large_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nWROTE", out)
    print("DONE")


if __name__ == "__main__":
    main()
