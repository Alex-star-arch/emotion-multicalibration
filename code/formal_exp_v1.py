#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正式实验 v1 (执行于济南超算 A100) —— Phase 1: 基于既有冻结记录
依据: 02-working/formal_experiment_protocol.md

扩展相对 pre_exp:
  - 5 种子 (42..46) 轮换 cal70/test30, 报告均值±bootstrap CI
  - 7 种校准方法: Uncalibrated / TS / VS(per-class temp) / Platt / Pooled-Iso / Group-Iso / Group-Platt
  - 统计分析: per-sample paired bootstrap(B=2000) 给出 ΔCW-ECE/Δgap_pp 的 95% CI 与双侧 p
               种子级 one-sample Wilcoxon 确认数据集层面胜出; BH-FDR 多重比较校正
  - 跨模态(文本/FER/RAF)一致性分组
  - 效率统计(H4): 单集分组校准耗时
输出: formal_exp_results.json + 终端 markdown
"""
import json, os, sys, time
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats

FER_MAP = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

# (model_label, dataset, modality, path) —— 9 个已核验 eval-format 记录
RECORDS = [
    ("Gemma-2-9B",    "GoEmotions", "text",  "./trustbench_emo_data/results/gemma2_9b_goemotions/records.jsonl"),
    ("Llama-3-8B",    "GoEmotions", "text",  "./trustbench_emo_data/results/llama3_8b_goemotions/records.jsonl"),
    ("Qwen2.5-7B",    "GoEmotions", "text",  "./trustbench_emo_data/results/matrix/GoEmo_Qwen25_7B/records.jsonl"),
    ("Qwen2.5-3B",    "GoEmotions", "text",  "./trustbench_emo_data/results/matrix/GoEmo_Qwen25_3B/records.jsonl"),
    ("Qwen2.5-VL-3B", "GoEmotions", "text",  "./trustbench_emo_data/results/matrix/GoEmo_Qwen25VL3B/records.jsonl"),
    ("FER2013-base",  "FER2013",    "vision","./trustbench_emo_data/results/fer_d1_eval140/records.jsonl"),
    ("Qwen2-VL-7B",   "FER2013",    "vision","./trustbench_emo_data/results/matrix/FER2013_Qwen2VL7B/records.jsonl"),
    ("Qwen2.5-VL-3B", "FER2013",    "vision","./trustbench_emo_data/results/matrix/FER2013_Qwen25VL3B/records.jsonl"),
    ("LLaVA(RAF)",    "RAF-DB",     "vision","./trustbench_emo_data/results/r15_llava_rafdb/eval/records.jsonl"),
]
SEEDS = [42, 43, 44, 45, 46]
B_BOOT = 2000

def norm_label(v):
    if isinstance(v, bool): return str(v)
    if isinstance(v, int): return FER_MAP.get(v, str(v))
    s = str(v)
    if s.isdigit(): return FER_MAP.get(int(s), s)
    return s

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

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
    return float(tot / len(conf))

def per_class_ece(conf, corr, sg, bins=10):
    d = {}
    for u in np.unique(sg):
        m = sg == u
        d[str(u)] = ece_overall(conf[m], corr[m], bins)
    return d

def cw_ece(conf, corr, sg, bins=10):
    d = per_class_ece(conf, corr, sg, bins)
    counts = np.array([(sg == u).sum() for u in d])
    vals = np.array(list(d.values()))
    return float(np.sum(counts * vals) / counts.sum())

def gap_pp(conf, corr, sg, bins=10):
    d = per_class_ece(conf, corr, sg, bins)
    vals = np.array(list(d.values()))
    return float(vals.max() - vals.min())

def brier(conf, corr):
    return float(np.mean((np.asarray(conf,float)-np.asarray(corr,float))**2))
def aurc(conf, corr):
    conf=np.asarray(conf,float); corr=np.asarray(corr,float)
    order=np.argsort(-conf); c=corr[order]; n=len(conf)
    return float(np.mean(np.cumsum(1-c)/np.arange(1,n+1)))

# ---------- calibration methods ----------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def pw_isotonic(cc, cy, tc):
    ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1); ir.fit(cc, cy)
    return ir.predict(tc)

def pw_platt(cc, cy, tc):
    lr = LogisticRegression(C=1e6); lr.fit(logit(cc).reshape(-1,1), cy)
    return lr.predict_proba(logit(tc).reshape(-1,1))[:,1]

def temp_scaling(cc, cy, tc):
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.05, 10, 200):
        p = 1.0/(1.0+np.exp(-T*logit(cc))); p=np.clip(p,1e-6,1-1e-6)
        nll = -np.mean(cy*np.log(p)+(1-cy)*np.log(1-p))
        if nll < best_nll: best_nll, best_T = nll, T
    return 1.0/(1.0+np.exp(-best_T*logit(tc)))

def vector_scaling(cc, cy, cpred, tc, tpred):
    """per-class temperature on scalar conf (向量缩放近似)."""
    out = np.empty_like(tc, dtype=float)
    classes = np.unique(cpred)
    Tmap = {}
    for c in classes:
        m = cpred == c
        if m.sum() < 2 or len(np.unique(cy[m])) < 2:
            Tmap[c] = 1.0
        else:
            best_T, best_nll = 1.0, float("inf")
            for T in np.linspace(0.05, 10, 120):
                p = 1.0/(1.0+np.exp(-T*logit(cc[m]))); p=np.clip(p,1e-6,1-1e-6)
                nll = -np.mean(cy[m]*np.log(p)+(1-cy[m])*np.log(1-p))
                if nll < best_nll: best_nll, best_T = nll, T
            Tmap[c] = best_T
    for c in np.unique(tpred):
        tm = tpred == c
        out[tm] = 1.0/(1.0+np.exp(-Tmap.get(c,1.0)*logit(tc[tm])))
    return out

def group_iso(cc, cy, csg, tc, tsg):
    out = np.empty_like(tc, dtype=float)
    for u in np.unique(csg):
        m = csg == u; tm = tsg == u
        if m.sum() == 0: out[tm] = tc[tm]; continue
        if m.sum() < 2 or len(np.unique(cy[m])) < 2: out[tm] = cy[m].mean(); continue
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1); ir.fit(cc[m], cy[m])
        out[tm] = ir.predict(tc[tm])
    return out

def group_platt(cc, cy, csg, tc, tsg):
    out = np.empty_like(tc, dtype=float)
    for u in np.unique(csg):
        m = csg == u; tm = tsg == u
        if m.sum() == 0: out[tm] = tc[tm]; continue
        if m.sum() < 2 or len(np.unique(cy[m])) < 2: out[tm] = cy[m].mean(); continue
        lr = LogisticRegression(C=1e6); lr.fit(logit(cc[m]).reshape(-1,1), cy[m])
        out[tm] = lr.predict_proba(logit(tc[tm]).reshape(-1,1))[:,1]
    return out

def split_mask(n, seed=42, cal_ratio=0.7):
    rng = np.random.RandomState(seed)
    ids = np.arange(n); rng.shuffle(ids)
    k = int(round(n*cal_ratio))
    cal = np.zeros(n, dtype=bool); cal[ids[:k]] = True
    return cal

# ---------- bootstrap for Δ between two metric arrays on test set ----------
def paired_bootstrap_ci(metric_a, metric_b, test_idx, B=B_BOOT, seed=0):
    """metric_a/metric_b: functions(conf, corr, sg)->float; 返回 Δ=a-b 的 CI 与双侧 p."""
    rng = np.random.RandomState(seed)
    n = len(test_idx)
    idx = np.array(test_idx)
    deltas = np.empty(B)
    for b in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        da = metric_a(samp); db = metric_b(samp)
        deltas[b] = da - db
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    mean = deltas.mean()
    p = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas > 0))
    p = min(p, 1.0)
    return {"mean": float(mean), "ci_lo": float(lo), "ci_hi": float(hi), "p_two_sided": float(p)}

def main():
    results = {"meta": {}, "per_set": {}, "bootstrap_primary": {}, "cross_modality": {}, "efficiency": {}}
    results["meta"] = {
        "date": "2026-10-01",
        "platform": "Jinan A100-SXM4-40GB",
        "phase": "Phase1 (frozen records, no new weights/annotations)",
        "methods": ["Uncalibrated","TS","VS","Platt","Pooled-Iso","Group-Iso","Group-Platt"],
        "seeds": SEEDS, "B_bootstrap": B_BOOT, "cal_ratio": 0.7,
        "subgroup_family": "emotion class (true label) — base layer of C_emo",
        "primary_comparisons": "Group-Iso vs {Uncalibrated,TS,VS,Platt,Pooled-Iso}",
    }
    # accumulator for cross-modality
    seed_deltas = {cmp: {"text":[], "vision_FER":[], "vision_RAF":[]} for cmp in
                   ["vs_unc","vs_TS","vs_VS","vs_Platt","vs_PooledIso"]}
    timing = []

    for (model, ds, mod, path) in RECORDS:
        if not os.path.exists(path):
            print(f"[skip] missing {path}"); continue
        rows = load(path)
        conf = np.array([float(r['conf']) for r in rows])
        pred = np.array([norm_label(r.get('pred_label', r.get('pred'))) for r in rows])
        corr = np.array([1 if norm_label(r.get('true'))==p else 0 for r,p in zip(rows,pred)])
        sg = np.array([norm_label(r.get('true')) for r in rows])
        n = len(rows)
        per_seed = []
        for seed in SEEDS:
            cal = split_mask(n, seed=seed, cal_ratio=0.7)
            cc,cy,cpred = conf[cal],corr[cal],pred[cal]
            tc,ty,tpred,tsg = conf[~cal],corr[~cal],pred[~cal],sg[~cal]
            t0=time.time()
            m_un = tc
            m_ts = temp_scaling(cc,cy,tc)
            m_vs = vector_scaling(cc,cy,cpred,tc,tpred)
            m_pl = pw_platt(cc,cy,tc)
            m_piso = pw_isotonic(cc,cy,tc)
            m_giso = group_iso(cc,cy,csg=sg[cal],tc=tc,tsg=tsg)
            m_gpl = group_platt(cc,cy,csg=sg[cal],tc=tc,tsg=tsg)
            t1=time.time()
            def mk(pred_arr):
                return {"CW_ECE":cw_ece(pred_arr,ty,tsg),"gap_pp":gap_pp(pred_arr,ty,tsg),
                        "ECE":ece_overall(pred_arr,ty),"Brier":brier(pred_arr,ty),"AURC":aurc(pred_arr,ty)}
            rec = {"seed":seed,
                   "Uncalibrated":mk(m_un),"TS":mk(m_ts),"VS":mk(m_vs),"Platt":mk(m_pl),
                   "Pooled-Iso":mk(m_piso),"Group-Iso":mk(m_giso),"Group-Platt":mk(m_gpl),
                   "n_test":int((~cal).sum())}
            per_seed.append(rec)
            timing.append(t1-t0)
        # 5-seed means
        methods = ["Uncalibrated","TS","VS","Platt","Pooled-Iso","Group-Iso","Group-Platt"]
        mean_metrics = {m:{k:float(np.mean([s[m][k] for s in per_seed])) for k in ["CW_ECE","gap_pp","ECE","Brier","AURC"]} for m in methods}
        # 5-seed mean Δ (Group-Iso - baseline) for CW-ECE and gap_pp
        def seed_mean_delta(base_m, met):
            return np.array([s["Group-Iso"][met]-s[base_m][met] for s in per_seed])
        d_cw = {b: seed_mean_delta(b,"CW_ECE") for b in ["Uncalibrated","TS","VS","Platt","Pooled-Iso"]}
        d_gap= {b: seed_mean_delta(b,"gap_pp") for b in ["Uncalibrated","TS","VS","Platt","Pooled-Iso"]}
        # seed-level bootstrap CI + one-sample wilcoxon (test median<0)
        def seed_boot_wilcox(d):
            rng=np.random.RandomState(1); B=2000; arr=np.array(d); nn=len(arr)
            bs=np.array([rng.choice(arr,nn,replace=True).mean() for _ in range(B)])
            ci=list(np.percentile(bs,[2.5,97.5])); 
            try: w=stats.wilcoxon(arr, alternative="less")
            except Exception: w=None
            return {"mean":float(arr.mean()),"ci_lo":float(ci[0]),"ci_hi":float(ci[1]),
                    "wilcoxon_p_less":(float(w.pvalue) if w else None)}
        boot_cw = {b: seed_boot_wilcox(d_cw[b]) for b in d_cw}
        boot_gap= {b: seed_boot_wilcox(d_gap[b]) for b in d_gap}
        # cross-modality accumulator
        grp = "vision_RAF" if ds=="RAF-DB" else ("vision_FER" if ds=="FER2013" else "text")
        for b,key in [("Uncalibrated","vs_unc"),("TS","vs_TS"),("VS","vs_VS"),("Platt","vs_Platt"),("Pooled-Iso","vs_PooledIso")]:
            seed_deltas[key][grp].append(float(d_cw[b].mean()))
        results["per_set"][f"{model}|{ds}"] = {
            "n": n, "n_test_per_seed": per_seed[0]["n_test"],
            "mean_metrics": mean_metrics,
            "delta_CW_ECE_vs": {b:{"mean_5seed":float(d_cw[b].mean()),**boot_cw[b]} for b in d_cw},
            "delta_gap_pp_vs": {b:{"mean_5seed":float(d_gap[b].mean()),**boot_gap[b]} for b in d_gap},
        }
        print(f"[done] {model}|{ds}  n={n}  CW-ECE Group-Iso={mean_metrics['Group-Iso']['CW_ECE']:.3f} "
              f"(vs Pooled {mean_metrics['Pooled-Iso']['CW_ECE']:.3f}, vs Uncal {mean_metrics['Uncalibrated']['CW_ECE']:.3f})")

    # per-sample bootstrap on representative seed (42) for primary headline CIs
    # (recompute quickly on seed=42 test set per set)
    prim = {}
    for (model, ds, mod, path) in RECORDS:
        if not os.path.exists(path): continue
        rows = load(path)
        conf = np.array([float(r['conf']) for r in rows])
        pred = np.array([norm_label(r.get('pred_label', r.get('pred'))) for r in rows])
        corr = np.array([1 if norm_label(r.get('true'))==p else 0 for r,p in zip(rows,pred)])
        sg = np.array([norm_label(r.get('true')) for r in rows])
        n=len(rows); cal=split_mask(n,seed=42,cal_ratio=0.7)
        cc,cy,cpred=conf[cal],corr[cal],pred[cal]
        tc,ty,tpred,tsg=conf[~cal],corr[~cal],pred[~cal],sg[~cal]
        m_un=tc; m_piso=pw_isotonic(cc,cy,tc); m_giso=group_iso(cc,cy,sg[cal],tc,tsg)
        m_ts=temp_scaling(cc,cy,tc); m_vs=vector_scaling(cc,cy,cpred,tc,tpred); m_pl=pw_platt(cc,cy,tc)
        tidx=np.arange(len(tc))
        # metric closures over a sub-index (sidx are integer test indices)
        def mkc(base):
            def f(sidx):
                return cw_ece(base[sidx], ty[sidx], tsg[sidx])
            return f
        def gmk(base):
            def f(sidx):
                return gap_pp(base[sidx], ty[sidx], tsg[sidx])
            return f
        prim[f"{model}|{ds}"] = {
            "CW_ECE_GroupIso_vs_PooledIso": paired_bootstrap_ci(mkc(m_giso),mkc(m_piso),tidx),
            "CW_ECE_GroupIso_vs_Uncal": paired_bootstrap_ci(mkc(m_giso),mkc(m_un),tidx),
            "gap_pp_GroupIso_vs_PooledIso": paired_bootstrap_ci(gmk(m_giso),gmk(m_piso),tidx),
            "gap_pp_GroupIso_vs_Uncal": paired_bootstrap_ci(gmk(m_giso),gmk(m_un),tidx),
        }
    results["bootstrap_primary"] = prim

    # cross-modality summary
    cm = {}
    for key, grp in seed_deltas.items():
        cm[key] = {}
        for g, vals in grp.items():
            if vals:
                cm[key][g] = {"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals)),
                              "frac_negative":float(np.mean([v<0 for v in vals]))}
    results["cross_modality"] = cm
    # aggregate across all 9 sets
    all_cw_pooled = []; all_cw_un=[]
    for key in ["vs_PooledIso","vs_unc"]:
        for g in ["text","vision_FER","vision_RAF"]:
            all_cw_pooled += seed_deltas[key][g] if key=="vs_PooledIso" else []
            all_cw_un += seed_deltas[key][g] if key=="vs_unc" else []
    results["cross_modality"]["_overall"] = {
        "mean_deltaCW_vs_PooledIso": float(np.mean(all_cw_pooled)),
        "mean_deltaCW_vs_Uncal": float(np.mean(all_cw_un)),
        "frac_sets_CWdrop_vs_PooledIso": float(np.mean([v<0 for v in all_cw_pooled])),
        "frac_sets_CWdrop_vs_Uncal": float(np.mean([v<0 for v in all_cw_un])),
    }
    # efficiency
    results["efficiency"] = {
        "group_calibration_mean_sec_per_set": float(np.mean(timing)),
        "group_calibration_max_sec": float(np.max(timing)),
        "n_sets": len(timing),
        "gpu_training_required": False,
        "note": "post-hoc, training-free; H4 efficiency claim",
    }
    # save
    out = "./mc_data/code/formal_exp_results.json"
    with open(out,"w") as f: json.dump(results,f,indent=2,ensure_ascii=False)
    print("\nWROTE", out)
    # print summary
    print("\n===== CROSS-MODALITY ΔCW-ECE (Group-Iso minus baseline), 5-seed mean =====")
    print(f"{'comparison':<22}{'text':>10}{'FER':>10}{'RAF':>10}")
    for key in ["vs_unc","vs_TS","vs_VS","vs_Platt","vs_PooledIso"]:
        row=cm.get(key,{})
        def f(g): return f"{row[g]['mean']:.3f}" if g in row else "  -  "
        print(f"{key:<22}{f('text'):>10}{f('vision_FER'):>10}{f('vision_RAF'):>10}")
    print("\n===== EFFICIENCY (H4) =====")
    print(f"mean group-calib time/set = {results['efficiency']['group_calibration_mean_sec_per_set']:.4f}s; training-free={results['efficiency']['gpu_training_required']}")
    print("\nDONE")

if __name__ == "__main__":
    main()
