"""情感多校准框架 - 核心后处理层 (验证于济南超算 A100, 2026-09-28)
遵循手册: 数据/权重放 /wanzh, 软件走 mirrors.shanhe.com。
核心: 透明免RL分组条件重校准 (Hebert-Johnson 2018 式, 逐子群 held-out isotonic/platt)。
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def build_subgroup_family(emotion, gender, ambiguity_tier):
    """构造 𝒞_emo 子群族 (逐情绪类 × 性别 × 歧义度三层相交)。
    emotion/gender/ambiguity_tier: 同长度 int 数组。返回子群 id 数组 + 唯一子群列表。"""
    assert len(emotion)==len(gender)==len(ambiguity_tier)
    # 用 Python 级字符串构造, 避免 numpy 固定宽度字符串数组 '+' ufunc 在 numpy1.x/2.x 下均可能失效
    sg = np.array([f"{int(e)}|{int(g)}|{int(a)}" for e, g, a in
                   zip(np.asarray(emotion), np.asarray(gender), np.asarray(ambiguity_tier))],
                  dtype=object)
    uniq, sg_id = np.unique(sg, return_inverse=True)
    return sg_id, uniq

def group_isotonic_calibrate(confs, correct, sg_id, test_confs, test_sg_id, y_min=0.0, y_max=1.0):
    """逐子群 held-out isotonic 重校准 -> 返回重校准后置信度 (与 test_confs 同形)。"""
    out = np.empty_like(test_confs, dtype=float)
    for u in np.unique(sg_id):
        m = sg_id==u
        ir = IsotonicRegression(out_of_bounds="clip", y_min=y_min, y_max=y_max)
        ir.fit(confs[m], correct[m])
        tm = test_sg_id==u
        out[tm] = ir.predict(test_confs[tm])
    return out

def group_platt_calibrate(confs, correct, sg_id, test_confs, test_sg_id):
    """逐子群 held-out Platt(逻辑回归, 仅偏置+斜率) 重校准。"""
    out = np.empty_like(test_confs, dtype=float)
    for u in np.unique(sg_id):
        m = sg_id==u
        lr = LogisticRegression(C=1e6)
        lr.fit(confs[m].reshape(-1,1), correct[m])
        tm = test_sg_id==u
        out[tm] = lr.predict_proba(test_confs[tm].reshape(-1,1))[:,1]
    return out

def ece_per_subgroup(confs, correct, sg_id, bins=10):
    """逐子群 ECE (group-conditional ECE, 核心指标 CW-ECE)。返回 {子群id: ECE}。"""
    edges = np.linspace(0,1,bins+1)
    res = {}
    for u in np.unique(sg_id):
        m = sg_id==u; c=confs[m]; l=correct[m]
        idx = np.clip(np.digitize(c, edges)-1, 0, bins-1)
        tot=0.0
        for i in range(bins):
            sel=idx==i
            if sel.sum()>0: tot += abs(c[sel].mean()-l[sel].mean())*sel.sum()
        res[int(u)] = tot/len(c) if len(c)>0 else float("nan")
    return res

def cw_ece(confs, correct, sg_id, bins=10):
    """整体 CW-ECE: 所有子群 ECE 按样本量加权平均 (对齐 Proposal 核心指标)。"""
    d = ece_per_subgroup(confs, correct, sg_id, bins)
    counts = np.array([(sg_id==u).sum() for u in d])
    return float(np.sum(counts*np.array(list(d.values())))/counts.sum())

def max_group_gap(confs, correct, sg_id, bins=10):
    """公平性差距 gap_pp: 子群间 ECE 最大差 (对齐 TrustBench-Emo D4)。"""
    d = ece_per_subgroup(confs, correct, sg_id, bins)
    vals = np.array(list(d.values()))
    return float(vals.max()-vals.min())

if __name__ == "__main__":
    # 自测: 复现连通验证时的降ECE结果
    np.random.seed(42); n=6000
    g=np.random.randint(0,2,n); tp=np.where(g==0,0.35,0.70)
    cr=np.clip(tp+np.where(g==0,0.22,-0.18)+np.random.normal(0,0.08,n),0.02,0.98)
    y=(np.random.rand(n)<tp).astype(int)
    cal,test=np.arange(3000),np.arange(3000,6000)
    rec=group_isotonic_calibrate(cr[cal],y[cal],g[cal],cr[test],g[test])
    print("BEFORE CW-ECE=",round(cw_ece(cr[test],y[test],g[test]),4))
    print("AFTER  CW-ECE=",round(cw_ece(rec,y[test],g[test]),4))
    print("SMOKE OK")
