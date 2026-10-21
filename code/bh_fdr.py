import json, os

ROOT = "E:/WorkBuddy-work/emotion-multicalibration-paper/02-working/broaden_emotic"
RES = os.path.join(ROOT, "results")
PG = os.path.join(ROOT, "pg_experiment")
models = ["Qwen2.5-VL-3B-Instruct", "llava-1.5-7b-hf"]

def load(p):
    return json.load(open(p))

orig = {m: load(os.path.join(RES, f"h5_{m}.json")) for m in models}
pg   = {m: load(os.path.join(PG,  f"h5_pg_{m}.json")) for m in models}

family = []
def add(label, d, key):
    agg = d["aggregate"].get(key)
    if agg is None:
        print("MISSING", label, key); return
    family.append({"label": label, "p": agg["stouffer_p"], "z": agg["stouffer_z"]})

# H5a: CW-ECE reduction (core vs pooled)  -- original H5 JSON key names
for m in models:
    agg = orig[m]["aggregate"]
    family.append({"label": f"H5a CW-ECE core<pool [{m}]",
                   "p": agg["stouffer_p_cw"], "z": agg["stouffer_z_cw"]})
# H5b: worst-group improvement (core vs pooled)
for m in models:
    agg = orig[m]["aggregate"]
    family.append({"label": f"H5b WG-ECE core<pool [{m}]",
                   "p": agg["stouffer_p_wg"], "z": agg["stouffer_z_wg"]})
# P-G CW-ECE: pg vs core (degradation -> test delta>0)
for m in models:
    add(f"PG CW-ECE pg>core [{m}]", pg[m], "cw_pg_minus_core")
# P-G WG-ECE: pg vs pool (improvement -> delta<0)
for m in models:
    add(f"PG WG-ECE pg<pool [{m}]", pg[m], "wg_pg_minus_pooled")
# P-G WG-ECE: pg vs core
for m in models:
    add(f"PG WG-ECE pg<core [{m}]", pg[m], "wg_pg_minus_core")

pvals = [f["p"] for f in family]
m = len(pvals)
order = sorted(range(m), key=lambda i: pvals[i])
adj = [0.0] * m
prev = 1.0
for rank, idx in reversed(list(enumerate(order, start=1))):
    val = min(prev, pvals[idx] * m / rank)
    adj[idx] = val
    prev = val

print(f"==== BH-FDR (Benjamini-Hochberg), alpha=0.05, m={m} ====")
print(f"{'test':46s} {'raw_p':>12s} {'BH_q':>12s} {'sig@FDR':>9s}")
for i, f in enumerate(family):
    sig = "YES" if adj[i] < 0.05 else "no"
    print(f"{f['label']:46s} {f['p']:12.3e} {adj[i]:12.3e} {sig:>9s}")

# per-seed data for forest plot
out = {"models": {}, "family_fdr": []}
for i, f in enumerate(family):
    out["family_fdr"].append({"label": f["label"], "p": f["p"], "q": adj[i], "z": f["z"]})
keys = ["cw_core_minus_pooled", "cw_pg_minus_pooled", "cw_pg_minus_core",
        "wg_core_minus_pooled", "wg_pg_minus_pooled", "wg_pg_minus_core"]
for m_ in models:
    ps = pg[m_]["per_seed"]
    rec = {}
    for k in keys:
        rec[k] = {"mean": [s[k]["mean"] for s in ps],
                  "se":   [s[k]["se"]   for s in ps],
                  "p":    [s[k]["p"]    for s in ps]}
    out["models"][m_] = rec
json.dump(out, open(os.path.join(PG, "pg_fdr_and_seeddata.json"), "w"), indent=2)
print("\nsaved pg_fdr_and_seeddata.json")
