# Multi-Calibrated Confidence for Affective Subjectivity and Group Fairness

Code and derived results for the paper **"Multi-Calibrated Confidence for Affective Subjectivity
and Group Fairness"** (under review at *Pattern Recognition*).

> **Partial release.** This repository contains the core subgroup-calibration method, the
> experiment and analysis scripts, and the frozen derived results reported in the paper.
> The complete codebase, the frozen per-sample prediction records, and all remaining data
> artifacts will be released **in full upon acceptance** of the manuscript.

## What this paper studies

Post-hoc confidence calibration for affective LLMs/MLLMs under **subjective** labels and
**group-fairness** requirements. The proposed framework $\mathcal{C}_{\text{emo}}$ extends
multicalibration to an affect-specific subgroup family and supports three calibration
targets: hard labels, the human label distribution $p_{\text{human}}$, and a correctness–agreement
trade-off controlled by $\lambda$.

Headline findings (see the paper for full details):

- Group calibration reduces class-weighted ECE (CW-ECE over $\mathcal{C}_{\text{emo}}$) by up to
  **83.5%** versus pooled isotonic regression;
- It **halves the worst-subgroup calibration error** on both vision settings tested
  (RAF-DB and EMOTIC), robust across 12 binning configurations and surviving BH-FDR;
- EMOTIC replication resolves the single-source threat to the vision claim; the worst-group
  null on the independent EMOTIC attributes is disclosed as a controlled negative result (P-G);
- Calibrating toward human agreement exposes a monotone **correctness–human-agreement
  trade-off**, quantified via a family-ladder analysis.

## Repository layout

```
code/
  calibration_core.py        Core C_emo subgroup-calibration operators (multi-group fixpoint)
  formal_exp_v1.py           Main study (H1/H2): GoEmotions text + FER2013 vision records
  formal_exp_large.py        Scale-up records (FERPlus, AffectNet)
  rigor_exp_v2.py            Rigour battery R1-R8 (seeds, splits, binning sensitivity)
  rigor_r5b_r8b.py           Generic multicalibration baselines (R5) + fairness study (R8)
  h3_power_scan.py           H3 power analysis (worst-subgroup claim)
  h3_worstgroup_robust.py    H3 robustness across binning configurations
  h3_power_curve.py          Power curve construction
  h3_metric_power.py         Metric-level power checks
  emotic_analysis.py         H5 EMOTIC replication analysis
  emotic_analysis_pg.py      P-G controlled negative result (4-way calibration family)
  bh_fdr.py                  Benjamini-Hochberg FDR correction
  make_pg_forest.py          P-G forest figure
  build_f4p_figures.py       Reproduces all main-paper figures (figkit publication style)
results/                     Frozen derived results (JSON) reported in the paper
```

All numeric claims in the paper can be traced to the frozen JSON files in `results/`.
Fixed seeds: `{42, 43, 44, 45, 46}` (main study) and `{42, 7, 13, 21, 101}` (EMOTIC and
scale-up experiments).

## Data

The five datasets used in this study are **public but licence-restricted**; we therefore do
not redistribute any raw data or model outputs containing dataset content:

| Dataset | Used for | Source |
|---|---|---|
| GoEmotions | Text (H1/H2) | `https://github.com/google-research/google-research/tree/master/goemotions` |
| RAF-DB | Vision (H3) | `http://www.whdeng.cn/RAF/model1.html` (registration required) |
| EMOTIC | Vision (H5, demographics) | `https://cvrc.ece.utexas.edu/EMOTIC.html` (request access) |
| FERPlus | Vision (scale-up) | `https://github.com/microsoft/FERPlus` |
| AffectNet | Vision (scale-up) | `http://mohammadmahoor.com/affectnet/` (request access) |

Scripts in `code/` expect the derived `records.jsonl` prediction files under
`./mc_data/` and `./trustbench_emo_data/` (paths normalised from the compute-node layout).
The frozen prediction records will be released **in full upon acceptance**.

## Requirements

```
python >= 3.10
numpy, scipy, pandas, matplotlib, scikit-learn
```

## Citation

If you use this code or the $\mathcal{C}_{\text{emo}}$ framework, please cite:

```bibtex
@article{wan2026multicalibrated,
  title   = {Multi-Calibrated Confidence for Affective Subjectivity and Group Fairness},
  author  = {Wan, Zhenghao and Wu, Chunlei and Zhang, Huan and Liu, Xu},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {Under review}
}
```

## Availability statement

A partial release of the subgroup-calibration core, the experiment and analysis scripts, and
the frozen derived results is available in this repository. **The complete codebase, frozen
prediction records, and all remaining data artifacts will be released in full upon
acceptance.**

## License

Code in this repository is released under the [MIT License](LICENSE).
The derived results in `results/` are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Dataset licences remain governed by the original providers.
