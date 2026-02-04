# Hybrid Knowledge Representation for Temporal Forecasting (Blind Submission)

This repository contains code, scripts, and notebooks to reproduce the experiments reported in the paper:

**“Hybrid Knowledge Representation for Temporal Forecasting: Interpretable Symbolic Extraction from ARIMAX and Rule-Based Learning.”**

The framework integrates:
- **Statistical forecasting** (SARIMAX / ARIMAX with exogenous variables),
- **Machine learning baselines** (Linear Regression, Random Forest, HistGradientBoosting),
- **Symbolic rule extraction** (RuleFit),
- **Knowledge Graph construction** from extracted logical rules,
- Optional **diagnostics & explainability** (residual diagnostics + SHAP).

> **Blind-review note:** this repository is anonymized for double-blind review. Please do not include author/institution identifiers in files, headers, paths, or figures.

---

## 1. Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── EDA_Consumo_Nacional.ipynb
│   ├── Consumo_Eletricidade.ipynb
│   ├── Meteorologia.ipynb
│   ├── build_temperature_clusters_and_map.ipynb
│   ├── Clustering_Consumo_Zonal.ipynb
│   ├── Feature Eng.ipynb
│   ├── feature_analysis.ipynb
│   ├── model_com_lags.ipynb
│   ├── Model.ipynb
│   └── knowledge_extraction_notebook.ipynb
│
├── src/
│   ├── arimax_experiments_pack.py
│   ├── extract_rules_with_coeffs.py
│   ├── make_knowledge_graph_from_rules.py
│   └── make_rule_consequents_to_latex.py
│
├── data/
│   ├── raw/
│   └── processed/
│
└── outputs/
    ├── datasets/
    ├── metrics/
    ├── rules/
    ├── figures/
    ├── exp_out/
    └── reports/
```

---

## 2. Installation

### Option A — pip
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

> If you do not need SHAP, you can still run the pipeline; SHAP outputs are generated only when the `shap` package is installed.

---

## 3. Data

This project uses open-access electricity consumption and meteorological data for Portuguese zones (district-level reference cities), covering the period:

- **2015-01-01 to 2025-10-05**

The paper considers the following zones:
- Lisboa, Porto, Faro, Évora, Braga, Coimbra

If the dataset is not included in this repository, it must be placed under:

```
data/raw/
```

---

## 4. Reproducing paper results (recommended pipeline)

This project can be reproduced either by running the notebooks (exploration + pipeline), or by using the scripts in `src/` for a more automated execution.

### A) Notebook pipeline (recommended for reviewers)
1. `notebooks/Consumo_Eletricidade.ipynb` — load and preprocess consumption.
2. `notebooks/Meteorologia.ipynb` — merge meteo + consumption and generate zonal files.
3. `notebooks/build_temperature_clusters_and_map.ipynb` and `notebooks/Clustering_Consumo_Zonal.ipynb` — clustering + maps.
4. `notebooks/Feature Eng.ipynb` and `notebooks/feature_analysis.ipynb` — engineered features and feature analysis.
5. `notebooks/model_com_lags.ipynb` and `notebooks/Model.ipynb` — forecasting models and evaluation.
6. `notebooks/knowledge_extraction_notebook.ipynb` — rule extraction and exports used downstream.

Expected notebook outputs (examples):
- `outputs/metrics/metrics_{zone}.csv`
- `outputs/figures/learning_curve_*.png`
- `outputs/rules/rules.csv` (or equivalent RuleFit export)

### B) Script pipeline (automated / CLI)

#### 1) ARIMAX experiments pack (grid + fair comparison + diagnostics + report)
```bash
python src/arimax_experiments_pack.py   --csv data/processed/dataset_meteo_com_consumo.csv   --out-dir outputs/exp_out   --exog-cols "tmean_c,hdd18,cdd22,rad_solar,humidade_relativa"   --lags "1,7,14"   --seasonal-periods "7,30"   --fair-compare
```

Main outputs:
- `outputs/exp_out/baselines_metrics.csv`
- `outputs/exp_out/sarimax_grid_summary.csv`
- `outputs/exp_out/comparison_arimax_vs_baselines.csv`
- `outputs/exp_out/compare_arimax_vs_baselines_test.png`
- `outputs/exp_out/diag_*/` (ACF + Ljung-Box)
- `outputs/exp_out/report_arimax_pack.md`

Optional notebook-style eval (in-sample):
```bash
python src/arimax_experiments_pack.py   --csv data/processed/dataset_meteo_com_consumo.csv   --out-dir outputs/exp_out_in_sample   --exog-cols "tmean_c,hdd18,cdd22,rad_solar,humidade_relativa"   --lags "1,7,14"   --fair-compare   --eval-mode in_sample
```

#### 2) Extract RuleFit rules + coefficients (target-driven coefficients)
```bash
python src/extract_rules_with_coeffs.py   --csv data/processed/dataset_meteo_com_consumo.csv   --target consumo_gwh   --out outputs/knowledge/rulefit_rules_complete.csv   --min-support 0.02   --max-len 3
```

Main output:
- `outputs/knowledge/rulefit_rules_complete.csv` with columns: `rule, coef, support, len, score`

#### 3) Create knowledge graph visualizations from rules
```bash
python src/make_knowledge_graph_from_rules.py   --rules-csv outputs/knowledge/rulefit_rules_complete.csv   --out-prefix outputs/figures/knowledge_graph   --min-support 0.02   --top 40
```

Main outputs:
- `outputs/figures/knowledge_graph_detailed.png`
- `outputs/figures/knowledge_graph_paper.png`
- `outputs/figures/knowledge_graph_vars_both.png`
- `outputs/figures/knowledge_graph_vars_increase.png`
- `outputs/figures/knowledge_graph_vars_decrease.png`

#### 4) Export rules to LaTeX with English consequents
```bash
python src/make_rule_consequents_to_latex.py   --rules-csv outputs/knowledge/rulefit_rules_complete.csv   --out-tex outputs/reports/rules_table.tex   --top 30   --min-support 0.02   --round 3   --sort-by abscoef
```

Main outputs:
- `outputs/reports/rules_table.tex`
- terminal printout of top rules and consequents

---

## 5. Notes for reviewers (blind submission)

- This repository is anonymized for double-blind review.
- Use relative paths only (no personal directories / OneDrive paths).
- If any dataset is not included, place it under `data/raw/` and follow the notebook instructions to generate `data/processed/`.
- If any script headers contain author names, remove them before making the repository public for review.

---

## 6. Citation

If you use this codebase, please cite the corresponding paper submission.
