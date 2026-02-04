# Hybrid Knowledge Representation for Temporal Forecasting (Blind Submission)

This repository contains the code and notebooks to reproduce the experiments reported in the paper:

**“Hybrid Knowledge Representation for Temporal Forecasting: Interpretable Symbolic Extraction from ARIMAX and Rule-Based Learning.”**

The proposed framework integrates:
- **Statistical forecasting** (SARIMAX / ARIMAX with exogenous variables),
- **Machine learning baselines** (Linear Regression, Random Forest, HistGradientBoosting),
- **Symbolic rule extraction** (RuleFit),
- **Knowledge Graph construction** from extracted logical rules.

The main goal is to achieve competitive forecasting performance while producing **interpretable symbolic representations** (rules and graphs) that describe climatic drivers of electricity consumption.

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
│   └── make_knowledge_graph_from_rules.py
│
├── data/
│   ├── raw/
│   └── processed/
│
└── outputs/
    ├── datasets/
    ├── metrics/
    ├── rules/
    └── figures/
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

### Option B — conda
If you prefer using conda, create an environment and install the dependencies listed in `requirements.txt`.

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

### Step 1 — Load and preprocess electricity consumption
Run:
- `notebooks/Consumo_Eletricidade.ipynb`

Outputs:
- processed national/zonal consumption datasets saved under:
  - `data/processed/`
  - `outputs/datasets/`

---

### Step 2 — Build meteorological dataset + merge with consumption
Run:
- `notebooks/Meteorologia.ipynb`

Expected outputs:
- `dataset_meteo_com_consumo.csv`
- `dataset_meteo_zonal.csv`
- `target_consumo.csv`
- `raw_{zone}.csv`

---

### Step 3 — Temperature clustering and mapping
Run:
- `notebooks/build_temperature_clusters_and_map.ipynb`
- `notebooks/Clustering_Consumo_Zonal.ipynb`

Expected outputs:
- cluster assignments per zone
- temperature cluster maps/figures

---

### Step 4 — Feature engineering and feature analysis
Run:
- `notebooks/Feature Eng.ipynb`
- `notebooks/feature_analysis.ipynb`

Expected outputs:
- engineered dataset (final training table)
- feature correlation heatmaps / mutual information ranking

---

### Step 5 — Forecasting models (SARIMAX + baselines)
Run:
- `notebooks/model_com_lags.ipynb`
- `notebooks/Model.ipynb`

Models included:
- SARIMAX / ARIMAX with exogenous inputs
- Linear Regression
- Random Forest
- HistGradientBoosting

Evaluation protocol:
- one-day-ahead forecasting
- temporal split: first 80% train, last 20% test

Expected outputs:
- `outputs/metrics/metrics_{zone}.csv`
- learning curve figures (e.g., `learning_curve_xgb.png`)

---

### Step 6 — Rule extraction (RuleFit) and rule export
Run:
- `notebooks/knowledge_extraction_notebook.ipynb`

Expected output:
- `outputs/rules/rules.csv`

This file contains extracted rules with:
- rule string (logical conjunction)
- coefficient (importance / polarity)
- support (coverage)

---

### Step 7 — Knowledge Graph generation from rules
Run:
```bash
python src/make_knowledge_graph_from_rules.py
```

Expected output:
- `outputs/figures/knowledge_graph_*.png`

Graph structure:
- **condition nodes → rule nodes → outcome nodes**
- outcome nodes: **Consumption Increase** / **Consumption Decrease**
- rule-to-outcome edge thickness is proportional to:
  - `|coef| × support`

---

## 5. Notes for reviewers (blind submission)

- This repository is anonymized for double-blind review.
- All scripts should use relative paths (no user-specific directories).
- If any dataset is not included, it must be placed under `data/raw/` following the structure described above.

---

## 6. Citation

If you use this codebase, please cite the corresponding paper submission.
