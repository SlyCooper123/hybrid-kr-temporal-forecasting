#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: arimax_experiments_pack.py (verbose + fair compare, notebook-like eval)
Author: XXX

What it does:
  - SARIMAX grid search over (p,d,q)(P,D,Q,s) with detailed logs
  - Builds lagged exogenous features
  - Temporal train/test split (default 80/20)
  - Residual diagnostics (Ljung-Box test + ACF plots)
  - Baselines (Linear Regression, RandomForest, HistGradientBoosting) + SHAP
  - FAIR COMPARISON: ARIMAX trained on the training split

"""

import argparse, itertools, time
from time import perf_counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ------------------------------
# Logging helpers
# ------------------------------
def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def step(title: str):
    class _CM:
        def __enter__(self):
            self.t0 = perf_counter()
            log(f"=== {title} — START ===")
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = perf_counter() - self.t0
            if exc_type is None:
                log(f"=== {title} — END (elapsed: {dt:.2f}s) ===")
            else:
                log(f"!!! {title} — ERROR after {dt:.2f}s: {exc}")
            return False

    return _CM()


# ------------------------------
# File & utils
# ------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    trials = [(e, s) for e in ("utf-8", "latin1") for s in (",", ";", "\t")]
    last_err = None
    for enc, sep in trials:
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep)
            if df.shape[1] >= 3 and df.shape[0] >= 100:
                log(f"Parsed CSV enc={enc} sep='{sep}' shape={df.shape}")
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not parse CSV '{path}'. Last error: {last_err}")


def coerce_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if df[date_col].isna().all():
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def make_lagged(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
    return df


def metrics_from(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    log(f"Saved: {path.resolve()} (rows={len(df):,})")


def plot_acf(series: np.ndarray, max_lag: int, out_file: Path, title: str):
    x = series.astype(float)
    x = x - np.nanmean(x)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        v1 = x[:-lag]
        v2 = x[lag:]
        ac = np.nansum(v1 * v2) / np.nansum(x * x)
        acf.append(ac)
    plt.figure(figsize=(7, 3))
    plt.stem(range(0, max_lag + 1), acf)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


@dataclass
class Config:
    csv: Path
    out_dir: Path
    date_col: str
    target: str
    exog_cols: List[str]
    lag_list: List[int]
    seasonal_periods: List[int]
    p: List[int]
    d: List[int]
    q: List[int]
    P: List[int]
    D: List[int]
    Q: List[int]
    split_ratio: float = 0.8
    maxiter: int = 80
    light_sarimax: bool = False
    log_interval: int = 10
    limit_combos: int = 0
    # fair compare
    fair_compare: bool = False
    order_fc: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order_fc: Tuple[int, int, int, int] = (1, 0, 1, 7)
    # notebook-like eval
    eval_mode: str = "grid"  # grid | in_sample


def build_exog_matrix(df: pd.DataFrame, cols: List[str]) -> Optional[np.ndarray]:
    if not cols:
        return None
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    return df[present].astype(float).values


def fit_sarimax(
    y: np.ndarray,
    X: Optional[np.ndarray],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    maxiter: int,
    light: bool,
):
    model = SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        simple_differencing=light,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=maxiter)
    return res


# ------------------------------
# EVAL one-step-ahead (TEST) — usa X_test!
# ------------------------------
def evaluate_sarimax_one_step(
    res,
    y_train: np.ndarray,
    X_train: Optional[np.ndarray],
    y_test: np.ndarray,
    X_test: Optional[np.ndarray],
    out_dir: Path,
    tag: str,
) -> Dict[str, float]:
    # in-sample (train)
    try:
        pred_in = res.get_prediction(start=0, end=len(y_train) - 1, exog=X_train)
        yhat_in = np.asarray(pred_in.predicted_mean).astype("float32")
        n_in = min(len(y_train), len(yhat_in))
        in_metrics = metrics_from(y_train[-n_in:], yhat_in[-n_in:])
        pd.DataFrame([{"set": "train", **in_metrics}]).to_csv(
            out_dir / f"sarimax_{tag}_train_metrics.csv", index=False
        )
    except Exception:
        pass

    # out-of-sample (one-step-ahead) no TEST
    try:
        pred_out = res.get_forecast(steps=len(y_test), exog=X_test)
    except Exception:
        pred_out = res.get_forecast(steps=len(y_test))

    yhat = np.asarray(pred_out.predicted_mean).astype("float32")
    n = min(len(y_test), len(yhat))
    test_metrics = metrics_from(y_test[:n], yhat[:n])
    pd.DataFrame([{"set": "test", **test_metrics}]).to_csv(
        out_dir / f"sarimax_{tag}_test_metrics.csv", index=False
    )
    return test_metrics


def compare_baselines(X_train, y_train, X_test, y_test, out_dir: Path) -> pd.DataFrame:
    rows = []
    models = []

    lr = LinearRegression().fit(X_train, y_train)
    rows.append(
        {"model": "LinearRegression", **metrics_from(y_test, lr.predict(X_test))}
    )
    models.append(("LinearRegression", lr))

    rf = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=10, n_jobs=-1, random_state=0
    ).fit(X_train, y_train)
    rows.append({"model": "RandomForest", **metrics_from(y_test, rf.predict(X_test))})
    models.append(("RandomForest", rf))

    hgb = HistGradientBoostingRegressor(
        max_depth=10, learning_rate=0.08, max_iter=400, random_state=0
    ).fit(X_train, y_train)
    rows.append(
        {"model": "HistGradientBoosting", **metrics_from(y_test, hgb.predict(X_test))}
    )
    models.append(("HistGradientBoosting", hgb))

    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "baselines_metrics.csv")

    if HAS_SHAP:
        try:
            import warnings

            warnings.filterwarnings("ignore")
            bg_idx = np.random.RandomState(0).choice(
                len(X_train), size=min(1000, len(X_train)), replace=False
            )
            background = X_train[bg_idx]
            for name, mdl in models:
                if name in ("RandomForest", "HistGradientBoosting"):
                    log(f"[SHAP] Explaining {name} …")
                    explainer = shap.Explainer(mdl, background)
                    shap_values = explainer(X_test[:2000])
                    shap_df = pd.DataFrame(
                        np.abs(shap_values.values).mean(axis=0),
                        columns=["mean_abs_shap"],
                    )
                    if hasattr(mdl, "feature_names_in_"):
                        shap_df["feature"] = mdl.feature_names_in_
                    else:
                        shap_df["feature"] = [f"x{i}" for i in range(shap_df.shape[0])]
                    shap_df = shap_df[["feature", "mean_abs_shap"]].sort_values(
                        "mean_abs_shap", ascending=False
                    )
                    plt.figure(figsize=(9, 5))
                    shap.summary_plot(
                        shap_values, X_test[:2000], show=False, plot_type="dot"
                    )
                    plt.tight_layout()
                    plt.savefig(out_dir / f"shap_{name}_summary.png", dpi=150)
                    plt.close()

                    shap_df.to_csv(out_dir / f"shap_{name}.csv", index=False)

        except Exception as e:
            log(f"[warn] SHAP skipped: {e}")

    return df


def residual_diagnostics(residuals: np.ndarray, out_dir: Path) -> None:
    ensure_dir(out_dir)
    residuals = residuals.astype(float)
    try:
        lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
        lb.to_csv(out_dir / "ljung_box.csv", index=False)
    except Exception as e:
        log(f"[warn] Ljung-Box failed: {e}")
    try:
        plot_acf(
            residuals,
            max_lag=60,
            out_file=out_dir / "acf_residuals.png",
            title="ACF of residuals",
        )
    except Exception:
        pass


def build_report(out_dir: Path, meta: Dict[str, str], best_rows):
    lines = []
    lines.append("# ARIMAX Experiments Report\n\n")
    lines.append(
        "This report summarizes SARIMAX grid-search, residual diagnostics, and baseline comparisons.\n\n"
    )
    lines.append("## Setup\n")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}\n")
    lines.append("\n---\n")
    if best_rows:
        lines.append("## Best SARIMAX Configurations (by Test RMSE)\n")
        df_best = pd.DataFrame(best_rows).sort_values("RMSE").reset_index(drop=True)
        lines.append(df_best.to_markdown(index=False) + "\n\n---\n")
    if (out_dir / "baselines_metrics.csv").exists():
        lines.append("## Baseline Models (Test Metrics)\n")
        dfb = pd.read_csv(out_dir / "baselines_metrics.csv")
        lines.append(dfb.to_markdown(index=False) + "\n\n---\n")
    if (out_dir / "ljung_box.csv").exists():
        lines.append("## Residual Diagnostics (Ljung-Box)\n")
        lines.append(
            pd.read_csv(out_dir / "ljung_box.csv").to_markdown(index=False) + "\n\n"
        )
    if (out_dir / "acf_residuals.png").exists():
        lines.append("![acf_residuals](acf_residuals.png)\n\n")
    for name in ("RandomForest", "HistGradientBoosting"):
        p = out_dir / f"shap_{name}.csv"
        if p.exists():
            lines.append(f"## SHAP — {name}\n")
            sdf = pd.read_csv(p)
            lines.append(sdf.head(20).to_markdown(index=False) + "\n\n---\n")
    (out_dir / "report_arimax_pack.md").write_text("".join(lines), encoding="utf-8")
    log(f"Report saved to {(out_dir/'report_arimax_pack.md').resolve()}")


# ------------------------------
# Fair comparison helper (one-step-ahead no TEST)
# ------------------------------
def fair_compare_arimax_vs_baselines(
    dates_test: pd.Series,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray],
    X_test: Optional[np.ndarray],
    order_fc: Tuple[int, int, int],
    seasonal_order_fc: Tuple[int, int, int, int],
    out_dir: Path,
):
    rows = []

    res = SARIMAX(
        y_train,
        exog=X_train,
        order=order_fc,
        seasonal_order=seasonal_order_fc,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    try:
        pred = res.get_forecast(steps=len(y_test), exog=X_test)
    except Exception:
        pred = res.get_forecast(steps=len(y_test))
    yhat_arimax = np.asarray(pred.predicted_mean).astype(float)
    rows.append({"model": "ARIMAX", **metrics_from(y_test, yhat_arimax)})

    yhat_lr = yhat_rf = yhat_hgb = None
    if X_train is not None and X_train.shape[1] > 0:
        lr = LinearRegression().fit(X_train, y_train)
        yhat_lr = lr.predict(X_test)
        rf = RandomForestRegressor(
            n_estimators=400, min_samples_leaf=10, n_jobs=-1, random_state=0
        ).fit(X_train, y_train)
        yhat_rf = rf.predict(X_test)
        hgb = HistGradientBoostingRegressor(
            max_depth=10, learning_rate=0.08, max_iter=400, random_state=0
        ).fit(X_train, y_train)
        yhat_hgb = hgb.predict(X_test)
        rows.append({"model": "LinearRegression", **metrics_from(y_test, yhat_lr)})
        rows.append({"model": "RandomForest", **metrics_from(y_test, yhat_rf)})
        rows.append({"model": "HistGradientBoosting", **metrics_from(y_test, yhat_hgb)})
    else:
        rows.append(
            {
                "model": "(no X) Baselines skipped",
                "RMSE": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
            }
        )

    cmp = pd.DataFrame(rows).sort_values("RMSE")
    save_csv(cmp, out_dir / "comparison_arimax_vs_baselines.csv")

    plt.figure(figsize=(11, 4))
    plt.plot(dates_test, y_test, label="Observed", linewidth=2)
    plt.plot(dates_test, yhat_arimax, label="ARIMAX", linewidth=1.8)
    if yhat_lr is not None:
        plt.plot(dates_test, yhat_lr, label="LinearReg.", alpha=0.9)
        plt.plot(dates_test, yhat_rf, label="RandomForest", alpha=0.9)
        plt.plot(dates_test, yhat_hgb, label="HistGB", alpha=0.9)
    plt.title("Out-of-sample comparison: ARIMAX vs baselines (one-step-ahead)")
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_arimax_vs_baselines_test.png", dpi=150)
    plt.close()

    log(
        "Fair comparison saved: comparison_arimax_vs_baselines.csv / compare_arimax_vs_baselines_test.png"
    )
    return cmp


# ------------------------------
# Main driver
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Parametric ARIMAX experiments with baselines, diagnostics, fair comparison and notebook-like eval (verbose)."
    )
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="exp_out")
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--exog-cols", default="")
    ap.add_argument("--lags", default="1,7,14")
    ap.add_argument("--seasonal-periods", default="7,30")
    ap.add_argument("--p", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--d", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--q", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--P", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--D", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--Q", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--limit-combos", type=int, default=0)
    # fair compare
    ap.add_argument("--fair-compare", action="store_true")
    ap.add_argument(
        "--order", nargs=3, type=int, metavar=("p", "d", "q"), default=[1, 1, 1]
    )
    ap.add_argument(
        "--seasonal-order",
        nargs=4,
        type=int,
        metavar=("P", "D", "Q", "s"),
        default=[1, 0, 1, 7],
    )
    # notebook-like
    ap.add_argument(
        "--eval-mode",
        choices=["grid", "in_sample"],
        default="grid",
        help="grid (padrão) ou in_sample (estilo notebook)",
    )

    args = ap.parse_args()

    cfg = Config(
        csv=Path(args.csv),
        out_dir=Path(args.out_dir),
        date_col=args.date_col if args.date_col else "",
        target=args.target if args.target else "",
        exog_cols=[c.strip() for c in args.exog_cols.split(",") if c.strip()],
        lag_list=[int(x) for x in args.lags.split(",") if x.strip()],
        seasonal_periods=[
            int(x) for x in args.seasonal_periods.split(",") if x.strip()
        ],
        p=args.p,
        d=args.d,
        q=args.q,
        P=args.P,
        D=args.D,
        Q=args.Q,
        split_ratio=float(args.split),
        maxiter=int(args.maxiter),
        light_sarimax=bool(args.light),
        log_interval=int(args.log_interval),
        limit_combos=int(args.limit_combos),
        fair_compare=bool(args.fair_compare),
        order_fc=tuple(args.order),
        seasonal_order_fc=tuple(args.seasonal_order),
        eval_mode=args.eval_mode,
    )

    log("Starting ARIMAX experiments …")
    log(
        f"Params: out_dir={cfg.out_dir} split={cfg.split_ratio} maxiter={cfg.maxiter} light={cfg.light_sarimax}"
    )
    log(
        f"       exog={cfg.exog_cols} lags={cfg.lag_list} seasonal_periods={cfg.seasonal_periods}"
    )
    log(f"       grids: p={cfg.p} d={cfg.d} q={cfg.q} | P={cfg.P} D={cfg.D} Q={cfg.Q}")
    if cfg.fair_compare:
        log(
            f"       FAIR COMPARE on: order={cfg.order_fc}, seasonal_order={cfg.seasonal_order_fc}"
        )
    log(f"       EVAL MODE: {cfg.eval_mode}")

    ensure_dir(cfg.out_dir)

    with step("Load & normalize CSV"):
        df = read_csv_robust(cfg.csv)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        log(f"Columns: {list(df.columns)[:12]}{' …' if len(df.columns)>12 else ''}")

    with step("Detect date/target + build lagged frame"):
        date_col = cfg.date_col or (
            "date"
            if "date" in df.columns
            else ("data" if "data" in df.columns else None)
        )
        if date_col is None:
            raise SystemExit("Date column not found (expected 'date' or 'data').")
        target = cfg.target
        if not target:
            for cand in (
                "consumo_gwh",
                "consumo",
                "target",
                "gwh",
                "consumo_diario_gwh",
            ):
                if cand in df.columns:
                    target = cand
                    break
        if not target:
            raise SystemExit("Target column not found (e.g., 'consumo_gwh').")

        df = coerce_dates(df, date_col).sort_values(date_col).reset_index(drop=True)
        if cfg.exog_cols and cfg.lag_list:
            df = make_lagged(df, cfg.exog_cols, cfg.lag_list)

        all_exog = []
        for c in cfg.exog_cols:
            all_exog.append(c)
            for L in cfg.lag_list:
                name = f"{c}_lag{L}"
                if name in df.columns:
                    all_exog.append(name)

        cols = [date_col, target] + all_exog
        cols = [c for c in cols if c in df.columns]
        df_model = df[cols].dropna().reset_index(drop=True)
        log(
            f"Modeling frame after drops: shape={df_model.shape} (exog_used={len(all_exog)})"
        )
        if len(df_model) < 300:
            log(
                "[warn] Too few rows after dropping NaNs from lags. Consider fewer/lower lags."
            )

    # --- Notebook-like: usa FULL DATA in-sample (igual ao caderno) ---
    if cfg.eval_mode == "in_sample":
        with step("Fit SARIMAX on FULL data (notebook-like)"):
            y_all = df_model[target].astype(float).values
            X_all = None if not all_exog else df_model[all_exog].astype(float).values
            res = fit_sarimax(
                y_all,
                X_all,
                cfg.order_fc,
                cfg.seasonal_order_fc,
                cfg.maxiter,
                cfg.light_sarimax,
            )

        with step("In-sample prediction + metrics + plot (notebook-like)"):
            pred = res.get_prediction()
            sf = pred.summary_frame()
            dates = pd.to_datetime(df_model[date_col].values)
            yhat = np.asarray(sf["mean"]).astype(float)
            lo = np.asarray(sf.get("mean_ci_lower", sf.iloc[:, 0])).astype(float)
            hi = np.asarray(sf.get("mean_ci_upper", sf.iloc[:, 1])).astype(float)

            rmse = float(np.sqrt(mean_squared_error(df_model[target].values, yhat)))
            mae = float(mean_absolute_error(df_model[target].values, yhat))
            r2 = float(r2_score(df_model[target].values, yhat))
            print(
                pd.DataFrame(
                    {"metric": ["RMSE", "MAE", "R²"], "value": [rmse, mae, r2]}
                )
            )

            plt.figure(figsize=(10, 4))
            plt.plot(dates, df_model[target].values, label="obs")
            plt.plot(dates, yhat, label="fit")
            plt.fill_between(dates, lo, hi, alpha=0.2, label="95% CI")
            plt.legend()
            plt.title("ARIMAX in-sample")
            plt.tight_layout()
            plt.savefig(cfg.out_dir / "arimax_fit_ci.png", dpi=150)
            plt.close()

        log("Notebook-like evaluation finished.")
        return

    # --- Pipeline padrão com split + baselines + fair compare + grid ---
    with step("Train/Test split + baselines"):
        train_df = df_model.iloc[: int(len(df_model) * cfg.split_ratio)].copy()
        test_df = df_model.iloc[int(len(df_model) * cfg.split_ratio) :].copy()

        y_train = train_df[target].astype(float).values
        y_test = test_df[target].astype(float).values
        X_train = None if not all_exog else train_df[all_exog].astype(float).values
        X_test = None if not all_exog else test_df[all_exog].astype(float).values

        dates_test = pd.to_datetime(test_df[date_col])

        if X_train is not None and X_train.shape[1] > 0:
            log("Fitting baseline models (Linear, RF, HGB)…")
            compare_baselines(X_train, y_train, X_test, y_test, cfg.out_dir)
        else:
            log("No exogenous features available; skipping baselines.")

        if cfg.fair_compare:
            log("Running FAIR COMPARISON: ARIMAX vs baselines (one-step-ahead TEST)…")
            fair_compare_arimax_vs_baselines(
                dates_test=dates_test,
                y_train=y_train,
                y_test=y_test,
                X_train=X_train,
                X_test=X_test,
                order_fc=cfg.order_fc,
                seasonal_order_fc=cfg.seasonal_order_fc,
                out_dir=cfg.out_dir,
            )

    best_rows = []
    with step("SARIMAX grid search"):
        combos = []
        for s in cfg.seasonal_periods:
            for p, d, q in itertools.product(cfg.p, cfg.d, cfg.q):
                for P, D, Q in itertools.product(cfg.P, cfg.D, cfg.Q):
                    combos.append(((p, d, q), (P, D, Q, s)))
        total = len(combos)
        if cfg.limit_combos and cfg.limit_combos < total:
            log(f"[info] Limiting combos: {cfg.limit_combos}/{total}")
            combos = combos[: cfg.limit_combos]
            total = len(combos)
        log(f"Grid size: {total} configs")

        t0 = perf_counter()
        for i, (order, sorder) in enumerate(combos, 1):
            tag = f"p{order[0]}d{order[1]}q{order[2]}_P{sorder[0]}D{sorder[1]}Q{sorder[2]}_s{sorder[3]}"
            if cfg.log_interval and (i % cfg.log_interval == 0 or i == 1):
                dt = perf_counter() - t0
                eta = dt / i * (total - i) if i > 0 else float("nan")
                log(
                    f"[{i}/{total}] Fitting {tag} … (elapsed {dt/60:.1f} min | ETA {eta/60:.1f} min)"
                )

            try:
                res = fit_sarimax(
                    y_train, X_train, order, sorder, cfg.maxiter, cfg.light_sarimax
                )
                try:
                    ddir = cfg.out_dir / f"diag_{tag}"
                    ensure_dir(ddir)
                    residual_diagnostics(res.resid, ddir)
                except Exception:
                    pass
                metrics = evaluate_sarimax_one_step(
                    res, y_train, X_train, y_test, X_test, cfg.out_dir, tag
                )
                row = {
                    "config": tag,
                    **metrics,
                    "AIC": float(getattr(res, "aic", np.nan)),
                }
                best_rows.append(row)
                log(
                    f" → {tag}: Test RMSE={row['RMSE']:.4f} | MAE={row['MAE']:.4f} | R2={row['R2']:.4f} | AIC={row['AIC']:.1f}"
                )
            except Exception as e:
                log(f"[skip] {tag} failed: {e}")

    with step("Save summary & build report"):
        if best_rows:
            df_best = pd.DataFrame(best_rows).sort_values("RMSE").reset_index(drop=True)
            save_csv(df_best, cfg.out_dir / "sarimax_grid_summary.csv")
        meta = {
            "CSV": str(cfg.csv),
            "Date column": date_col,
            "Target": target,
            "Exogenous": ", ".join(all_exog) if all_exog else "(none)",
            "Train/Test split": f"{cfg.split_ratio:.2f}/{1-cfg.split_ratio:.2f}",
            "Seasonal periods tried": ", ".join(map(str, cfg.seasonal_periods)),
            "Grid sizes": f"p={len(cfg.p)}, d={len(cfg.d)}, q={len(cfg.q)}, P={len(cfg.P)}, D={len(cfg.D)}, Q={len(cfg.Q)}",
            "Combos evaluated": str(len(best_rows)),
            "Fair compare": str(cfg.fair_compare),
            "Fair compare order": str(cfg.order_fc),
            "Fair compare seasonal": str(cfg.seasonal_order_fc),
            "Eval mode": cfg.eval_mode,
        }
        build_report(cfg.out_dir, meta, best_rows)

    log("All done.")


if __name__ == "__main__":
    main()
