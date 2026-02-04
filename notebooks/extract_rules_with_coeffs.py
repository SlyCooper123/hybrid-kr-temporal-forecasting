#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_rules_with_coeffs.py
Generates rules (RuleFit) and estimates coefficients directly from the TARGET (consumo_gwh).

Key fixes:
- Always passes NumPy arrays to rf.fit (prevents the error (slice(None,None,None), 0))
- Sparse activation matrix (CSC) -> much lower memory usage
- LassoCV/ElasticNetCV with precompute=False; alphas=100 (silences FutureWarning)

"""

import argparse, re, sys, time
from time import perf_counter
import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------------
# Logging
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
# IO helpers
# ------------------------------
def read_csv_robust(path: Path) -> pd.DataFrame:
    trials = [(e, s) for e in ("utf-8", "latin1") for s in (",", ";", "\t")]
    last = None
    for enc, sep in trials:
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep)
            if df.shape[1] >= 6 and df.shape[0] >= 100:
                log(f"Parsed CSV enc={enc} sep='{sep}' shape={df.shape}")
                return df
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not parse CSV '{path}': {last}")


# ------------------------------
# Rule helpers
# ------------------------------
def compute_support(
    rule_series: pd.Series, X: pd.DataFrame, log_interval: int = 50
) -> pd.Series:
    sup, n = [], len(rule_series)
    t0 = perf_counter()
    for j, rule in enumerate(rule_series.astype(str), 1):
        if log_interval and (j % log_interval == 0 or j == 1):
            dt = perf_counter() - t0
            log(f"[support] processing rule {j}/{n} (elapsed {dt:.1f}s)")
        mask = np.ones(len(X), dtype=bool)
        for part in re.split(r"\s*&\s*", rule):
            m = re.match(r"\s*([\w_]+)\s*(<=|>=|<|>)\s*([-+]?\d+(\.\d+)?)\s*", part)
            if m:
                feat, op, thr = m.group(1), m.group(2), float(m.group(3))
                if feat not in X.columns:
                    mask &= False
                else:
                    s = pd.to_numeric(X[feat], errors="coerce")
                    if op == "<":
                        mask &= s < thr
                    elif op == "<=":
                        mask &= s <= thr
                    elif op == ">":
                        mask &= s > thr
                    elif op == ">=":
                        mask &= s >= thr
            else:
                m2 = re.match(r"\s*([\w_]+)\s*==\s*([01])\s*", part)
                if m2:
                    feat, val = m2.group(1), int(m2.group(2))
                    if feat not in X.columns:
                        mask &= False
                    else:
                        mask &= pd.to_numeric(X[feat], errors="coerce") == val
        sup.append(mask.mean())
    return pd.Series(sup)


def build_activation_matrix_sparse(
    rules: pd.Series, X: pd.DataFrame, log_interval: int = 25
):
    """
    Retorna matriz CSC esparsa (n_samples x n_rules) com 0/1 indicando ativação da regra.
    Usa layout por coluna (CSC) para construir de forma eficiente.
    """
    from scipy.sparse import csc_matrix

    n_rows = len(X)
    n_cols = len(rules)

    indptr = [0]
    indices = []
    data = []

    t0 = perf_counter()
    for j, rule in enumerate(rules.astype(str), 1):
        if log_interval and (j % log_interval == 0 or j == 1):
            dt = perf_counter() - t0
            log(f"[activation] building col {j}/{n_cols} (elapsed {dt:.1f}s)")

        mask = np.ones(n_rows, dtype=bool)
        for part in re.split(r"\s*&\s*", rule):
            m = re.match(r"\s*([\w_]+)\s*(<=|>=|<|>)\s*([-+]?\d+(\.\d+)?)\s*", part)
            if m:
                feat, op, thr = m.group(1), m.group(2), float(m.group(3))
                s = pd.to_numeric(X.get(feat, np.nan), errors="coerce").to_numpy()
                if op == "<":
                    mask &= s < thr
                elif op == "<=":
                    mask &= s <= thr
                elif op == ">":
                    mask &= s > thr
                elif op == ">=":
                    mask &= s >= thr
            else:
                m2 = re.match(r"\s*([\w_]+)\s*==\s*([01])\s*", part)
                if m2:
                    feat, val = m2.group(1), int(m2.group(2))
                    s = pd.to_numeric(X.get(feat, np.nan), errors="coerce").to_numpy()
                    mask &= s == val

        idx = np.flatnonzero(mask)  # linhas onde a regra é verdadeira
        indices.extend(idx.tolist())
        data.extend([1.0] * len(idx))
        indptr.append(len(indices))

    Z = csc_matrix(
        (
            np.array(data, dtype=np.float64),
            np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int64),
        ),
        shape=(n_rows, n_cols),
        dtype=np.float64,
    )
    return Z


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_meteo_com_consumo.csv")
    ap.add_argument(
        "--target", default=None, help="Target column; auto-detect if not given"
    )
    ap.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Feature columns to seed trees/rules",
    )
    ap.add_argument("--out", default="outputs_knowledge/rulefit_rules_complete.csv")
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--min-support", type=float, default=0.02)
    ap.add_argument("--max-len", type=int, default=3)
    ap.add_argument("--n-est", type=int, default=140)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-leaf", type=int, default=50)
    ap.add_argument("--log-interval", type=int, default=50)
    args = ap.parse_args()

    log("Starting rules extraction pipeline …")
    log(f"Params: csv={args.csv} target={args.target} out={args.out}")
    log(
        f"       max-samples={args.max_samples} min-support={args.min_support} max-len={args.max_len}"
    )
    log(
        f"       n-est={args.n_est} max-depth={args.max_depth} min-leaf={args.min_leaf} log-interval={args.log_interval}"
    )

    try:
        with step("Load & normalize CSV"):
            df = read_csv_robust(Path(args.csv))
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            log(f"Columns: {list(df.columns)[:12]}{' …' if len(df.columns)>12 else ''}")

        with step("Detect target & features"):
            target = args.target
            if not target:
                for cand in [
                    "consumo_gwh",
                    "consumo",
                    "target",
                    "gwh",
                    "consumo_diario_gwh",
                ]:
                    if cand in df.columns:
                        target = cand
                        break
            if not target:
                raise SystemExit("Target column not found; pass --target.")

            feats = [c for c in (args.features or []) if c in df.columns] or [
                c
                for c in ["tmean_c", "hdd18", "cdd22", "rad_solar", "humidade_relativa"]
                if c in df.columns
            ]
            if not feats:
                raise SystemExit("No seed features found; pass --features …")

            rdf = df[[target] + feats].dropna().copy()
            X = rdf[feats].astype(float)
            y = rdf[target].astype(float).values
            log(f"Modeling frame: X={X.shape}, y={y.shape}  (features={feats})")

        with step("Optional subsample for induction"):
            if len(X) > args.max_samples:
                idx = np.random.RandomState(0).choice(
                    len(X), size=args.max_samples, replace=False
                )
                Xfit = X.iloc[idx].copy()
                yfit = y[idx].copy()
                log(f"Subsampled to {Xfit.shape[0]} rows for rule induction.")
            else:
                Xfit = X
                yfit = y
                log("Using full data for rule induction.")

        with step("Fit RuleFit (rule generator)"):
            impl = None
            RuleFitCls = None
            try:
                from rulefit import RuleFit as RFClassic

                RuleFitCls = RFClassic
                impl = "rulefit"
            except Exception:
                try:
                    from imodels import RuleFitRegressor as RFImodels

                    RuleFitCls = RFImodels
                    impl = "imodels"
                except Exception:
                    raise SystemExit(
                        "Neither 'rulefit' nor 'imodels' available. pip install rulefit-py or imodels"
                    )
            log(f"Implementation: {impl}")

            from sklearn.ensemble import RandomForestRegressor

            tree_gen = RandomForestRegressor(
                n_estimators=args.n_est,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_leaf,
                n_jobs=-1,
                random_state=0,
            )
            rf_kwargs = dict(
                tree_generator=tree_gen,
                tree_size=4,
                max_rules=500,
                sample_fract=0.5,
                random_state=0,
            )
            rf = RuleFitCls(
                **{
                    k: v
                    for k, v in rf_kwargs.items()
                    if k in RuleFitCls.__init__.__code__.co_varnames
                }
            )

            # >>> SEMPRE numpy na entrada
            X_fit_np = np.asarray(Xfit, dtype=float)
            rf.fit(X_fit_np, yfit, feature_names=list(Xfit.columns))
            log("RuleFit model fitted.")

        with step("Extract raw rule strings"):
            rule_strs = []
            if impl == "rulefit":
                try:
                    rules_df = rf.get_rules()
                    rule_strs = list(
                        rules_df.loc[rules_df.get("type", "rule") != "linear", "rule"]
                        .astype(str)
                        .values
                    )
                except Exception:
                    rule_strs = []
            if not rule_strs:
                rules_list = getattr(rf, "rules_", None)
                if not rules_list:
                    ens = getattr(rf, "rule_ensemble_", None)
                    rules_list = getattr(ens, "rules", []) if ens is not None else []
                for r in rules_list or []:
                    s = str(getattr(r, "rule", r))
                    if any(op in s for op in ["<", ">", "=="]):
                        rule_strs.append(s)
            log(f"Extracted raw rules: {len(rule_strs)}")
            if not rule_strs:
                raise SystemExit("Could not extract any rules from the model.")

        with step("Compute support & length + filtering"):
            rule_series = pd.Series(rule_strs, dtype="object")
            support = compute_support(rule_series, X, log_interval=args.log_interval)
            length = rule_series.str.count("&") + 1

            keep = (support >= args.min_support) & (length <= args.max_len)
            pre = len(rule_series)
            rule_series = rule_series[keep].reset_index(drop=True)
            support = support[keep].reset_index(drop=True)
            length = length[keep].reset_index(drop=True)
            log(
                f"Filtered rules: kept {len(rule_series)}/{pre} (min_support={args.min_support}, max_len={args.max_len})"
            )
            if len(rule_series) == 0:
                raise SystemExit(
                    "All rules filtered out by thresholds. Relax constraints."
                )

        with step("Build rule activation matrix (SPARSE, FULL X)"):
            Z = build_activation_matrix_sparse(
                rule_series, X, log_interval=args.log_interval
            )
            log(f"Activation matrix shape: {Z.shape}, nnz={Z.nnz}")

        with step("Refit sparse linear model (LassoCV) to assign coefficients"):
            from sklearn.linear_model import LassoCV, ElasticNetCV
            from sklearn.exceptions import ConvergenceWarning
            import warnings

            y64 = np.asarray(y, dtype=np.float64)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    lcv = LassoCV(
                        cv=5,
                        random_state=0,
                        alphas=100,  # silencia FutureWarning do n_alphas
                        max_iter=20000,
                        selection="cyclic",
                        precompute=False,  # evita Gram
                        n_jobs=None,
                    )
                    lcv.fit(Z, y64)
                    coefs = lcv.coef_
                    used = "LassoCV"
            except Exception as e:
                log(f"[warn] LassoCV failed ({e}); falling back to ElasticNetCV.")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    encv = ElasticNetCV(
                        cv=5,
                        random_state=0,
                        alphas=100,
                        l1_ratio=[1.0, 0.95, 0.9],
                        max_iter=20000,
                        selection="cyclic",
                        precompute=False,
                        n_jobs=None,
                    )
                    encv.fit(Z, y64)
                    coefs = encv.coef_
                    used = "ElasticNetCV"

            n_nonzero = int((np.abs(coefs) > 0).sum())
            log(f"{used} done. Non-zero coefficients: {n_nonzero}/{len(coefs)}")

        with step("Assemble & save output"):
            score = np.abs(coefs) * support.values
            out = (
                pd.DataFrame(
                    {
                        "rule": rule_series.values,
                        "coef": coefs,
                        "support": support.values,
                        "len": length.values,
                        "score": score,
                    }
                )
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )

            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
            log(f"Saved rules: {len(out)} → {out_path.resolve()}")

            # Preview top-10
            head = out.head(10)
            log("Top rules (preview):")
            for _, r in head.iterrows():
                direction = (
                    "increase"
                    if r["coef"] > 0
                    else ("decrease" if r["coef"] < 0 else "no-change")
                )
                log(
                    f"- IF {r['rule']} THEN consumption tends to {direction} (coef={r['coef']:.4f}, support={r['support']:.3f})"
                )

        log("Pipeline finished successfully.")

    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl+C). Exiting gracefully.")
        sys.exit(130)
    except Exception as e:
        log(f"FATAL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
