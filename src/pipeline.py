import os, json, itertools, random
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler

from .utils import set_seed, ensure_dir, pick_device
from . import data as data_mod
from . import features as F
from .evaluation import compute_metrics, expanding_r2
from .models.baseline import make_model as make_classical_model
from .models.deep import fit_deep, predict_deep

def sample_params(space: dict, n: int):
    keys = list(space.keys())
    options = []
    for _ in range(n):
        params = {}
        for k in keys:
            v = space[k]
            if isinstance(v, list):
                params[k] = random.choice(v)
            else:
                params[k] = v
        options.append(params)
    return options

def prepare_xy(df: pd.DataFrame, target_horizon=1, dropna_after_build=True):
    # target is return at t (already computed). We shift features so they are known at t-1
    y = df["ret"].shift(-target_horizon)  # predict next-step return
    X = df.drop(columns=["ret"]).shift(1) # features up to t-1
    out = pd.concat({"y": y, "X": X}, axis=1).dropna() if dropna_after_build else pd.concat({"y": y, "X": X}, axis=1)
    y = out["y"]
    X = out["X"]
    return X, y

def run_pipeline(cfg: Dict):
    seed = int(cfg["run"].get("seed", 42))
    set_seed(seed)

    # Load
    dcfg = cfg["data"]
    df = data_mod.load_main(dcfg["main_csv"], dcfg["timestamp_col"], dcfg["price_col"], dcfg.get("freq","D"), dcfg.get("tz","UTC"))
    df = data_mod.compute_returns(df, log_return=dcfg.get("log_return", True))

    # Exogenous
    ecfg = cfg.get("exogenous", {"enabled": False})
    if ecfg.get("enabled", False):
        df = data_mod.merge_exogenous(df, ecfg.get("assets", []))

    # Features
    fcfg = cfg["features"]
    df = F.build_features(df, fcfg)
    if fcfg.get("dropna_after_build", True):
        df = df.dropna()

    # Prepare supervised X, y
    X, y = prepare_xy(df, target_horizon=dcfg.get("target_horizon", 1), dropna_after_build=fcfg.get("dropna_after_build", True))
    n = len(X)
    idx = X.index

    # Walk-forward
    wcfg = cfg["windows"]
    splits = list(data_mod.train_val_test_indices(n, wcfg["train"], wcfg["val"], wcfg["test"], wcfg.get("step", None)))

    out_dir = os.path.join(cfg["outputs"]["out_dir"], cfg["run"]["run_name"])
    ensure_dir(out_dir)

    device = pick_device(cfg["run"].get("device", "auto"))

    results = []
    pred_store = {}
    for model_group in ["classical", "deep"]:
        for m in cfg["models"].get(model_group, []):
            mname = m["name"]
            space = m.get("params", {})
            print(f'current model is {mname}')
            all_preds = np.full(n, np.nan)
            all_truth = np.full(n, np.nan)
            window_metrics = []

            for wnum, (tr, va, te) in enumerate(splits):
                X_tr, y_tr = X.iloc[tr], y.iloc[tr]
                X_va, y_va = X.iloc[va], y.iloc[va]
                X_te, y_te = X.iloc[te], y.iloc[te]

                # Scaling per window (fit on train only, apply to val/test)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)
                X_te_s = scaler.transform(X_te)

                # Tune: sample candidate params and evaluate on validation
                candidates = sample_params(space, int(cfg["tuning"].get("n_candidates", 10))) or [{}]

                best_score = -1e18
                best_obj = None
                best_params = None
                is_deep = (model_group == "deep")

                for cand in candidates:
                    if not is_deep:
                        try:
                            model = make_classical_model(mname, cand)
                        except Exception as e:
                            continue
                        if model is None:  # e.g. xgboost missing
                            continue
                        model.fit(X_tr_s, y_tr.values)
                        val_pred = model.predict(X_va_s)
                        from sklearn.metrics import r2_score
                        score = r2_score(y_va.values, val_pred)
                        if score > best_score:
                            best_score = score
                            best_obj = (model, None, scaler)
                            best_params = cand
                    else:
                        model, lookback = fit_deep(mname, X_tr_s, y_tr.values, X_va_s, y_va.values, cand, device)
                        val_pred = predict_deep(model, np.vstack([X_tr_s, X_va_s]), np.concatenate([y_tr.values, y_va.values]), lookback, device)
                        val_pred = val_pred[len(y_tr)+lookback: ]  # aligns to y_va[lookback:]
                        y_va_aligned = y_va.values[lookback:]
                        if len(y_va_aligned) == 0:
                            continue
                        from sklearn.metrics import r2_score
                        score = r2_score(y_va_aligned, val_pred)
                        if score > best_score:
                            best_score = score
                            best_obj = (model, lookback, scaler)  # we'll refit below anyway
                            best_params = cand

                # Refit best on Train+Val, then test
                X_trva = np.vstack([X_tr_s, X_va_s])
                y_trva = np.concatenate([y_tr.values, y_va.values])

                if model_group == "deep":
                    model, lookback = fit_deep(mname, X_trva, y_trva, X_va_s, y_va.values, best_params or {}, device)
                    te_pred_full = predict_deep(model, np.vstack([X_trva, X_te_s]), np.concatenate([y_trva, y_te.values]), lookback, device)
                    te_pred = te_pred_full[len(y_trva)+lookback: ]
                    y_te_aligned = y_te.values[lookback:]
                    te_idx = idx[te][lookback:]  # aligned timestamps
                else:
                    model = make_classical_model(mname, best_params or {})
                    model.fit(X_trva, y_trva)
                    te_pred = model.predict(X_te_s)
                    y_te_aligned = y_te.values
                    te_idx = idx[te]

                # --------- [liqin: ensure y_te_aligned dimension] -----------
                y_te_aligned = np.asarray(y_te_aligned).reshape(-1)
                # -----------------------------------------------------------

                # Store window preds
                mask_idx = np.isin(idx, te_idx)
                #print(f'mast_idx is {mask_idx}')
                #print(f'y_te_aligned is {y_te_aligned}')
                #print(f'all_truth is {all_truth}')
                all_preds[mask_idx] = te_pred
                all_truth[mask_idx] = y_te_aligned

                # Window metrics
                from .evaluation import compute_metrics
                wm = compute_metrics(y_te_aligned, te_pred)
                window_metrics.append({"window": int(wnum), **wm, "best_params": best_params})

            # Aggregate
            mask = ~np.isnan(all_truth) & ~np.isnan(all_preds)
            from .evaluation import compute_metrics, expanding_r2
            agg_metrics = compute_metrics(all_truth[mask], all_preds[mask]) if mask.any() else {}
            print(agg_metrics)
            results.append({"model": mname, **agg_metrics})

            # Save predictions per model
            df_pred = pd.DataFrame({"timestamp": idx, "y_true": all_truth, "y_pred": all_preds}).dropna()
            df_pred.to_csv(os.path.join(out_dir, f"predictions_{mname}.csv"), index=False)

            # Save metrics per window & aggregate
            pd.DataFrame(window_metrics).to_csv(os.path.join(out_dir, f"window_metrics_{mname}.csv"), index=False)
            with open(os.path.join(out_dir, f"aggregate_metrics_{mname}.json"), "w") as f:
                json.dump({"model": mname, "aggregate": agg_metrics}, f, indent=2)

            # Plots
            if cfg["outputs"].get("make_plots", True) and mask.any():
                import matplotlib.pyplot as plt
                # expanding / cumulative R²
                r2_curve = expanding_r2(all_truth[mask], all_preds[mask])
                plt.figure()
                plt.plot(r2_curve)
                plt.title(f"Expanding R² — {mname}")
                plt.xlabel("Test observations")
                plt.ylabel("R² (expanding)")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"r2_curve_{mname}.png"))
                plt.close()

                # predictions vs actuals
                plt.figure()
                plt.plot(df_pred["timestamp"], df_pred["y_true"], label="true")
                plt.plot(df_pred["timestamp"], df_pred["y_pred"], label="pred")
                plt.title(f"Pred vs True — {mname}")
                plt.xlabel("Time")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"pred_vs_true_{mname}.png"))
                plt.close()

    # Save leaderboard
    #lb = pd.DataFrame(results).sort_values("r2", ascending=False)
    lb = pd.DataFrame(results)
    lb.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)

    return out_dir, lb
