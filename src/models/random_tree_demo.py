"""
random_tree_demo.py

训练随机森林回归模型，对比不同超参数的组合，选出最优模型。

随机森林通过 Bagging + 随机特征子集降低方差，天然抗过拟合。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.preprocess import MODELS_DIR

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# ── 加载数据 ──────────────────────────────────────────────

def load_scaled_data() -> dict:
    def _load(name: str) -> tuple:
        path = PROCESSED_DIR / f"Xy_{name}_scaled.csv"
        if not path.exists():
            raise FileNotFoundError(f"标准化数据不存在: {path}")
        df = pd.read_csv(path)
        feats = [c for c in df.columns if c != "MedHouseVal"]
        return df[feats].values, df["MedHouseVal"].values, feats

    X_train, y_train, fnames = _load("train")
    X_test,  y_test,  _      = _load("test")
    X_val = y_val = None
    if (PROCESSED_DIR / "Xy_val_scaled.csv").exists():
        X_val, y_val, _ = _load("val")

    return {"X_train": X_train, "y_train": y_train, "X_val": X_val,
            "y_val": y_val, "X_test": X_test, "y_test": y_test,
            "feature_names": fnames}


def evaluate(y_true, y_pred, label=""):
    mse = mean_squared_error(y_true, y_pred)
    m = {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(y_true, y_pred),
         "R2": r2_score(y_true, y_pred)}
    if label:
        print(f"  [{label}]  MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R^2={m['R2']:.4f}")
    return m


# ── 超参数调优 ──────────────────────────────────────────

def tune_hyperparams(
    X_train, y_train, X_val, y_val, random_state=42,
) -> pd.DataFrame:
    """
    网格搜索 n_estimators × max_depth，基于验证集 R^2 选最优组合。
    """
    n_estimators_list = [50, 100, 200]
    max_depth_list = [5, 8, 10, 12, 15, None]

    rows = []
    for n_est in n_estimators_list:
        for md in max_depth_list:
            rf = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=md,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            )
            rf.fit(X_train, y_train)
            rows.append({
                "n_estimators": n_est,
                "max_depth": str(md),
                "train_R2": r2_score(y_train, rf.predict(X_train)),
                "val_R2":   r2_score(y_val,   rf.predict(X_val)),
                "val_RMSE": np.sqrt(mean_squared_error(y_val, rf.predict(X_val))),
            })

    return pd.DataFrame(rows)


def print_tuning_table(df):
    best_idx = df["val_R2"].idxmax()
    print(f"\n{'n_est':>7s} {'depth':>7s} {'train_R2':>10s} {'val_R2':>10s} {'val_RMSE':>12s}")
    print("-" * 55)
    for _, row in df.iterrows():
        m = "  <-- BEST" if _ == best_idx else ""
        print(f"{row['n_estimators']:7d} {row['max_depth']:>7s} {row['train_R2']:10.4f} {row['val_R2']:10.4f} {row['val_RMSE']:12.4f}{m}")
    print("-" * 55)


# ── 可视化 ──────────────────────────────────────────────

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    order = np.argsort(importances)
    names = [feature_names[i] for i in order]
    imps  = importances[order]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdBu_r(imps / imps.max())
    bars = plt.barh(range(len(names)), imps, color=colors)
    plt.yticks(range(len(names)), names)
    plt.xlabel("Feature Importance (MDI)")
    plt.title("Random Forest — Feature Importance")
    for bar, val in zip(bars, imps):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "rf_feature_importance.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


def plot_predictions(y_train, y_pred_train, y_test, y_pred_test, r2_tr, r2_te, n_est, md):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, yt, yp, r2v, title in [
        (axes[0], y_train, y_pred_train, r2_tr, "Train"),
        (axes[1], y_test,  y_pred_test,  r2_te, "Test"),
    ]:
        ax.scatter(yt, yp, alpha=0.3, s=8, c="darkorange", edgecolor="none")
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
        ax.set_xlabel("True Value ($100,000)")
        ax.set_ylabel("Predicted Value ($100,000)")
        ax.set_title(f"{title}  (R^2 = {r2v:.4f})")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Random Forest (n={n_est}, depth={md}) — Predictions vs True Values", fontsize=13)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "rf_predictions.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


# ── 训练流水线 ──────────────────────────────────────────

def run_training_pipeline(save_model=True, model_name="random_forest"):
    # 1. 加载
    print("[1/5] 加载标准化数据 ...")
    data = load_scaled_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    fnames = data["feature_names"]
    print(f"    训练集: {X_train.shape[0]:,} samples")

    # 2. 调优
    print("[2/5] 网格搜索 n_estimators x max_depth ...")
    df_tune = tune_hyperparams(X_train, y_train, X_val, y_val)
    print_tuning_table(df_tune)
    best = df_tune.loc[df_tune["val_R2"].idxmax()]
    n_est, md_str = int(best["n_estimators"]), best["max_depth"]
    md = None if md_str == "None" else int(md_str)
    print(f"\n  >>> 最优: n_estimators={n_est}, max_depth={md_str}  (val R^2={best['val_R2']:.4f})")

    # 3. 训练
    print("[3/5] 训练最优 RandomForest ...")
    model = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=md,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 4. 评估
    print("[4/5] 评估:")
    y_tr_pred = model.predict(X_train)
    y_va_pred = model.predict(X_val) if X_val is not None else None
    y_te_pred = model.predict(X_test)
    tr_m = evaluate(y_train, y_tr_pred, "Train")
    va_m = evaluate(y_val,   y_va_pred, "Val  ") if y_va_pred is not None else None
    te_m = evaluate(y_test,  y_te_pred, "Test ")

    # 5. 图 + 保存
    print("[5/5] 绘图 & 保存 ...")
    imp_path = plot_feature_importance(model, fnames)
    pred_path = plot_predictions(
        y_train, y_tr_pred, y_test, y_te_pred,
        tr_m["R2"], te_m["R2"], n_est, md_str,
    )
    out = {"importance_fig": imp_path, "prediction_fig": pred_path}
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        mp = MODELS_DIR / f"{model_name}.pkl"
        with open(mp, "wb") as f:
            pickle.dump(model, f)
        print(f"    模型已保存: {mp}")
        out["model"] = mp

    return {
        "model": model, "best_params": {"n_estimators": n_est, "max_depth": md},
        "tune_results": df_tune, "train_metrics": tr_m, "val_metrics": va_m,
        "test_metrics": te_m, "output_files": out,
    }


# ── CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_training_pipeline()

    print()
    print("=" * 65)
    print(">>> 三模型对比")
    print("-" * 65)
    print(f"  {'Model':<22s} {'Test R^2':>10s} {'Test RMSE':>12s}")
    print(f"  {'-'*46}")
    print(f"  {'Linear Regression':<22s} {'0.5758':>10s} {'$74,561':>12s}")
    print(f"  {'Decision Tree':<22s} {'0.6849':>10s} {'$64,260':>12s}")
    rf_r2  = result["test_metrics"]["R2"]
    rf_rmse = result["test_metrics"]["RMSE"] * 100000
    print(f"  {'Random Forest':<22s} {rf_r2:10.4f} ${rf_rmse:>11,.0f}")
    print("=" * 65)
