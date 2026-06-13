"""
decision_tree_demo.py

训练决策树回归模型，对比不同深度的表现，选出最优模型。

对比对象：线性回归（基线）vs 决策树（非线性）。
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.data.preprocess import MODELS_DIR

# ── 路径常量 ──────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# ── 加载标准化数据 ──────────────────────────────────────────

def load_scaled_data() -> dict:
    """从 data/processed/ 加载标准化后的拆分数据。"""
    def _load(name: str) -> tuple:
        path = PROCESSED_DIR / f"Xy_{name}_scaled.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"标准化数据不存在: {path}\n请先运行: python src/data/preprocess.py"
            )
        df = pd.read_csv(path)
        feature_cols = [c for c in df.columns if c != "MedHouseVal"]
        return df[feature_cols].values, df["MedHouseVal"].values, feature_cols

    X_train, y_train, fnames = _load("train")
    X_test,  y_test,  _      = _load("test")

    val_path = PROCESSED_DIR / "Xy_val_scaled.csv"
    X_val, y_val = None, None
    if val_path.exists():
        X_val, y_val, _ = _load("val")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_names": fnames,
    }


# ── 评估函数 ────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """计算回归指标。"""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    if label:
        print(f"  [{label}]  MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R^2={r2:.4f}")

    return metrics


# ── 深度调优 ────────────────────────────────────────────────

def tune_max_depth(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    depths: list = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    遍历 max_depth，基于验证集选出最优深度。

    Returns
    -------
    pd.DataFrame — 每行一个 depth 的 train/val 指标。
    """
    if depths is None:
        depths = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]

    rows = []
    for d in depths:
        tree = DecisionTreeRegressor(
            max_depth=d,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
        )
        tree.fit(X_train, y_train)

        y_tr_pred = tree.predict(X_train)
        y_va_pred = tree.predict(X_val)

        rows.append({
            "max_depth": str(d),
            "train_R2":   r2_score(y_train, y_tr_pred),
            "train_RMSE": np.sqrt(mean_squared_error(y_train, y_tr_pred)),
            "val_R2":     r2_score(y_val, y_va_pred),
            "val_RMSE":   np.sqrt(mean_squared_error(y_val, y_va_pred)),
            "n_leaves":   tree.get_n_leaves(),
        })

    return pd.DataFrame(rows)


def print_depth_table(df_result: pd.DataFrame):
    """打印深度调优表格并高亮最优行。"""
    best_idx = df_result["val_R2"].idxmax()
    print("\n" + "=" * 80)
    print(f"{'max_depth':>10s} {'train_R2':>10s} {'train_RMSE':>12s} {'val_R2':>10s} {'val_RMSE':>12s} {'n_leaves':>10s}")
    print("-" * 80)
    for _, row in df_result.iterrows():
        marker = "  <-- BEST" if _ == best_idx else ""
        print(
            f"{row['max_depth']:>10s} {row['train_R2']:10.4f} {row['train_RMSE']:12.4f} "
            f"{row['val_R2']:10.4f} {row['val_RMSE']:12.4f} {row['n_leaves']:10d}{marker}"
        )
    print("=" * 80)


# ── 特征重要性图 ──────────────────────────────────────────

def plot_feature_importance(
    model: DecisionTreeRegressor,
    feature_names: list,
) -> Path:
    """绘制特征重要性柱状图。"""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_imps  = importances[order]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdBu_r(sorted_imps / sorted_imps.max())
    bars = plt.barh(range(len(sorted_names))[::-1], sorted_imps[::-1], color=colors[::-1])
    plt.yticks(range(len(sorted_names))[::-1], sorted_names[::-1])
    plt.xlabel("Feature Importance")
    plt.title("Decision Tree — Feature Importance")
    for bar, val in zip(bars, sorted_imps[::-1]):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "dt_feature_importance.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


# ── 预测 vs 真实散点图 ──────────────────────────────────

def plot_predictions(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    r2_train: float,
    r2_test: float,
    best_depth: str,
) -> Path:
    """绘制预测值 vs 真实值散点图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_true, y_pred, r2_val, title in [
        (axes[0], y_train, y_train_pred, r2_train, "Train"),
        (axes[1], y_test,  y_test_pred,  r2_test,  "Test"),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, c="forestgreen", edgecolor="none")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("True Value ($100,000)")
        ax.set_ylabel("Predicted Value ($100,000)")
        ax.set_title(f"{title}  (R^2 = {r2_val:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Decision Tree (max_depth={best_depth}) — Predictions vs True Values", fontsize=13)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "dt_predictions.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


# ── 完整训练流水线 ────────────────────────────────────────

def run_training_pipeline(
    save_model: bool = True,
    model_name: str = "decision_tree",
) -> dict:
    """
    一键执行：加载数据 → 深度调优 → 训练最优模型 → 评估 → 保存。
    """
    # 1. 加载数据
    print("[1/6] 加载标准化数据 ...")
    data = load_scaled_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    feature_names = data["feature_names"]
    print(f"    训练集: {X_train.shape[0]:,}  samples")

    # 2. 深度调优
    print("[2/6] 遍历 max_depth，基于验证集选择最优深度 ...")
    df_depth = tune_max_depth(X_train, y_train, X_val, y_val)
    print_depth_table(df_depth)

    best_row = df_depth.loc[df_depth["val_R2"].idxmax()]
    best_depth_str = best_row["max_depth"]
    best_depth = None if best_depth_str == "None" else int(best_depth_str)
    print(f"\n  >>> 最优 max_depth = {best_depth_str}  (val R^2 = {best_row['val_R2']:.4f})")

    # 3. 训练最优模型
    print("[3/6] 使用最优深度训练最终模型 ...")
    model = DecisionTreeRegressor(
        max_depth=best_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"    叶子节点数: {model.get_n_leaves()}")

    # 4. 评估
    print("[4/6] 评估:")
    y_train_pred = model.predict(X_train)
    y_val_pred   = model.predict(X_val) if X_val is not None else None
    y_test_pred  = model.predict(X_test)

    train_metrics = evaluate(y_train, y_train_pred, "Train")
    if y_val_pred is not None:
        val_metrics = evaluate(y_val, y_val_pred, "Val  ")
    test_metrics  = evaluate(y_test,  y_test_pred,  "Test ")

    # 5. 特征重要性 & 预测图
    print("[5/6] 绘制特征重要性 & 预测图 ...")
    imp_path = plot_feature_importance(model, feature_names)
    print(f"    特征重要性图: {imp_path}")

    pred_path = plot_predictions(
        y_train, y_train_pred,
        y_test,  y_test_pred,
        train_metrics["R2"], test_metrics["R2"],
        best_depth_str,
    )
    print(f"    预测图: {pred_path}")

    # 6. 保存模型
    output_files = {"importance_fig": imp_path, "prediction_fig": pred_path}

    if save_model:
        print("[6/6] 保存模型 ...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"    模型已保存: {model_path}")
        output_files["model"] = model_path

    return {
        "model": model,
        "best_depth": best_depth,
        "depth_results": df_depth,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics if y_val_pred is not None else None,
        "test_metrics": test_metrics,
        "output_files": output_files,
    }


# ── CLI 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_training_pipeline()

    # 最终判定 + 与线性回归对比
    print()
    print("=" * 60)
    print(">>> 决策树 vs 线性回归 对比")
    print("-" * 60)
    r2_dt = result["test_metrics"]["R2"]
    r2_lr = 0.5758  # 来自上一轮线性回归的训练结果
    rmse_dt = result["test_metrics"]["RMSE"]

    print(f"  {'Model':<20s} {'Test R^2':>10s} {'Test RMSE':>12s}")
    print(f"  {'-'*42}")
    print(f"  {'Linear Regression':<20s} {r2_lr:10.4f} {'0.7456':>12s}")
    print(f"  {'Decision Tree':<20s} {r2_dt:10.4f} {rmse_dt:12.4f}")
    delta_r2 = r2_dt - r2_lr
    direction = "better" if delta_r2 > 0 else "worse"
    print(f"  {'Delta':<20s} {delta_r2:+10.4f}  (DT is {direction} than LR)")
    print("=" * 60)
