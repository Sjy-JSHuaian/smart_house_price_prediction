"""
train_linear_regression.py

训练线性回归模型（基线模型）。

流程：
  加载标准化数据 → 训练 LinearRegression → 评估指标 → 系数分析 → 保存模型
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.data.preprocess import MODELS_DIR, load_preprocessor

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
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
        "feature_names": fnames,
    }


# ── 评估函数 ────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """计算回归指标：MSE, RMSE, MAE, R^2。"""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    if label:
        print(f"  [{label}]  MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R^2={r2:.4f}")

    return metrics


# ── 训练函数 ────────────────────────────────────────────────

def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LinearRegression:
    """训练线性回归模型。"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ── 系数分析 ────────────────────────────────────────────────

def print_coefficients(model: LinearRegression, feature_names: list) -> pd.Series:
    """
    打印并返回按绝对值排序的系数 Series。

    系数含义（数据已标准化）：每增加 1 个标准差的特征值，房价变动系数个单位。
    """
    coef_series = pd.Series(model.coef_, index=feature_names).sort_values(key=abs, ascending=False)

    print("\n" + "=" * 60)
    print(">>> 特征系数（按重要性排序）")
    print("-" * 60)
    print(f"{'Feature':<16s} {'Coefficient':>12s}  {'Impact':>20s}")
    print("-" * 60)
    for feat, coef in coef_series.items():
        direction = "pushes price UP  " if coef > 0 else "pushes price DOWN"
        print(f"  {feat:<14s} {coef:>+12.6f}  {direction:>20s}")
    print("-" * 60)
    print(f"  {'Intercept':<14s} {model.intercept_:>+12.6f}")
    print("=" * 60)

    return coef_series


# ── 预测 vs 真实值散点图 ──────────────────────────────────

def plot_predictions(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    r2_train: float,
    r2_test: float,
) -> Path:
    """绘制预测值 vs 真实值散点图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_true, y_pred, r2_val, title in [
        (axes[0], y_train, y_train_pred, r2_train, "Train"),
        (axes[1], y_test,  y_test_pred,  r2_test,  "Test"),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, c="steelblue", edgecolor="none")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("True Value ($100,000)")
        ax.set_ylabel("Predicted Value ($100,000)")
        ax.set_title(f"{title}  (R^2 = {r2_val:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Linear Regression — Predictions vs True Values", fontsize=13)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "lr_predictions.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


# ── 完整训练流水线 ────────────────────────────────────────

def run_training_pipeline(
    save_model: bool = True,
    model_name: str = "linear_regression",
) -> dict:
    """
    一键执行：加载数据 → 训练 → 评估 → 系数分析 → 保存模型。

    Returns
    -------
    dict — model, metrics, coefficients, output_files
    """
    # 1. 加载数据
    print("[1/5] 加载标准化数据 ...")
    data = load_scaled_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    feature_names = data["feature_names"]
    print(f"    训练集: {X_train.shape[0]:,} samples")

    # 2. 训练
    print("[2/5] 训练 LinearRegression ...")
    model = train_linear_regression(X_train, y_train)

    # 3. 评估
    print("[3/5] 评估:")
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    y_val_pred   = model.predict(X_val) if X_val is not None else None

    train_metrics = evaluate(y_train, y_train_pred, "Train")
    if y_val_pred is not None:
        val_metrics = evaluate(y_val, y_val_pred, "Val  ")
    test_metrics  = evaluate(y_test,  y_test_pred,  "Test ")

    # 交叉验证
    print("\n  >>> 5-Fold Cross-Validation (R^2) ...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"     CV R^2 scores: {cv_scores.round(4)}")
    print(f"     CV R^2 mean  : {cv_scores.mean():.4f}  (+/- {cv_scores.std() * 2:.4f})")

    # 4. 系数分析
    print("[4/5] 系数分析 ...")
    coef_series = print_coefficients(model, feature_names)

    # 5. 预测图 & 保存模型
    print("\n[5/5] 绘制预测图 & 保存模型 ...")
    plot_path = plot_predictions(
        y_train, y_train_pred,
        y_test,  y_test_pred,
        train_metrics["R2"], test_metrics["R2"],
    )
    print(f"    预测图已保存: {plot_path}")

    output_files = {"figure": plot_path}

    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"    模型已保存: {model_path}")
        output_files["model"] = model_path

    return {
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics if y_val_pred is not None else None,
        "test_metrics": test_metrics,
        "cv_scores": cv_scores,
        "coefficients": coef_series,
        "output_files": output_files,
    }


# ── CLI 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_training_pipeline()

    # 最终判定
    print()
    print("=" * 60)
    print(">>> 最终评估报告")
    r2_train = result["train_metrics"]["R2"]
    r2_test  = result["test_metrics"]["R2"]
    r2_gap   = r2_train - r2_test
    print(f"    Train R^2: {r2_train:.4f}")
    print(f"    Test  R^2: {r2_test:.4f}")
    print(f"    R^2 Gap  : {r2_gap:.4f}  ", end="")
    if r2_gap < 0.05:
        print("(无明显过拟合)")
    elif r2_gap < 0.10:
        print("(轻微过拟合，可接受)")
    else:
        print("(存在过拟合，需正则化)")

    # RMSE 解读
    rmse_test = result["test_metrics"]["RMSE"]
    print(f"    Test RMSE: {rmse_test:.4f} (约 ${rmse_test*100000:,.0f})")
    print(f"   -> 模型预测误差约 ±${rmse_test*100000:,.0f}")
    print("=" * 60)
