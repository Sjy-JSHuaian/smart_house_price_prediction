"""
preprocess.py

数据标准化 / 预处理流水线。

基于 EDA 分析结论，对不同特征分组采用不同的缩放策略：
  - StandardScaler（均值 0 / 标准差 1）→ MedInc, HouseAge, Latitude, Longitude
  - RobustScaler（中位数 / IQR）       → AveRooms, AveBedrms, Population, AveOccup

设计原则：
  1. 仅在训练集上 fit，避免数据泄漏。
  2. 支持设置 random_state，保证可复现。
  3. 预处理后数据可直接喂入模型，无需再做缩放。
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import pickle

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

from src.features.split_features_target import load_splits, save_splits

# ── 路径常量 ──────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ── 分组配置（基于 EDA 结论）─────────────────────────────────
STANDARD_FEATURES = ["MedInc", "HouseAge", "Latitude", "Longitude"]
ROBUST_FEATURES   = ["AveRooms", "AveBedrms", "Population", "AveOccup"]

# 验证：两组应覆盖全部 8 个特征
_ALL_FEATURES = set(STANDARD_FEATURES + ROBUST_FEATURES)
assert len(_ALL_FEATURES) == 8, f"特征分组未覆盖全部 8 列，当前 {len(_ALL_FEATURES)} 列"


def build_preprocessor() -> ColumnTransformer:
    """
    构建 ColumnTransformer：StandardScaler + RobustScaler 混合流水线。

    Returns
    -------
    ColumnTransformer（未 fit）。
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("standard", StandardScaler(), STANDARD_FEATURES),
            ("robust",   RobustScaler(),   ROBUST_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def fit_preprocessor(
    X_train: np.ndarray,
    feature_names: list,
) -> ColumnTransformer:
    """
    在训练集上 fit 预处理器。

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    feature_names : list of str

    Returns
    -------
    ColumnTransformer（已 fit）。后续可对 val / test 直接 .transform()。
    """
    preprocessor = build_preprocessor()
    df_train = pd.DataFrame(X_train, columns=feature_names)
    preprocessor.fit(df_train)
    return preprocessor


def transform(
    preprocessor: ColumnTransformer,
    X: np.ndarray,
    feature_names: list,
) -> np.ndarray:
    """对 X 应用已 fit 的预处理器，返回标准化后的 np.ndarray。"""
    df = pd.DataFrame(X, columns=feature_names)
    return preprocessor.transform(df)


def fit_transform_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    fit + transform 训练集，一步完成。

    Returns
    -------
    X_train_scaled, y_train, preprocessor
    """
    preprocessor = fit_preprocessor(X_train, feature_names)
    X_train_scaled = transform(preprocessor, X_train, feature_names)
    return X_train_scaled, y_train, preprocessor


# ── 持久化 ──────────────────────────────────────────────────

def save_preprocessor(preprocessor: ColumnTransformer, path: Optional[Path] = None) -> Path:
    """保存已 fit 的预处理器为 .pkl 文件。"""
    path = path or (MODELS_DIR / "preprocessor.pkl")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    return path


def load_preprocessor(path: Optional[Path] = None) -> ColumnTransformer:
    """加载已保存的预处理器。"""
    path = path or (MODELS_DIR / "preprocessor.pkl")
    if not path.exists():
        raise FileNotFoundError(f"预处理器文件不存在: {path}，请先运行 fit + save")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_scaled_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    保存标准化后的数据到 data/processed/（文件名为 Xy_*_scaled.csv）。
    """
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    def _save(name: str, X_arr: np.ndarray, y_arr: np.ndarray):
        df_out = pd.DataFrame(X_arr, columns=feature_names)
        df_out["MedHouseVal"] = y_arr
        path = output_dir / f"{name}_scaled.csv"
        df_out.to_csv(path, index=False, float_format="%.6f")
        saved[name] = path

    _save("Xy_train", X_train, y_train)
    _save("Xy_test", X_test, y_test)
    if X_val is not None and y_val is not None:
        _save("Xy_val", X_val, y_val)

    return saved


# ── 完整流水线（一键从原始拆分到标准化）────────────────────

def run_full_pipeline(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    一键执行完整预处理流水线：
      加载原始数据 → 拆分 → fit 预处理器 → transform → 保存标准化数据 + 预处理器。

    Returns
    -------
    dict — 包含各数组和路径的汇总字典。
    """
    # 1. 加载已拆分的原始数据（若不存在则自动生成）
    try:
        X_train, X_test, y_train, y_test, X_val, y_val, feature_names = load_splits()
        print("[1/4] 加载已拆分的原始数据 ...")
    except FileNotFoundError:
        print("[1/4] 拆分文件不存在，重新生成 ...")
        from src.features.split_features_target import split_xy, train_test_split_xy
        X, y, feature_names = split_xy()
        X_train, X_test, y_train, y_test, X_val, y_val = train_test_split_xy(
            X, y, test_size=test_size, val_size=val_size, random_state=random_state
        )
        save_splits(X_train, X_test, y_train, y_test, feature_names,
                     X_val=X_val, y_val=y_val)
        print("    已保存至 data/processed/")

    # 2. fit + transform
    print("[2/4] fit 预处理器（Standard + Robust）并 transform 训练集 ...")
    X_train_scaled, y_train, preprocessor = fit_transform_train(
        X_train, y_train, feature_names
    )

    # 3. transform val / test
    print("[3/4] transform 验证集 & 测试集（不 re-fit）...")
    X_test_scaled = transform(preprocessor, X_test, feature_names)
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = transform(preprocessor, X_val, feature_names)

    # 4. 保存
    print("[4/4] 保存标准化数据 & 预处理器 ...")
    scaled_paths = save_scaled_data(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names,
        X_val=X_val_scaled, y_val=y_val,
    )
    for name, p in scaled_paths.items():
        print(f"    {p.name:20s} -> {p}")
    pkl_path = save_preprocessor(preprocessor)
    print(f"    {pkl_path.name:20s} -> {pkl_path}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_val": X_val_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "preprocessor_path": pkl_path,
    }


# ── CLI 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    result = run_full_pipeline()

    # 校验 —— 注意 ColumnTransformer 输出列序 = Standard 组在前，Robust 组在后
    output_order = STANDARD_FEATURES + ROBUST_FEATURES
    X_tr = result["X_train"]
    X_te = result["X_test"]

    print()
    print("=" * 60)
    print(">>> 校验标准化效果（按 transformer 输出列序）")

    # StandardScaler 列：前 4 列
    n_std = len(STANDARD_FEATURES)
    means = X_tr[:, :n_std].mean(axis=0)
    stds  = X_tr[:, :n_std].std(axis=0)
    for i, feat in enumerate(STANDARD_FEATURES):
        print(f"    {feat:<14s}  mean={means[i]:+.4f}  std={stds[i]:.4f}  (目标: mean~0, std~1)")

    # RobustScaler 列：后 4 列
    medians = np.median(X_tr[:, n_std:], axis=0)
    q1 = np.percentile(X_tr[:, n_std:], 25, axis=0)
    q3 = np.percentile(X_tr[:, n_std:], 75, axis=0)
    iqrs = q3 - q1
    for i, feat in enumerate(ROBUST_FEATURES):
        print(f"    {feat:<14s}  median={medians[i]:+.4f}  IQR={iqrs[i]:.4f}  (目标: median~0, IQR~1)")

    # 测试集 vs 训练集 均值对比（无数据泄漏检查）
    print()
    print(">>> 测试集均值（应与训练集接近，不应出现大幅偏移）")
    for i, feat in enumerate(output_order):
        train_mean = X_tr[:, i].mean()
        test_mean  = X_te[:, i].mean()
        drift = abs(test_mean - train_mean)
        flag = " [WARN]" if drift > 0.1 else ""
        print(f"    {feat:<14s}  train={train_mean:+.4f}  test={test_mean:+.4f}  drift={drift:.4f}{flag}")

    print("=" * 60)
    print("\n预处理完成。")
