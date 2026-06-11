"""
split_features_target.py

划分特征矩阵 X 和目标向量 y，并提供训练集 / 验证集 / 测试集拆分。
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_california_housing

# ── 输出路径 ──────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def split_xy(
    df: Optional["pd.DataFrame"] = None,
    target: str = "MedHouseVal",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    将 DataFrame 拆分为特征矩阵 X 和标签向量 y。

    Parameters
    ----------
    df : pd.DataFrame, optional
        输入数据，若为 None 则自动加载 California Housing。
    target : str, default="MedHouseVal"
        目标列名称。

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    feature_names : list of str
    """
    if df is None:
        df = load_california_housing(as_frame=True)

    feature_names = [c for c in df.columns if c != target]
    X = df[feature_names].values
    y = df[target].values

    return X, y, feature_names


def train_test_split_xy(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    划分训练集、测试集，以及可选的验证集。

    Parameters
    ----------
    X : np.ndarray, optional
        特征矩阵，若为 None 则自动加载。
    y : np.ndarray, optional
        标签向量，若为 None 则自动加载。
    test_size : float, default=0.2
        测试集比例（占全体）。
    val_size : float, default=0.0
        验证集比例（占全体）。若 > 0 则从训练集中再切分。
    random_state : int, default=42
        随机种子，保证可复现。
    shuffle : bool, default=True
        是否打乱。

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    X_val, y_val : np.ndarray or None
    """
    if X is None or y is None:
        df = load_california_housing(as_frame=True)
        X, y, _ = split_xy(df)

    # 第一步：分出测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    # 第二步：可选地从训练集中再分出验证集
    X_val, y_val = None, None
    if val_size > 0:
        # val_size 原为占全体的比例，换算为占训练集的比例
        val_frac = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_frac,
            random_state=random_state,
            shuffle=shuffle,
        )

    return X_train, X_test, y_train, y_test, X_val, y_val


def print_split_info(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> None:
    """打印各子集的样本数、特征数和目标均值。"""
    total = len(y_train) + len(y_test) + (len(y_val) if y_val is not None else 0)
    print("=" * 60)
    print(f"{'Subset':<12s} {'Samples':>8s} {'Features':>8s} {'Target Mean':>12s}")
    print("-" * 60)
    print(f"{'Train':<12s} {len(y_train):8d} {X_train.shape[1]:8d} {y_train.mean():12.4f}")
    if X_val is not None:
        print(f"{'Validation':<12s} {len(y_val):8d} {X_val.shape[1]:8d} {y_val.mean():12.4f}")
    print(f"{'Test':<12s} {len(y_test):8d} {X_test.shape[1]:8d} {y_test.mean():12.4f}")
    print("-" * 60)
    print(f"{'Total':<12s} {total:8d}")
    print("=" * 60)


# ── 持久化 ──────────────────────────────────────────────────

def save_splits(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    target_name: str = "MedHouseVal",
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    将拆分后的数据集保存为 CSV 文件。

    Returns
    -------
    dict[str, Path] — 文件名 → 文件路径的映射。
    """
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    def _save(name: str, X_arr: np.ndarray, y_arr: np.ndarray):
        df_out = pd.DataFrame(X_arr, columns=feature_names)
        df_out[target_name] = y_arr
        path = output_dir / f"{name}.csv"
        df_out.to_csv(path, index=False)
        saved[name] = path

    _save("Xy_train", X_train, y_train)
    _save("Xy_test", X_test, y_test)
    if X_val is not None and y_val is not None:
        _save("Xy_val", X_val, y_val)

    return saved


def load_splits(
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], list]:
    """
    从 CSV 文件中加载已保存的拆分数据。

    Returns
    -------
    X_train, X_test, y_train, y_test, X_val, y_val, feature_names
    """
    data_dir = data_dir or PROCESSED_DIR

    def _load(name: str) -> Tuple[np.ndarray, np.ndarray, list]:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"拆分文件不存在: {path}，请先运行 save_splits()")
        df_in = pd.read_csv(path)
        feature_names = [c for c in df_in.columns if c != "MedHouseVal"]
        return df_in[feature_names].values, df_in["MedHouseVal"].values, feature_names

    X_train, y_train, fnames = _load("Xy_train")
    X_test, y_test, _ = _load("Xy_test")

    val_path = data_dir / "Xy_val.csv"
    if val_path.exists():
        X_val, y_val, _ = _load("Xy_val")
    else:
        X_val, y_val = None, None

    return X_train, X_test, y_train, y_test, X_val, y_val, fnames


# ── CLI 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(">>> 加载数据并拆分 X / y ...")
    X, y, feature_names = split_xy()
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")
    print(f"    Features: {feature_names}")
    print(f"    y range : [{y.min():.4f}, {y.max():.4f}]")

    print()
    print(">>> 拆分 train / val / test (70% / 10% / 20%) ...")
    X_train, X_test, y_train, y_test, X_val, y_val = train_test_split_xy(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )
    print_split_info(X_train, X_test, y_train, y_test, X_val, y_val)

    # 验证数据无泄漏
    print()
    print(">>> 验证无数据泄漏（train & test 交集为空）...")
    train_set = set(map(tuple, X_train))
    test_set = set(map(tuple, X_test))
    intersection = train_set & test_set
    print(f"    交集大小: {len(intersection)}" + (" [OK]" if len(intersection) == 0 else " [FAIL] 存在泄漏！"))

    # 持久化到 data/processed/
    print()
    print(">>> 保存拆分数据到 data/processed/ ...")
    saved = save_splits(X_train, X_test, y_train, y_test, feature_names,
                        X_val=X_val, y_val=y_val)
    for name, path in saved.items():
        df_tmp = pd.read_csv(path)
        print(f"    {path.name:12s}  -> {path}  ({df_tmp.shape[0]:,} rows)")

    print()
    print(">>> 验证可回读 ...")
    X_tr, X_te, y_tr, y_te, X_va, y_va, fnames = load_splits()
    assert X_tr.shape == X_train.shape, "X_train 回读失败"
    assert X_te.shape == X_test.shape, "X_test 回读失败"
    print("    回读成功，形状一致 [OK]")
