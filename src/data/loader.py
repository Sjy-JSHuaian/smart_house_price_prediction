"""
Data loader for the California housing dataset.

设计原则：
    1. 职责清晰：loader.py 只负责读取数据，无论是 CSV、数据库还是 API，都从这里读取。
    2. 可复用：以后换数据源（比如 Kaggle 数据集或真实数据库），其他模块（模型训练、特征工程）不用改。
    3. 符合工程规范：把"数据输入"模块化，有利于单元测试和维护。
"""

from pathlib import Path
from typing import Union, Optional

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing


# ── 项目路径常量 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ── 核心加载函数 ──────────────────────────────────────────────

def load_california_housing(as_frame: bool = True) -> Union[pd.DataFrame, tuple]:
    """
    从 sklearn 内置数据集加载 California Housing 数据。

    Parameters
    ----------
    as_frame : bool, default=True
        若为 True，返回 pd.DataFrame；若为 False，返回 (X, y) 元组。

    Returns
    -------
    pd.DataFrame 或 tuple
        包含特征和目标列"MedHouseVal"的 DataFrame，或 (X, y) 元组。
    """
    housing = fetch_california_housing(as_frame=True)

    df = housing.frame          # 已包含特征 + 目标列 MedHouseVal
    df.columns = df.columns.str.strip()

    if as_frame:
        return df
    else:
        X = df.drop(columns="MedHouseVal")
        y = df["MedHouseVal"]
        return X, y


def load_csv(
    filepath: Union[str, Path],
    *,
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    从本地 CSV 文件加载数据。

    Parameters
    ----------
    filepath : str or Path
        CSV 文件路径。
    target_col : str, optional
        目标列名称，仅用于日志提示，不影响返回。

    Returns
    -------
    pd.DataFrame
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path.resolve()}")

    df = pd.read_csv(path)
    return df


# ── 数据写出 ──────────────────────────────────────────────────

def save_to_csv(
    df: pd.DataFrame,
    filename: str = "housing_cleaned.csv",
    directory: Optional[Path] = None
) -> Path:
    """
    将 DataFrame 保存到 data/raw/ 目录。

    Parameters
    ----------
    df : pd.DataFrame
        要保存的数据。
    filename : str, default="housing_cleaned.csv"
        输出文件名。
    directory : Path, optional
        目标目录，默认为 data/raw/。

    Returns
    -------
    Path
        实际写入的文件路径。
    """
    target_dir = directory or RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


# ── 数据信息 ──────────────────────────────────────────────────

def get_data_info(df: pd.DataFrame) -> dict:
    """
    获取 DataFrame 的基本统计信息，用于快速数据探查。

    Returns
    -------
    dict
        shape, dtypes, missing_count, describe 等。
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "describe": df.describe().round(4),
        "head": df.head(3),
    }


# ── 快速预览 ──────────────────────────────────────────────────

def preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """返回前 n 行，方便在 Notebook 中快速查看。"""
    return df.head(n)


# ── CLI 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    # 快速自检：加载数据 → 查看信息 → 保存
    print("=" * 60)
    print(">>> 加载 California Housing 数据 ...")
    df = load_california_housing(as_frame=True)

    print(">>> 前 5 行预览：")
    print(preview(df))

    info = get_data_info(df)
    print(f"\n>>> 数据形状: {info['shape']}")
    print(f"   列名: {info['columns']}")
    print(f"   缺失值比例: {info['missing_pct']}")

    saved_path = save_to_csv(df, filename="california_housing_raw.csv")
    print(f"\n>>> 原始数据已保存至: {saved_path}")
    print("=" * 60)
