"""
boxplot_analysis.py

绘制加州房价数据集各数值特征的箱线图，检测异常值（IQR 方法），
并保存到 reports/figures/。
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保直接运行此脚本也能导入 src 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import load_california_housing

# ── 加载数据 ──────────────────────────────────────────────────
df = load_california_housing()

# ── 异常值检测（IQR 方法）────────────────────────────────────
def detect_outliers_iqr(series: np.ndarray) -> dict:
    """
    使用 IQR（四分位距）方法检测异常值。

    Returns
    -------
    dict
        lower_fence, upper_fence, outlier_count, outlier_pct, indices
    """
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "outlier_count": int(mask.sum()),
        "outlier_pct": round(100 * mask.sum() / len(series), 2),
    }

# ── 数值列 ────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ── 绘制分组箱线图 ───────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    bp = ax.boxplot(
        df[col].dropna().values,
        vert=True,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor="steelblue", alpha=0.7),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(color="red", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="red", markersize=3, alpha=0.5),
    )

    info = detect_outliers_iqr(df[col].dropna().values)
    ax.set_title(f"{col}\n(outliers: {info['outlier_pct']}%)", fontsize=10)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)

# 隐藏多余的子图（9 个格子，9 列刚好填满则无需隐藏）
for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("California Housing — Boxplot & Outlier Analysis (IQR Method)", fontsize=14, y=1.01)
plt.tight_layout()

# ── 保存图片 ──────────────────────────────────────────────────
output_dir = PROJECT_ROOT / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "boxplot_analysis.png",
    dpi=300,
    bbox_inches="tight",
)

# ── 显示 ──────────────────────────────────────────────────────
plt.show()

# ── 终端输出异常值报告 ───────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Feature':<16s} {'Q1':>10s} {'Q3':>10s} {'IQR':>10s} {'Lower':>10s} {'Upper':>10s} {'Count':>7s} {'%':>7s}")
print("-" * 70)
for col in numeric_cols:
    info = detect_outliers_iqr(df[col].dropna().values)
    print(
        f"{col:<16s} {info['Q1']:10.4f} {info['Q3']:10.4f} {info['IQR']:10.4f} "
        f"{info['lower_fence']:10.4f} {info['upper_fence']:10.4f} "
        f"{info['outlier_count']:7d} {info['outlier_pct']:7.2f}%"
    )
print("=" * 70)
print(f"\n图片已保存至: {output_dir / 'boxplot_analysis.png'}")
