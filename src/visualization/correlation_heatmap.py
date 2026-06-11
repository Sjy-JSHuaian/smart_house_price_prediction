"""
correlation_heatmap.py

绘制加州房价数据集各特征间的相关系数热力图，并保存到 reports/figures/。
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保直接运行此脚本也能导入 src 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data.loader import load_california_housing

# ── 加载数据 ──────────────────────────────────────────────────
df = load_california_housing()

# ── 计算相关系数矩阵 ─────────────────────────────────────────
corr_matrix = df.corr()

# ── 绘制热力图 ───────────────────────────────────────────────
plt.figure(figsize=(12, 10))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
)

plt.title("California Housing — Feature Correlation Heatmap", fontsize=14, pad=20)

# ── 保存图片 ──────────────────────────────────────────────────
output_dir = PROJECT_ROOT / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

# ── 显示 ──────────────────────────────────────────────────────
plt.show()

# ── 输出与目标变量最相关的特征 ───────────────────────────────
target_corr = corr_matrix["MedHouseVal"].drop("MedHouseVal").sort_values(key=abs, ascending=False)
print("各特征与 MedHouseVal 的相关系数（按绝对值排序）:")
for feat, val in target_corr.items():
    print(f"  {feat:15s}: {val:+.4f}")
print(f"\n图片已保存至: {output_dir / 'correlation_heatmap.png'}")
