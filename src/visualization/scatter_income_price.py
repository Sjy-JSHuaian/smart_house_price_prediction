"""
scatter_income_price.py

绘制加州房价中位收入与房价的散点图，并保存到 reports/figures/。
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保直接运行此脚本也能导入 src 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from src.data.loader import load_california_housing

# ── 加载数据 ──────────────────────────────────────────────────
df = load_california_housing()

# ── 提取特征 ──────────────────────────────────────────────────
x = df["MedInc"].values.reshape(-1, 1)
y = df["MedHouseVal"].values

# ── 拟合趋势线 ───────────────────────────────────────────────
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# ── 绘制散点图 + 趋势线 ─────────────────────────────────────
plt.figure(figsize=(10, 6))

plt.scatter(x, y, alpha=0.3, s=10, c="steelblue", edgecolor="none", label="Data")
plt.plot(x, y_pred, color="red", linewidth=2, label=f"Trend (slope={model.coef_[0]:.4f})")

plt.title("Median Income vs. House Value in California")
plt.xlabel("Median Income ($100,000)")
plt.ylabel("Median House Value ($100,000)")
plt.legend()
plt.grid(True, alpha=0.3)

# ── 保存图片 ──────────────────────────────────────────────────
output_dir = PROJECT_ROOT / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "scatter_income_price.png",
    dpi=300,
    bbox_inches="tight",
)

# ── 显示 ──────────────────────────────────────────────────────
plt.show()

# ── 输出相关系数 ─────────────────────────────────────────────
corr = np.corrcoef(df["MedInc"], df["MedHouseVal"])[0, 1]
print(f"皮尔逊相关系数: {corr:.4f}")
print(f"图片已保存至: {output_dir / 'scatter_income_price.png'}")
