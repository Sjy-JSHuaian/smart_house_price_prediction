"""
scatter_rooms_price.py

绘制加州房价房间数与房价的散点图，并保存到 reports/figures/。
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

# ── 提取特征（剔除极端异常值，AveRooms 长尾严重）────────────
x_raw = df["AveRooms"].values.reshape(-1, 1)
y_raw = df["MedHouseVal"].values

# 只保留 AveRooms 在 99 分位数以内的数据，避免极端值扭曲散点图
upper = np.percentile(x_raw, 99)
mask = (x_raw <= upper).ravel()

x = x_raw[mask].reshape(-1, 1)
y = y_raw[mask]

# ── 拟合趋势线 ───────────────────────────────────────────────
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# ── 绘制散点图 + 趋势线 ─────────────────────────────────────
plt.figure(figsize=(10, 6))

plt.scatter(x, y, alpha=0.3, s=10, c="steelblue", edgecolor="none", label="Data (99th pctl)")
plt.plot(x, y_pred, color="red", linewidth=2, label=f"Trend (slope={model.coef_[0]:.4f})")

plt.title("Average Rooms vs. House Value in California")
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median House Value ($100,000)")
plt.legend()
plt.grid(True, alpha=0.3)

# ── 保存图片 ──────────────────────────────────────────────────
output_dir = PROJECT_ROOT / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "scatter_rooms_price.png",
    dpi=300,
    bbox_inches="tight",
)

# ── 显示 ──────────────────────────────────────────────────────
plt.show()

# ── 输出相关系数 ─────────────────────────────────────────────
corr = np.corrcoef(df["AveRooms"], df["MedHouseVal"])[0, 1]
corr_trimmed = np.corrcoef(x.flatten(), y)[0, 1]
print(f"皮尔逊相关系数 (AveRooms vs MedHouseVal, 全量): {corr:.4f}")
print(f"皮尔逊相关系数 (AveRooms vs MedHouseVal, 99分位内): {corr_trimmed:.4f}")
print(f"图片已保存至: {output_dir / 'scatter_rooms_price.png'}")
