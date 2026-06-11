"""
plot_distribution.py

绘制加州房价分布图（直方图），并保存到 reports/figures/。
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保直接运行此脚本也能导入 src 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from src.data.loader import load_california_housing

# ── 加载数据 ──────────────────────────────────────────────────
df = load_california_housing()

# ── 绘制直方图 ────────────────────────────────────────────────
plt.figure(figsize=(10, 6))

df["MedHouseVal"].hist(bins=30)

plt.title("California Housing Price Distribution")
plt.xlabel("Median House Value ($100,000)")
plt.ylabel("Frequency")

# ── 保存图片 ──────────────────────────────────────────────────
output_dir = PROJECT_ROOT / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "house_price_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

# ── 显示 ──────────────────────────────────────────────────────
plt.show()
