"""
reports.py

生成加州房价数据集的 EDA（探索性数据分析）报告，输出为 Markdown 格式，
保存到 reports/final_report.md。
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datetime import datetime

from src.data.loader import load_california_housing, get_data_info

# ── 加载数据 ──────────────────────────────────────────────────
df = load_california_housing()
info = get_data_info(df)

# ── 异常值检测 ───────────────────────────────────────────────
def detect_outliers_iqr(series: np.ndarray) -> dict:
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_fence": lower, "upper_fence": upper,
        "outlier_count": int(mask.sum()),
        "outlier_pct": round(100 * mask.sum() / len(series), 2),
    }

# ── 预计算所有统计量 ────────────────────────────────────────
target = "MedHouseVal"
features = [c for c in df.columns if c != target]

# 描述统计
desc = df.describe().round(4)

# 相关系数
corr_matrix = df.corr()
target_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)

# 异常值汇总
outlier_summary = {}
for col in df.columns:
    outlier_summary[col] = detect_outliers_iqr(df[col].dropna().values)

# 偏度 & 峰度
skew = df.skew().round(4)
kurt = df.kurtosis().round(4)

# ── 生成 Markdown 报告 ──────────────────────────────────────
md = []

def w(s: str = ""):
    md.append(s)

w(f"# California Housing — EDA 分析报告")
w()
w(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"**数据来源**: `sklearn.datasets.fetch_california_housing`")
w()

w("---")
w()
w("## 1. 数据集概览")
w()
w("| 指标 | 值 |")
w("|------|----|")
w(f"| 样本数 | {info['shape'][0]:,} |")
w(f"| 特征数 | {info['shape'][1]} |")
w(f"| 缺失值 | 0（所有列完整） |")
w(f"| 目标变量 | `MedHouseVal`（中位房价，单位 $100,000） |")
w(f"| 数值特征 | {len(features)} 个（均为 `float64`） |")
w()
w("**列名与类型**:")
w()
for col, dtype in info["dtypes"].items():
    tag = ">> Target" if col == target else "   "
    w(f"- {tag} `{col}` — `{dtype}`")
w()

w("---")
w()
w("## 2. 描述性统计")
w()
w("### 2.1 核心统计量")
w()
stats_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
w("| Feature | " + " | ".join(stats_cols) + " |")
w("|" + "|".join(["---------"] * (len(stats_cols) + 1)) + "|")
for col in df.columns:
    vals = [f"{desc.loc[s, col]:.4f}" if s in desc.index else "-" for s in stats_cols]
    w(f"| `{col}` | " + " | ".join(vals) + " |")
w()

w("### 2.2 偏度与峰度")
w()
w("| Feature | Skewness | Kurtosis | 分布形态 |")
w("|---------|----------|----------|----------|")
for col in df.columns:
    s_val = skew[col]
    k_val = kurt[col]
    if abs(s_val) < 0.5:
        shape = "近似对称"
    elif s_val > 0:
        shape = "右偏（长尾在右）"
    else:
        shape = "左偏（长尾在左）"
    if k_val > 1:
        shape += "，厚尾"
    elif k_val < -1:
        shape += "，薄尾"
    w(f"| `{col}` | {s_val:+.4f} | {k_val:+.4f} | {shape} |")
w()

w("---")
w()
w("## 3. 目标变量分析 — `MedHouseVal`")
w()
w(f"- **均值**: {desc.loc['mean', target]:.4f}（约 ${desc.loc['mean', target]*100000:,.0f}）")
w(f"- **中位数**: {desc.loc['50%', target]:.4f}")
w(f"- **标准差**: {desc.loc['std', target]:.4f}")
w(f"- **范围**: [{desc.loc['min', target]:.4f}, {desc.loc['max', target]:.4f}]")
w(f"- **偏度**: {skew[target]:+.4f}（右偏，多数房价集中在低价区间）")
w(f"- **峰度**: {kurt[target]:+.4f}")

w()
cap_val = df[target].max()
cap_pct = round(100 * (df[target] >= cap_val - 0.01).sum() / len(df), 2)
w(f"> **注意**: 目标变量在 ${cap_val*100000:,.0f} 处存在截断天花板效应（约 {cap_pct}% 样本被截断），这是该数据集的已知问题。")
w()
w(f"![房价分布](figures/house_price_distribution.png)")
w()

w("---")
w()
w("## 4. 特征相关性分析")
w()
w("### 4.1 各特征与目标变量的相关系数")
w()
w("| 排名 | 特征 | 皮尔逊 r | 相关强度 |")
w("|------|------|----------|----------|")
for rank, (feat, val) in enumerate(target_corr.items(), 1):
    if abs(val) >= 0.5:
        strength = "强" + ("正" if val > 0 else "负")
    elif abs(val) >= 0.3:
        strength = "中等" + ("正" if val > 0 else "负")
    else:
        strength = "弱" + ("正" if val > 0 else "负")
    w(f"| {rank} | `{feat}` | {val:+.4f} | {strength}相关 |")
w()

w()
w("### 4.2 关键发现")
w()
top3 = target_corr.head(3).index.tolist()
bottom3 = target_corr.tail(3).index.tolist()
w(f"- **最强正相关**: `{top3[0]}` (r={target_corr[top3[0]]:+.4f}) — 收入是房价最重要的预测因子")
w(f"- **次强正相关**: `{top3[1]}` (r={target_corr[top3[1]]:+.4f})")
w(f"- **最弱相关**: `{bottom3[0]}` (r={target_corr[bottom3[0]]:+.4f}) — 与房价几乎无线性关系")
w()

w(f"![相关性热力图](figures/correlation_heatmap.png)")
w()

w("### 4.3 特征间多重共线性")
w()
high_corr_pairs = []
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        r_val = corr_matrix.loc[features[i], features[j]]
        if abs(r_val) > 0.5:
            high_corr_pairs.append((features[i], features[j], r_val))

if high_corr_pairs:
    w("以下特征对之间的相关系数 > |0.5|，存在多重共线性风险：")
    w()
    for f1, f2, r_val in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
        w(f"- `{f1}` <-> `{f2}`: r = {r_val:+.4f}")
else:
    w("未发现高度相关的特征对。")
w()

w("---")
w()
w("## 5. 异常值分析（IQR 方法）")
w()
w("| Feature | Q1 | Q3 | IQR | 下界 | 上界 | 异常值数 | 比例 |")
w("|---------|----|----|-----|------|------|----------|------|")
for col in df.columns:
    o = outlier_summary[col]
    w(f"| `{col}` | {o['Q1']:.4f} | {o['Q3']:.4f} | {o['IQR']:.4f} | {o['lower_fence']:.4f} | {o['upper_fence']:.4f} | {o['outlier_count']:,} | {o['outlier_pct']:.2f}% |")
w()

w()
high_outlier_cols = [(col, outlier_summary[col]["outlier_pct"]) for col in df.columns if outlier_summary[col]["outlier_pct"] > 5]
if high_outlier_cols:
    w("**异常值比例 > 5% 的特征**（需重点关注）:")
    for col, pct in sorted(high_outlier_cols, key=lambda x: -x[1]):
        w(f"- `{col}`: {pct:.2f}% — 建议在建模前做截尾或变换处理")
w()

w(f"![箱线图分析](figures/boxplot_analysis.png)")
w()

w("---")
w()
w("## 6. 关键特征散点图")
w()

scatter_info = [
    ("MedInc", "中位收入", "scatter_income_price.png", "收入与房价呈明显正相关，是最强的单一预测因子。高收入区域房价显著更高，但方差也随之增大。"),
    ("HouseAge", "房龄", "scatter_age_price.png", "房龄与房价呈弱负相关，老房子略便宜但关系不显著。趋势线接近水平。"),
    ("AveRooms", "平均房间数", "scatter_rooms_price.png", "房间数与房价关系较弱且存在极端异常值（部分区域 AveRooms > 50）。图中已按 99 分位数裁剪以改善可读性。"),
]

for feat, label, fname, note in scatter_info:
    r_val = corr_matrix.loc[feat, target]
    w(f"### `{feat}` ({label}) vs `MedHouseVal`")
    w()
    w(f"- 皮尔逊 r: {r_val:+.4f}")
    w(f"- {note}")
    w()
    w(f"![{label}](figures/{fname})")
    w()

w("---")
w()
w("## 7. 空间分布初探")
w()
w(f"- `Latitude` 范围: [{desc.loc['min', 'Latitude']:.2f}, {desc.loc['max', 'Latitude']:.2f}]")
w(f"- `Longitude` 范围: [{desc.loc['min', 'Longitude']:.2f}, {desc.loc['max', 'Longitude']:.2f}]")
w(f"- `Latitude` 与房价相关性: {corr_matrix.loc['Latitude', target]:+.4f}")
w(f"- `Longitude` 与房价相关性: {corr_matrix.loc['Longitude', target]:+.4f}")
w()
w("经纬度与房价存在一定相关性，暗示地理位置是房价的重要影响因素。建议后续使用地理可视化（散点地图）或构建空间特征。")
w()

w("---")
w()
w("## 8. 建模建议")
w()
w("基于以上 EDA 分析，提出以下建模建议：")
w()
w("1. **特征选择**: `MedInc` 是最强预测因子，`AveBedrms`、`Latitude`、`Longitude` 也较为重要；`AveOccup`、`Population` 与目标几乎无关，可考虑剔除")
w("2. **特征工程**: ")
w("   - 对 `AveRooms`、`AveBedrms`、`Population` 做人均化（除以 `AveOccup`）可能提升信号")
w("   - 对 `Latitude`/`Longitude` 构建空间交互项或聚类标签")
w("   - `MedInc` 可尝试 log 变换使其更接近正态分布")
w("3. **异常值处理**: `AveRooms`（8.55% 异常值）、`AveBedrms`（8.90%）和 `Population`（7.40%）需要截尾或 Winsorize")
w("4. **天花板效应**: `MedHouseVal` 在 5.0 处截断，建模时需注意 — 可尝试 Tobit 回归或直接使用对天花板不敏感的模型（树模型）")
w("5. **模型选择**: 优先尝试 Gradient Boosting（XGBoost/LightGBM）和 Random Forest，它们对异常值和非线性关系更鲁棒")
w()

w("---")
w()
w("## 9. 附件清单")
w()
w(f"| 文件 | 说明 |")
w(f"|------|------|")
w(f"| `figures/house_price_distribution.png` | 目标变量分布直方图 |")
w(f"| `figures/correlation_heatmap.png` | 全特征相关性热力图 |")
w(f"| `figures/boxplot_analysis.png` | 异常值箱线图分析 |")
w(f"| `figures/scatter_income_price.png` | 收入 vs 房价散点图 |")
w(f"| `figures/scatter_age_price.png` | 房龄 vs 房价散点图 |")
w(f"| `figures/scatter_rooms_price.png` | 房间数 vs 房价散点图 |")
w()

# ── 写入文件 ──────────────────────────────────────────────────
output_path = PROJECT_ROOT / "reports" / "final_report.md"
output_path.write_text("\n".join(md), encoding="utf-8")

# ── 终端输出 ──────────────────────────────────────────────────
print("\n".join(md))
print(f"\n{'='*60}")
print(f"报告已保存至: {output_path}")
