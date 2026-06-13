"""
streamlit_app.py

California Housing Price Prediction — 交互式 Web 应用。

功能：
  1. EDA 探索 — 展示所有分析图表
  2. 房价预测 — 输入特征，三模型同时预测
  3. 模型对比 — 指标汇总
"""

import sys
from pathlib import Path
import pickle

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.data.loader import load_california_housing
from src.data.preprocess import load_preprocessor

# ── 页面配置 ──────────────────────────────────────────────────

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon=":house:",
    layout="wide",
)

# ── 缓存：加载模型和数据 ────────────────────────────────────

@st.cache_resource
def load_models() -> dict:
    """加载所有已训练的模型。"""
    models_dir = PROJECT_ROOT / "models"
    models = {}
    for name in ["linear_regression", "decision_tree", "random_forest"]:
        path = models_dir / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


@st.cache_resource
def load_preproc():
    return load_preprocessor()


@st.cache_data
def load_data():
    df = load_california_housing()
    return df


# ── 特征输入默认值 ──────────────────────────────────────────

FEATURE_DEFAULTS = {
    "MedInc": 3.87,
    "HouseAge": 28.6,
    "AveRooms": 5.43,
    "AveBedrms": 1.10,
    "Population": 1425.0,
    "AveOccup": 3.07,
    "Latitude": 35.63,
    "Longitude": -119.57,
}

FEATURE_RANGES = {
    "MedInc": (0.5, 15.0),
    "HouseAge": (1.0, 52.0),
    "AveRooms": (0.8, 20.0),
    "AveBedrms": (0.3, 5.0),
    "Population": (0, 5000),
    "AveOccup": (0.7, 10.0),
    "Latitude": (32.5, 42.0),
    "Longitude": (-124.4, -114.3),
}

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


# ── 侧边栏 ──────────────────────────────────────────────────

st.sidebar.title("California Housing")
st.sidebar.markdown("房价预测系统")

page = st.sidebar.radio(
    "导航",
    ["EDA 探索", "房价预测", "模型对比"],
)

st.sidebar.markdown("---")
st.sidebar.caption("数据: sklearn California Housing (20,640 samples)")
st.sidebar.caption("目标: MedHouseVal ($100,000)")


# =================================================================
# Page 1: EDA 探索
# =================================================================

if page == "EDA 探索":
    st.title("探索性数据分析 (EDA)")

    df = load_data()

    # 数据快照
    st.header("1. 数据快照")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("样本数", f"{df.shape[0]:,}")
    col2.metric("特征数", df.shape[1] - 1)
    col3.metric("缺失值", "0")
    col4.metric("目标均值", f"{df['MedHouseVal'].mean():.2f}")

    with st.expander("查看原始数据（前 100 行）"):
        st.dataframe(df.head(100), use_container_width=True)

    # 描述统计
    st.header("2. 描述统计")
    st.dataframe(df.describe().round(4), use_container_width=True)

    # 图表展示
    st.header("3. 可视化分析")
    figures_dir = PROJECT_ROOT / "reports" / "figures"

    tab1, tab2, tab3, tab4 = st.tabs([
        "目标分布", "相关性热力图", "散点图", "异常值分析"
    ])

    with tab1:
        img = figures_dir / "house_price_distribution.png"
        if img.exists():
            st.image(str(img), caption="房价分布直方图", use_container_width=True)

    with tab2:
        img = figures_dir / "correlation_heatmap.png"
        if img.exists():
            st.image(str(img), caption="特征相关性热力图", use_container_width=True)

    with tab3:
        imgs = ["scatter_income_price.png", "scatter_age_price.png", "scatter_rooms_price.png"]
        captions = ["收入 vs 房价", "房龄 vs 房价", "房间数 vs 房价"]
        cols = st.columns(3)
        for col, fn, cap in zip(cols, imgs, captions):
            p = figures_dir / fn
            if p.exists():
                col.image(str(p), caption=cap, use_container_width=True)

    with tab4:
        img = figures_dir / "boxplot_analysis.png"
        if img.exists():
            st.image(str(img), caption="IQR 异常值分析", use_container_width=True)

    # EDA 结论
    st.header("4. 关键发现")
    st.markdown("""
    | 发现 | 详情 |
    |------|------|
    | 最强预测因子 | `MedInc`（中位收入），r = **+0.688** |
    | 天花板效应 | `MedHouseVal` 在 $500K 截断（~4.8% 样本） |
    | 多重共线性 | `Latitude` <-> `Longitude` (r=-0.92), `AveRooms` <-> `AveBedrms` (r=+0.85) |
    | 异常值重灾区 | `AveBedrms` (6.9%), `Population` (5.8%) |
    """)


# =================================================================
# Page 2: 房价预测
# =================================================================

elif page == "房价预测":
    st.title("房价预测")

    models = load_models()
    preprocessor = load_preproc()

    if not models:
        st.error("未找到已训练的模型！请先运行 train_*.py")
        st.stop()

    # 输入区
    st.header("输入房屋特征")

    col1, col2 = st.columns(2)

    inputs = {}
    features_order = list(FEATURE_DEFAULTS.keys())
    half = len(features_order) // 2

    for i, feat in enumerate(features_order):
        col = col1 if i < half else col2
        lo, hi = FEATURE_RANGES[feat]
        inputs[feat] = col.slider(
            f"{feat}",
            float(lo), float(hi),
            FEATURE_DEFAULTS[feat],
            step=0.01 if feat in ("MedInc", "AveRooms", "AveBedrms", "AveOccup") else 0.1,
        )

    # 预测按钮
    st.markdown("---")
    if st.button("预测房价", type="primary", use_container_width=True):
        # 构建输入
        X_input = np.array([[inputs[f] for f in features_order]])
        X_scaled = preprocessor.transform(pd.DataFrame(X_input, columns=features_order))

        st.header("预测结果")
        cols = st.columns(len(models))

        for col, (name, model) in zip(cols, models.items()):
            pred = model.predict(X_scaled)[0]
            price = pred * 100_000
            col.metric(
                MODEL_LABELS[name],
                f"${price:,.0f}",
            )
            col.caption(f"{pred:.4f} units")

        # 三模型对比条
        st.markdown("---")
        st.subheader("三模型预测对比")
        preds = {}
        for name, model in models.items():
            preds[MODEL_LABELS[name]] = model.predict(X_scaled)[0] * 100_000

        df_pred = pd.DataFrame({
            "Model": list(preds.keys()),
            "Prediction ($)": list(preds.values()),
        }).sort_values("Prediction ($)")

        st.bar_chart(df_pred.set_index("Model"), use_container_width=True)

        # 快速参考
        with st.expander("特征输入参考"):
            st.markdown("""
            **加州典型房屋特征（均值附近）**:
            - MedInc: 3.87（中位收入 $38,700）
            - HouseAge: 28.6 年
            - AveRooms: 5.43（每户平均房间数）
            - Latitude: 35.63 / Longitude: -119.57（加州中部）

            **高房价信号**:
            - MedInc > 6（高收入区域）
            - Latitude 在 34-38 之间（沿海）
            """)

    else:
        st.info("调整滑块后点击「预测房价」按钮")


# =================================================================
# Page 3: 模型对比
# =================================================================

else:
    st.title("模型对比")

    # 指标汇总
    st.header("测试集性能对比")

    perf_data = {
        "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
        "R^2": [0.5758, 0.6849, 0.8013],
        "RMSE ($)": [74561, 64260, 51021],
        "MAE ($)": [53330, 43460, 34010],
    }
    df_perf = pd.DataFrame(perf_data).set_index("Model")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.dataframe(df_perf, use_container_width=True)

    with col2:
        st.metric("最佳模型", "Random Forest", delta="R^2 = 0.8013")
        st.metric("vs 线性回归", "+0.2255 R^2", delta="误差降低 $23,540")
        st.metric("vs 决策树", "+0.1164 R^2", delta="误差降低 $13,239")

    # 可视化对比
    st.header("预测效果可视化")
    figures_dir = PROJECT_ROOT / "reports" / "figures"

    tab_lr, tab_dt, tab_rf = st.tabs([
        "Linear Regression", "Decision Tree", "Random Forest"
    ])
    with tab_lr:
        img = figures_dir / "lr_predictions.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
    with tab_dt:
        img = figures_dir / "dt_predictions.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
    with tab_rf:
        img = figures_dir / "rf_predictions.png"
        if img.exists():
            st.image(str(img), use_container_width=True)

    # 特征重要性
    st.header("特征重要性")
    col1, col2 = st.columns(2)
    with col1:
        img = figures_dir / "dt_feature_importance.png"
        if img.exists():
            st.image(str(img), caption="Decision Tree", use_container_width=True)
    with col2:
        img = figures_dir / "rf_feature_importance.png"
        if img.exists():
            st.image(str(img), caption="Random Forest", use_container_width=True)

    # 结论
    st.header("总结")
    st.markdown("""
    1. **Random Forest 最优**: R^2=0.8013，解释 80% 房价方差，预测误差约 +/-$51K
    2. **非线性关系显著**: Decision Tree (0.68) 大幅优于 Linear Regression (0.58)
    3. **最重要特征**: 收入 (MedInc) 始终是第一预测因子
    4. **天花板限制**: 目标变量在 $500K 截断，所有模型在高端房产预测上均被低估
    """)
