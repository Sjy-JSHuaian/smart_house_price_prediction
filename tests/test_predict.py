"""
test_predict.py — 测试预测流水线（端到端）
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pickle
from src.data.loader import load_california_housing
from src.features.split_features_target import load_splits
from src.data.preprocess import load_preprocessor, transform


class TestPredictionPipeline:
    """端到端预测流水线测试"""

    @classmethod
    def setup_class(cls):
        cls.df = load_california_housing()
        cls.preprocessor = load_preprocessor()
        X_tr, X_te, _, _, _, _, fnames = load_splits()
        cls.X_train, cls.X_test = X_tr, X_te
        cls.fnames = fnames

    def _load_model(self, name):
        path = PROJECT_ROOT / "models" / f"{name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_single_sample_predict_all_three(self):
        sample = self.df[self.fnames].iloc[0:1].values
        X_scaled = transform(self.preprocessor, sample, self.fnames)
        for name in ["linear_regression", "decision_tree", "random_forest"]:
            pred = self._load_model(name).predict(X_scaled)[0]
            assert 0.1 < pred < 5.0, f"{name} 预测值异常: {pred:.4f}"

    def test_batch_prediction_shape(self):
        X_scaled = transform(self.preprocessor, self.X_test[:100], self.fnames)
        for name in ["linear_regression", "decision_tree", "random_forest"]:
            pred = self._load_model(name).predict(X_scaled)
            assert pred.shape == (100,), f"{name} 输出 shape={pred.shape}，期望 (100,)"

    def test_raw_csv_input_vs_scaled_input_consistency(self):
        sample_raw = self.X_train[:1]
        scaled_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "Xy_train_scaled.csv")
        feats = [c for c in scaled_df.columns if c != "MedHouseVal"]
        sample_scaled = scaled_df[feats].values[:1]
        X_transformed = transform(self.preprocessor, sample_raw, self.fnames)
        assert np.allclose(X_transformed, sample_scaled, atol=1e-4), \
            "实时 transform 与已保存的标准化数据不一致"

    def test_input_out_of_range_still_predicts(self):
        extreme = np.array([[0.5, 1.0, 0.8, 0.3, 3.0, 0.7, 32.5, -124.4]])
        X_scaled = transform(self.preprocessor, extreme, self.fnames)
        pred = self._load_model("random_forest").predict(X_scaled)
        assert not np.any(np.isnan(pred)), "极端输入导致 NaN"
        assert np.all(pred >= 0), f"预测值为负: {pred}"

    def test_predict_output_is_invertible(self):
        X_scaled = transform(self.preprocessor, self.X_test, self.fnames)
        pred_units = self._load_model("random_forest").predict(X_scaled)
        pred_dollars = pred_units * 100_000
        assert np.all(pred_dollars >= 10_000), f"预测价格过低: min=${pred_dollars.min():,.0f}"
        assert np.all(pred_dollars <= 550_000), f"预测价格过高: max=${pred_dollars.max():,.0f}"


if __name__ == "__main__":
    tester = TestPredictionPipeline()
    tester.setup_class()
    tests = [m for m in dir(tester) if m.startswith("test_")]
    passed = 0
    for name in tests:
        try:
            getattr(tester, name)()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
