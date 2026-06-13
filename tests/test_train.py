"""
test_train.py — 测试模型训练与评估
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pickle
from sklearn.metrics import r2_score
from src.features.split_features_target import load_splits
from src.data.preprocess import fit_transform_train, transform


class TestModelTraining:
    """模型训练单元测试"""

    @classmethod
    def setup_class(cls):
        X_tr, X_te, y_tr, y_te, _, _, fnames = load_splits()
        _, _, pre = fit_transform_train(X_tr, y_tr, fnames)
        cls.X_train = transform(pre, X_tr, fnames)
        cls.X_test  = transform(pre, X_te, fnames)
        cls.y_train = y_tr
        cls.y_test  = y_te
        cls.models_dir = PROJECT_ROOT / "models"
        cls.fnames = fnames

    def _load_model(self, name):
        path = self.models_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"模型不存在: {path}（请先训练）")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_linear_regression_exists_and_predicts(self):
        model = self._load_model("linear_regression")
        pred = model.predict(self.X_test[:10])
        assert len(pred) == 10
        assert np.all(pred > 0) and np.all(pred < 10), f"预测值异常: {pred}"

    def test_linear_regression_r2_above_baseline(self):
        model = self._load_model("linear_regression")
        pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, pred)
        assert r2 > 0.5, f"LinearRegression R^2={r2:.4f} < 0.5"

    def test_decision_tree_exists_and_predicts(self):
        model = self._load_model("decision_tree")
        pred = model.predict(self.X_test[:10])
        assert len(pred) == 10
        assert np.all(pred > 0) and np.all(pred < 10)

    def test_decision_tree_r2_above_baseline(self):
        model = self._load_model("decision_tree")
        pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, pred)
        assert r2 > 0.6, f"DecisionTree R^2={r2:.4f} < 0.6"

    def test_random_forest_exists_and_predicts(self):
        model = self._load_model("random_forest")
        pred = model.predict(self.X_test[:10])
        assert len(pred) == 10
        assert np.all(pred > 0) and np.all(pred < 10)

    def test_random_forest_r2_best(self):
        model = self._load_model("random_forest")
        pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, pred)
        assert r2 > 0.75, f"RandomForest R^2={r2:.4f} < 0.75"

    def test_all_three_models_produce_different_predictions(self):
        sample = self.X_test[:5]
        preds = []
        for name in ["linear_regression", "decision_tree", "random_forest"]:
            preds.append(self._load_model(name).predict(sample))
        assert not np.allclose(preds[0], preds[1], rtol=0.01), "LR 和 DT 预测完全一致"
        assert not np.allclose(preds[0], preds[2], rtol=0.01), "LR 和 RF 预测完全一致"


if __name__ == "__main__":
    tester = TestModelTraining()
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
