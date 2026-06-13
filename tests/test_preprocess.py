"""
test_preprocess.py — 测试数据预处理流水线
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.data.preprocess import (
    build_preprocessor,
    fit_transform_train,
    transform,
    load_preprocessor,
    save_preprocessor,
    STANDARD_FEATURES,
)
from src.features.split_features_target import load_splits


class TestPreprocessor:
    """预处理器单元测试"""

    @classmethod
    def setup_class(cls):
        X_tr, X_te, y_tr, y_te, _, _, fnames = load_splits()
        cls.X_train, cls.X_test = X_tr, X_te
        cls.y_train, cls.y_test = y_tr, y_te
        cls.fnames = fnames

    def test_build_returns_column_transformer(self):
        pre = build_preprocessor()
        assert hasattr(pre, "fit"), "预处理器缺少 fit 方法"
        assert hasattr(pre, "transform"), "预处理器缺少 transform 方法"

    def test_standard_features_mean_zero_std_one(self):
        _, _, pre = fit_transform_train(self.X_train, self.y_train, self.fnames)
        X_tr = transform(pre, self.X_train, self.fnames)
        means = X_tr[:, :len(STANDARD_FEATURES)].mean(axis=0)
        stds  = X_tr[:, :len(STANDARD_FEATURES)].std(axis=0)
        assert np.allclose(means, 0, atol=1e-6), f"Standard 列均值偏离 0: {means}"
        assert np.allclose(stds, 1, atol=1e-4), f"Standard 列标准差偏离 1: {stds}"

    def test_robust_features_median_zero_iqr_one(self):
        _, _, pre = fit_transform_train(self.X_train, self.y_train, self.fnames)
        X_tr = transform(pre, self.X_train, self.fnames)
        n_std = len(STANDARD_FEATURES)
        medians = np.median(X_tr[:, n_std:], axis=0)
        q1 = np.percentile(X_tr[:, n_std:], 25, axis=0)
        q3 = np.percentile(X_tr[:, n_std:], 75, axis=0)
        iqrs = q3 - q1
        assert np.allclose(medians, 0, atol=1e-6), f"Robust 列中位数偏离 0: {medians}"
        assert np.allclose(iqrs, 1, atol=1e-4), f"Robust 列 IQR 偏离 1: {iqrs}"

    def test_fit_only_on_train_no_leakage(self):
        _, _, pre = fit_transform_train(self.X_train, self.y_train, self.fnames)
        X_te = transform(pre, self.X_test, self.fnames)
        means_te = X_te[:, :len(STANDARD_FEATURES)].mean(axis=0)
        assert np.all(np.abs(means_te) < 0.1), f"测试集 mean 偏移过大: {means_te}"

    def test_save_and_load_preprocessor(self):
        _, _, pre1 = fit_transform_train(self.X_train, self.y_train, self.fnames)
        pkl_path = save_preprocessor(pre1, path=PROJECT_ROOT / "models" / "_test_preproc.pkl")
        pre2 = load_preprocessor(pkl_path)
        assert np.allclose(
            transform(pre1, self.X_test, self.fnames),
            transform(pre2, self.X_test, self.fnames),
        ), "保存/加载后的预处理器输出不一致"
        pkl_path.unlink()

    def test_output_shape_preserved(self):
        _, _, pre = fit_transform_train(self.X_train, self.y_train, self.fnames)
        X_tr = transform(pre, self.X_train, self.fnames)
        X_te = transform(pre, self.X_test, self.fnames)
        assert X_tr.shape == self.X_train.shape
        assert X_te.shape == self.X_test.shape

    def test_no_nan_after_transform(self):
        _, _, pre = fit_transform_train(self.X_train, self.y_train, self.fnames)
        X_tr = transform(pre, self.X_train, self.fnames)
        X_te = transform(pre, self.X_test, self.fnames)
        assert not np.any(np.isnan(X_tr)), "训练集包含 NaN"
        assert not np.any(np.isnan(X_te)), "测试集包含 NaN"


if __name__ == "__main__":
    tester = TestPreprocessor()
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
