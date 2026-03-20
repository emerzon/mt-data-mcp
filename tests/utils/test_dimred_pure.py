"""Tests for utils/dimred.py — DimReducer classes and create_reducer factory."""

import numpy as np
import pytest

from mtdata.utils.dimred import (
    NoneReducer,
    PCAReducer,
    SVDReducer,
    SparsePCAReducer,
    KPCAReducer,
    LaplacianReducer,
    IsomapReducer,
    TSNEReducer,
    create_reducer,
    list_dimred_methods,
)


def _synthetic_data(n=50, d=5, seed=42):
    return np.random.RandomState(seed).randn(n, d).astype(np.float32)


class TestNoneReducer:
    def test_passthrough(self):
        X = _synthetic_data()
        r = NoneReducer()
        out = r.fit_transform(X)
        np.testing.assert_allclose(out, X, atol=1e-6)
        assert r.supports_transform()
        assert r.info() == {"method": "none"}


class TestPCAReducer:
    def test_fit_transform(self):
        X = _synthetic_data()
        r = PCAReducer(n_components=2)
        out = r.fit_transform(X)
        assert out.shape == (50, 2)

    def test_transform_after_fit(self):
        X = _synthetic_data()
        r = PCAReducer(n_components=2)
        r.fit(X)
        out = r.transform(X)
        assert out.shape == (50, 2)

    def test_info(self):
        r = PCAReducer(n_components=3)
        info = r.info()
        assert info["method"] == "pca"
        assert info["n_components"] == 3


class TestSVDReducer:
    def test_fit_transform(self):
        X = _synthetic_data()
        r = SVDReducer(n_components=2)
        out = r.fit_transform(X)
        assert out.shape == (50, 2)

    def test_info(self):
        assert SVDReducer(n_components=2).info()["method"] == "svd"


class TestSparsePCAReducer:
    def test_fit_transform(self):
        X = _synthetic_data()
        r = SparsePCAReducer(n_components=2, alpha=0.5)
        out = r.fit_transform(X)
        assert out.shape == (50, 2)

    def test_info(self):
        info = SparsePCAReducer(n_components=2, alpha=0.5).info()
        assert info["method"] == "spca"
        assert info["alpha"] == 0.5


class TestKPCAReducer:
    def test_fit_transform(self):
        X = _synthetic_data()
        r = KPCAReducer(n_components=2, kernel="rbf")
        out = r.fit_transform(X)
        assert out.shape == (50, 2)

    def test_info_fields(self):
        info = KPCAReducer(
            n_components=2, kernel="poly", gamma=0.1, degree=2, coef0=0.5
        ).info()
        assert info["kernel"] == "poly"
        assert info["gamma"] == 0.1
        assert info["degree"] == 2


class TestLaplacianReducer:
    def test_fit_transform(self):
        X = _synthetic_data()
        r = LaplacianReducer(n_components=2, n_neighbors=5)
        out = r.fit_transform(X)
        assert out.shape == (50, 2)

    def test_no_transform(self):
        r = LaplacianReducer(n_components=2)
        assert not r.supports_transform()
        r.fit(_synthetic_data())
        with pytest.raises(RuntimeError):
            r.transform(_synthetic_data(10, 5))


class TestIsomapReducer:
    def test_fit_transform(self):
        X = _synthetic_data()
        r = IsomapReducer(n_components=2, n_neighbors=5)
        out = r.fit_transform(X)
        assert out.shape == (50, 2)

    def test_info(self):
        info = IsomapReducer(n_components=2, n_neighbors=7).info()
        assert info["n_neighbors"] == 7


class TestTSNEReducer:
    def test_fit_transform(self):
        X = _synthetic_data()
        r = TSNEReducer(n_components=2, perplexity=10.0, n_iter=300)
        out = r.fit_transform(X)
        assert out.shape == (50, 2)

    def test_no_transform(self):
        r = TSNEReducer(n_components=2, perplexity=10.0, n_iter=300)
        assert not r.supports_transform()
        with pytest.raises(RuntimeError):
            r.transform(_synthetic_data())

    def test_info(self):
        info = TSNEReducer(
            n_components=2, perplexity=15.0, learning_rate=100.0, n_iter=500
        ).info()
        assert info["perplexity"] == 15.0
        assert info["supports_transform"] is False


class TestCreateReducer:
    def test_none(self):
        r, p = create_reducer(None)
        assert isinstance(r, NoneReducer)
        assert p["method"] == "none"

    def test_none_string(self):
        r, p = create_reducer("none")
        assert isinstance(r, NoneReducer)

    def test_pca(self):
        r, p = create_reducer("pca", {"n_components": 3})
        assert isinstance(r, PCAReducer)
        assert p["n_components"] == 3

    def test_pca_no_components_raises(self):
        with pytest.raises(ValueError):
            create_reducer("pca", {})

    def test_svd(self):
        r, p = create_reducer("svd", {"n_components": 2})
        assert isinstance(r, SVDReducer)

    def test_svd_no_components_raises(self):
        with pytest.raises(ValueError):
            create_reducer("svd", {})

    def test_spca(self):
        r, p = create_reducer("spca", {"n_components": 2, "alpha": 0.5})
        assert isinstance(r, SparsePCAReducer)

    def test_kpca(self):
        r, p = create_reducer(
            "kpca", {"n_components": 2, "kernel": "poly", "gamma": 0.1, "degree": 2}
        )
        assert isinstance(r, KPCAReducer)
        assert p["kernel"] == "poly"

    def test_kpca_gamma_none(self):
        r, p = create_reducer("kpca", {"n_components": 2, "gamma": "none"})
        assert p["gamma"] is None

    def test_isomap(self):
        r, p = create_reducer("isomap", {"n_components": 2, "n_neighbors": 7})
        assert isinstance(r, IsomapReducer)

    def test_laplacian(self):
        r, p = create_reducer("laplacian", {"n_components": 2, "n_neighbors": 8})
        assert isinstance(r, LaplacianReducer)

    def test_tsne(self):
        r, p = create_reducer(
            "tsne", {"n_components": 2, "perplexity": 10, "n_iter": 300}
        )
        assert isinstance(r, TSNEReducer)

    def test_lda_raises(self):
        with pytest.raises(RuntimeError, match="supervised"):
            create_reducer("lda")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_reducer("invalid_method")

    def test_deep_diffusion_maps_raises(self):
        with pytest.raises(RuntimeError):
            create_reducer("deep_diffusion_maps")

    def test_dreams_raises(self):
        with pytest.raises(RuntimeError):
            create_reducer("dreams")

    def test_pcc_raises(self):
        with pytest.raises(RuntimeError):
            create_reducer("pcc")


class TestListDimredMethods:
    def test_returns_dict(self):
        methods = list_dimred_methods()
        assert isinstance(methods, dict)
        assert "pca" in methods
        assert "none" in methods
        assert methods["none"]["available"] is True

    def test_all_have_description(self):
        for name, info in list_dimred_methods().items():
            assert "description" in info
            assert "available" in info
