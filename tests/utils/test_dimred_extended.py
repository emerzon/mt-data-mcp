"""Extended coverage tests for utils/dimred.py targeting uncovered lines."""
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from mtdata.utils.dimred import (
    DimReducer,
    IsomapReducer,
    KPCAReducer,
    LaplacianReducer,
    NoneReducer,
    PCAReducer,
    SparsePCAReducer,
    SVDReducer,
    TSNEReducer,
    create_reducer,
    list_dimred_methods,
)


def _data(n=60, d=5, seed=7):
    return np.random.RandomState(seed).randn(n, d).astype(np.float32)


# ===== DiffusionMapsReducer (lines 205-243) ================================

class TestDiffusionMapsReducerInit:
    """Lines 204-218: __init__ with various params."""

    def test_missing_pydiffmap_raises(self):
        with patch("mtdata.utils.dimred._DMap", None):
            from mtdata.utils.dimred import DiffusionMapsReducer
            with pytest.raises(RuntimeError, match="pydiffmap"):
                DiffusionMapsReducer(n_components=2)

    def test_init_with_epsilon_and_k(self):
        """Lines 214-218: epsilon and k kwargs."""
        mock_dmap_mod = MagicMock()
        mock_model = MagicMock()
        mock_dmap_mod.DiffusionMap.return_value = mock_model
        with patch("mtdata.utils.dimred._DMap", mock_dmap_mod):
            from mtdata.utils.dimred import DiffusionMapsReducer
            r = DiffusionMapsReducer(n_components=3, alpha=0.7, epsilon=1.5, k=10)
            assert r.n_components == 3
            assert r.alpha == 0.7
            assert r.epsilon == 1.5
            assert r.k == 10

    def test_info(self):
        mock_dmap_mod = MagicMock()
        with patch("mtdata.utils.dimred._DMap", mock_dmap_mod):
            from mtdata.utils.dimred import DiffusionMapsReducer
            r = DiffusionMapsReducer(n_components=2, alpha=0.5, epsilon=None, k=None)
            info = r.info()
            assert info["method"] == "diffusion"
            assert info["epsilon"] is None
            assert info["k"] is None


class TestDiffusionMapsReducerFitTransform:
    """Lines 220-240: fit, transform, fit_transform branches."""

    def _make_reducer(self):
        mock_dmap_mod = MagicMock()
        mock_model = MagicMock()
        mock_dmap_mod.DiffusionMap.return_value = mock_model
        with patch("mtdata.utils.dimred._DMap", mock_dmap_mod):
            from mtdata.utils.dimred import DiffusionMapsReducer
            return DiffusionMapsReducer(n_components=2), mock_model

    def test_fit(self):
        """Line 222."""
        r, model = self._make_reducer()
        r.fit(_data(20, 4))
        model.fit.assert_called_once()

    def test_transform_no_attr_raises(self):
        """Lines 227-228: no transform attr → RuntimeError."""
        r, model = self._make_reducer()
        del model.transform
        with pytest.raises(RuntimeError, match="does not support"):
            r.transform(_data(10, 4))

    def test_transform_success(self):
        """Lines 225-229."""
        r, model = self._make_reducer()
        model.transform.return_value = np.zeros((10, 2))
        out = r.transform(_data(10, 4))
        assert out.dtype == np.float32

    def test_fit_transform_has_method(self):
        """Lines 232-233: model has fit_transform."""
        r, model = self._make_reducer()
        model.fit_transform.return_value = np.ones((20, 2))
        out = r.fit_transform(_data(20, 4))
        np.testing.assert_array_equal(out, np.ones((20, 2), dtype=np.float32))

    def test_fit_transform_fallback_evecs(self):
        """Lines 234-238: no fit_transform, use _evecs."""
        r, model = self._make_reducer()
        del model.fit_transform
        model._evecs = np.random.randn(20, 5)
        out = r.fit_transform(_data(20, 4))
        assert out.shape == (20, 2)
        assert out.dtype == np.float32

    def test_fit_transform_fallback_transform(self):
        """Lines 239-240: no fit_transform, no _evecs → transform."""
        r, model = self._make_reducer()
        del model.fit_transform
        # Also ensure no _evecs so it falls through to transform
        if hasattr(model, "_evecs"):
            del model._evecs
        model.transform.return_value = np.zeros((20, 2))
        out = r.fit_transform(_data(20, 4))
        assert out.shape == (20, 2)


# ===== UMAPReducer (lines 269-277) =========================================

class TestUMAPReducerInit:
    def test_missing_umap_raises(self):
        with patch("mtdata.utils.dimred._UMAP", None):
            from mtdata.utils.dimred import UMAPReducer
            with pytest.raises(RuntimeError, match="umap-learn"):
                UMAPReducer(n_components=2)

    def test_info(self):
        mock_umap = MagicMock()
        with patch("mtdata.utils.dimred._UMAP", mock_umap):
            from mtdata.utils.dimred import UMAPReducer
            r = UMAPReducer(n_components=3, n_neighbors=10, min_dist=0.2)
            info = r.info()
            assert info["method"] == "umap"
            assert info["n_components"] == 3
            assert info["n_neighbors"] == 10
            assert info["min_dist"] == 0.2


# ===== TSNEReducer (lines 280-314) =========================================

class TestTSNEReducerExtended:
    """Cover all TSNEReducer methods without hitting deprecated param issue."""

    def test_supports_transform_false(self):
        """Line 293-294."""
        mock_tsne_cls = MagicMock()
        with patch("mtdata.utils.dimred._SKTSNE", mock_tsne_cls):
            from mtdata.utils.dimred import TSNEReducer
            r = TSNEReducer(n_components=2, perplexity=10.0, n_iter=300)
            assert r.supports_transform() is False

    def test_fit_returns_self(self):
        """Lines 296-298."""
        mock_tsne_cls = MagicMock()
        with patch("mtdata.utils.dimred._SKTSNE", mock_tsne_cls):
            from mtdata.utils.dimred import TSNEReducer
            r = TSNEReducer(n_components=2)
            assert r.fit(_data()) is r

    def test_transform_raises(self):
        """Lines 300-301."""
        mock_tsne_cls = MagicMock()
        with patch("mtdata.utils.dimred._SKTSNE", mock_tsne_cls):
            from mtdata.utils.dimred import TSNEReducer
            r = TSNEReducer(n_components=2)
            with pytest.raises(RuntimeError, match="does not support"):
                r.transform(_data())

    def test_fit_transform_delegates(self):
        """Lines 303-304."""
        mock_tsne_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.fit_transform.return_value = np.zeros((60, 2))
        mock_tsne_cls.return_value = mock_model
        with patch("mtdata.utils.dimred._SKTSNE", mock_tsne_cls):
            from mtdata.utils.dimred import TSNEReducer
            r = TSNEReducer(n_components=2)
            out = r.fit_transform(_data())
            assert out.shape == (60, 2)

    def test_info_fields(self):
        """Lines 306-314."""
        mock_tsne_cls = MagicMock()
        with patch("mtdata.utils.dimred._SKTSNE", mock_tsne_cls):
            from mtdata.utils.dimred import TSNEReducer
            r = TSNEReducer(n_components=2, perplexity=15.0, learning_rate=100.0, n_iter=500)
            info = r.info()
            assert info["method"] == "tsne"
            assert info["perplexity"] == 15.0
            assert info["learning_rate"] == 100.0
            assert info["n_iter"] == 500
            assert info["supports_transform"] is False

    def test_missing_sklearn_raises(self):
        """Line 285."""
        with patch("mtdata.utils.dimred._SKTSNE", None):
            from mtdata.utils.dimred import TSNEReducer
            with pytest.raises(RuntimeError, match="scikit-learn"):
                TSNEReducer(n_components=2)


# ===== DreamsCNEReducer (lines 339-429) ====================================

class TestDreamsCNEReducerInit:
    """Lines 339-354: __init__."""

    def test_missing_cne_raises(self):
        with patch("mtdata.utils.dimred._CNE", None):
            from mtdata.utils.dimred import DreamsCNEReducer
            with pytest.raises(RuntimeError, match="DREAMS-CNE"):
                DreamsCNEReducer()

    def test_init_params(self):
        mock_cne = MagicMock()
        with patch("mtdata.utils.dimred._CNE", mock_cne):
            from mtdata.utils.dimred import DreamsCNEReducer
            r = DreamsCNEReducer(
                n_components=3, k=10, negative_samples=200,
                n_epochs=100, batch_size=512, learning_rate=0.01,
                parametric=False, device="cpu",
                regularizer=False, reg_lambda=1e-3, reg_scaling="std",
                seed=42,
            )
            assert r.n_components == 3
            assert r.parametric is False
            assert r.seed == 42


class TestDreamsCNEReducerMethods:
    """Lines 356-429."""

    def _make_reducer(self, parametric=True, regularizer=True):
        mock_cne = MagicMock()
        with patch("mtdata.utils.dimred._CNE", mock_cne):
            from mtdata.utils.dimred import DreamsCNEReducer
            r = DreamsCNEReducer(
                n_components=2, parametric=parametric,
                regularizer=regularizer, reg_embedding=None,
            )
        r._cne_mod = mock_cne
        return r, mock_cne

    def test_supports_transform_parametric(self):
        """Lines 356-358."""
        r, _ = self._make_reducer(parametric=True)
        assert r.supports_transform() is True

    def test_supports_transform_non_parametric(self):
        r, _ = self._make_reducer(parametric=False)
        assert r.supports_transform() is False

    def test_transform_not_fitted_raises(self):
        """Lines 403-404."""
        r, _ = self._make_reducer()
        with pytest.raises(RuntimeError, match="not fitted"):
            r.transform(_data(10, 4))

    def test_transform_tuple_result(self):
        """Lines 408-409: transform returns tuple."""
        r, _ = self._make_reducer()
        mock_embedder = MagicMock()
        mock_embedder.transform.return_value = (np.zeros((10, 2)), "extra")
        r._embedder = mock_embedder
        out = r.transform(_data(10, 4))
        assert out.shape == (10, 2)

    def test_transform_array_result(self):
        """Lines 406-410."""
        r, _ = self._make_reducer()
        mock_embedder = MagicMock()
        mock_embedder.transform.return_value = np.ones((10, 2))
        r._embedder = mock_embedder
        out = r.transform(_data(10, 4))
        np.testing.assert_array_equal(out, np.ones((10, 2), dtype=np.float32))

    def test_fit_transform(self):
        """Lines 412-414."""
        mock_cne = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.transform.return_value = np.zeros((60, 2))
        mock_cne.CNE.return_value = mock_embedder
        with patch("mtdata.utils.dimred._CNE", mock_cne):
            from mtdata.utils.dimred import DreamsCNEReducer
            r = DreamsCNEReducer(n_components=2, parametric=True, regularizer=False)
            out = r.fit_transform(_data())
        assert out.shape == (60, 2)

    def test_info(self):
        """Lines 416-429."""
        r, _ = self._make_reducer()
        info = r.info()
        assert info["method"] == "dreams_cne"
        assert "n_components" in info
        assert "k" in info
        assert "regularizer" in info


# ===== create_reducer factory (lines 480-553) ==============================

class TestCreateReducerExtended:
    def test_umap(self):
        """Lines 480-484."""
        mock_umap = MagicMock()
        with patch("mtdata.utils.dimred._UMAP", mock_umap):
            r, p = create_reducer("umap", {"n_components": 3, "n_neighbors": 10, "min_dist": 0.05})
            assert p["method"] == "umap"
            assert p["n_components"] == 3

    def test_diffusion(self):
        """Lines 485-493."""
        mock_dmap = MagicMock()
        with patch("mtdata.utils.dimred._DMap", mock_dmap):
            r, p = create_reducer("diffusion", {"n_components": 3, "alpha": 0.7, "epsilon": 1.5, "k": 8})
            assert p["method"] == "diffusion"
            assert p["n_components"] == 3
            assert p["epsilon"] == 1.5
            assert p["k"] == 8

    def test_diffusion_null_epsilon_and_k(self):
        """Lines 489-491: null string values."""
        mock_dmap = MagicMock()
        with patch("mtdata.utils.dimred._DMap", mock_dmap):
            r, p = create_reducer("diffusion", {"epsilon": "null", "k": "none"})
            assert p["epsilon"] is None
            assert p["k"] is None

    def test_dreams_cne(self):
        """Lines 494-522."""
        mock_cne = MagicMock()
        with patch("mtdata.utils.dimred._CNE", mock_cne):
            r, p = create_reducer("dreams_cne", {
                "n_components": 3, "k": 10, "negative_samples": 100,
                "n_epochs": 50, "batch_size": 512, "learning_rate": 0.005,
                "parametric": True, "device": "cpu",
                "regularizer": False, "reg_lambda": 1e-3, "reg_scaling": "std",
            })
            assert p["method"] == "dreams_cne"

    def test_dreams_cne_missing_raises(self):
        """Lines 495-496."""
        with patch("mtdata.utils.dimred._CNE", None):
            with pytest.raises(RuntimeError, match="DREAMS-CNE"):
                create_reducer("dreams_cne")

    def test_dreams_cne_fast(self):
        """Lines 523-541."""
        mock_cne = MagicMock()
        with patch("mtdata.utils.dimred._CNE", mock_cne):
            r, p = create_reducer("dreams_cne_fast", {"n_components": 3})
            assert p["method"] == "dreams_cne"

    def test_dreams_cne_fast_missing_raises(self):
        """Lines 524-525."""
        with patch("mtdata.utils.dimred._CNE", None):
            with pytest.raises(RuntimeError, match="DREAMS-CNE"):
                create_reducer("dreams_cne_fast")

    def test_tsne_factory(self):
        """Lines 542-548."""
        mock_tsne = MagicMock()
        with patch("mtdata.utils.dimred._SKTSNE", mock_tsne):
            r, p = create_reducer("tsne", {"n_components": 3, "perplexity": 20.0, "learning_rate": 150.0, "n_iter": 800})
            assert p["method"] == "tsne"
            assert p["n_components"] == 3

    def test_null_false_string_returns_none(self):
        """Line 439: 'null' and 'false' strings → NoneReducer."""
        for s in ("null", "false", "False", "NULL"):
            r, p = create_reducer(s)
            assert isinstance(r, NoneReducer)


# ===== list_dimred_methods (lines 565-585) =================================

class TestListDimredMethodsExtended:
    def test_all_methods_present(self):
        methods = list_dimred_methods()
        expected = {"none", "pca", "svd", "spca", "kpca", "isomap", "laplacian",
                    "umap", "diffusion", "tsne", "lda", "dreams_cne",
                    "dreams_cne_fast", "deep_diffusion_maps", "dreams", "pcc"}
        assert expected.issubset(set(methods.keys()))

    def test_unavailable_methods(self):
        methods = list_dimred_methods()
        assert methods["deep_diffusion_maps"]["available"] is False
        assert methods["dreams"]["available"] is False
        assert methods["pcc"]["available"] is False
