from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Dict, Optional, Tuple

import numpy as np


class _SkModelMixin:
    """Mixin providing fit/transform helpers for sklearn-like models on self._model.

    Expects subclasses to set `self._model` in __init__ and may override
    supports_transform() when the underlying model cannot transform new samples.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):  # type: ignore[override]
        self._model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self._model.transform(X), dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self._model.fit_transform(X), dtype=np.float32)

@lru_cache(maxsize=None)
def _optional_dependency(module_name: str, attribute: Optional[str] = None) -> Any:
    """Import an optional backend on first use and cache its resolved object."""
    try:
        module = import_module(module_name)
        return getattr(module, attribute) if attribute else module
    except Exception:
        return None


@lru_cache(maxsize=None)
def _dependency_available(module_name: str) -> bool:
    """Probe package metadata without importing an optional runtime."""
    try:
        return find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _require_dependency(
    module_name: str,
    attribute: Optional[str],
    unavailable_message: str,
) -> Any:
    dependency = _optional_dependency(module_name, attribute)
    if dependency is None:
        raise RuntimeError(unavailable_message)
    return dependency


class DimReducer:
    """Abstract interface for dimensionality reducers.

    Implementations should support fit, transform, and fit_transform. For some
    algorithms (e.g., t-SNE), transform on new samples is not supported; in such
    cases `supports_transform` should return False.
    """

    name: str = "none"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DimReducer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def supports_transform(self) -> bool:
        return True

    def info(self) -> Dict[str, Any]:
        return {"method": self.name}


class NoneReducer(DimReducer):
    name = "none"

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


class PCAReducer(_SkModelMixin, DimReducer):
    name = "pca"

    def __init__(self, n_components: int) -> None:
        model_cls = _require_dependency(
            "sklearn.decomposition",
            "PCA",
            "scikit-learn not available; cannot use PCA",
        )
        self.n_components = int(max(1, n_components))
        self._model = model_cls(
            n_components=self.n_components,
            svd_solver="auto",
            whiten=False,
        )

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components)}


class SVDReducer(_SkModelMixin, DimReducer):
    name = "svd"

    def __init__(self, n_components: int) -> None:
        model_cls = _require_dependency(
            "sklearn.decomposition",
            "TruncatedSVD",
            "scikit-learn not available; cannot use TruncatedSVD",
        )
        self.n_components = int(max(1, n_components))
        self._model = model_cls(n_components=self.n_components)

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components)}


class SparsePCAReducer(_SkModelMixin, DimReducer):
    name = "spca"

    def __init__(self, n_components: int = 2, alpha: float = 1.0) -> None:
        model_cls = _require_dependency(
            "sklearn.decomposition",
            "SparsePCA",
            "scikit-learn not available; cannot use SparsePCA",
        )
        self.n_components = int(max(1, n_components))
        self.alpha = float(alpha)
        self._model = model_cls(n_components=self.n_components, alpha=self.alpha)

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components), "alpha": float(self.alpha)}


class KPCAReducer(_SkModelMixin, DimReducer):
    name = "kpca"

    def __init__(self, n_components: int = 2, kernel: str = "rbf", gamma: Optional[float] = None, degree: int = 3, coef0: float = 1.0) -> None:
        model_cls = _require_dependency(
            "sklearn.decomposition",
            "KernelPCA",
            "scikit-learn not available; cannot use KernelPCA",
        )
        self.n_components = int(max(1, n_components))
        self.kernel = str(kernel)
        self.gamma = None if gamma is None else float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self._model = model_cls(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0, fit_inverse_transform=False)

    def info(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_components": int(self.n_components),
            "kernel": str(self.kernel),
            "gamma": None if self.gamma is None else float(self.gamma),
            "degree": int(self.degree),
            "coef0": float(self.coef0),
        }


class LaplacianReducer(DimReducer):
    name = "laplacian"

    def __init__(self, n_components: int = 2, n_neighbors: int = 10) -> None:
        model_cls = _require_dependency(
            "sklearn.manifold",
            "SpectralEmbedding",
            "scikit-learn not available; cannot use SpectralEmbedding",
        )
        self.n_components = int(max(1, n_components))
        self.n_neighbors = int(max(1, n_neighbors))
        self._model = model_cls(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
        )

    def supports_transform(self) -> bool:
        return False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LaplacianReducer":
        self._model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("SpectralEmbedding does not support transforming new samples; use 'pca' or 'umap'")

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(self._model.fit_transform(X), dtype=np.float32)

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components), "n_neighbors": int(self.n_neighbors), "supports_transform": False}


class DiffusionMapsReducer(DimReducer):
    name = "diffusion"

    def __init__(self, n_components: int = 2, alpha: float = 0.5, epsilon: Optional[float] = None, k: Optional[int] = None) -> None:
        dmap_module = _require_dependency(
            "pydiffmap",
            "diffusion_map",
            "pydiffmap not available; `pip install pydiffmap` to use diffusion maps",
        )
        self.n_components = int(max(1, n_components))
        self.alpha = float(alpha)
        self.epsilon = None if epsilon is None else float(epsilon)
        self.k = None if k is None else int(k)
        # Construct DiffusionMap model; pydiffmap API
        # Use kwargs only if provided to avoid overriding library defaults
        kwargs: Dict[str, Any] = {"alpha": self.alpha}
        if self.epsilon is not None:
            kwargs["epsilon"] = self.epsilon
        if self.k is not None:
            kwargs["k"] = self.k
        self._model = dmap_module.DiffusionMap(n_evecs=self.n_components, **kwargs)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DiffusionMapsReducer":
        # Fit is equivalent to computing eigenvectors on training data
        self._model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Nyström extension for new samples
        if not hasattr(self._model, "transform"):
            raise RuntimeError("This diffusion map implementation does not support transforming new samples")
        return np.asarray(self._model.transform(X), dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if hasattr(self._model, "fit_transform"):
            return np.asarray(self._model.fit_transform(X), dtype=np.float32)
        self.fit(X)
        # If fit_transform not available, attempt eigenvectors on fitted model
        if hasattr(self._model, "_evecs"):
            Y = np.asarray(self._model._evecs[:, : self.n_components])
            return Y.astype(np.float32)
        # Fallback to transform itself
        return self.transform(X)

    def info(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_components": int(self.n_components),
            "alpha": float(self.alpha),
            "epsilon": None if self.epsilon is None else float(self.epsilon),
            "k": None if self.k is None else int(self.k),
        }

class IsomapReducer(_SkModelMixin, DimReducer):
    name = "isomap"

    def __init__(self, n_components: int = 2, n_neighbors: int = 5) -> None:
        model_cls = _require_dependency(
            "sklearn.manifold",
            "Isomap",
            "scikit-learn not available; cannot use Isomap",
        )
        self.n_components = int(max(1, n_components))
        self.n_neighbors = int(max(1, n_neighbors))
        self._model = model_cls(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
        )

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components), "n_neighbors": int(self.n_neighbors)}


class UMAPReducer(_SkModelMixin, DimReducer):
    name = "umap"

    def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> None:
        model_cls = _require_dependency(
            "umap",
            "UMAP",
            "umap-learn not available; `pip install umap-learn`",
        )
        self.n_components = int(max(1, n_components))
        self.n_neighbors = int(max(1, n_neighbors))
        self.min_dist = float(min_dist)
        self._model = model_cls(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
        )

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components), "n_neighbors": int(self.n_neighbors), "min_dist": float(self.min_dist)}


class TSNEReducer(DimReducer):
    name = "tsne"

    def __init__(self, n_components: int = 2, perplexity: float = 30.0, learning_rate: float = 200.0, n_iter: int = 1000) -> None:
        model_cls = _require_dependency(
            "sklearn.manifold",
            "TSNE",
            "scikit-learn not available; cannot use TSNE",
        )
        self.n_components = int(max(1, n_components))
        self.perplexity = float(perplexity)
        self.learning_rate = float(learning_rate)
        self.n_iter = int(max(250, n_iter))
        import inspect
        _tsne_params = inspect.signature(model_cls).parameters
        iter_kwarg = "max_iter" if "max_iter" in _tsne_params else "n_iter"
        self._model = model_cls(n_components=self.n_components, perplexity=self.perplexity, learning_rate=self.learning_rate, init="pca", **{iter_kwarg: self.n_iter})

    def supports_transform(self) -> bool:
        # sklearn TSNE does not support transforming new samples
        return False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TSNEReducer":
        # Fit returns self; TSNE computes embedding in fit_transform
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("TSNE does not support transforming new samples; use 'pca' or 'umap' instead")

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(self._model.fit_transform(X), dtype=np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_components": int(self.n_components),
            "perplexity": float(self.perplexity),
            "learning_rate": float(self.learning_rate),
            "n_iter": int(self.n_iter),
            "supports_transform": False,
        }


class DreamsCNEReducer(DimReducer):
    name = "dreams_cne"

    def __init__(
        self,
        n_components: int = 2,
        # CNE graph / training params
        k: int = 15,
        negative_samples: int = 500,
        n_epochs: int = 250,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        parametric: bool = True,
        device: str = "auto",
        # DREAMS regularizer params
        regularizer: bool = True,
        reg_lambda: float = 5e-4,
        reg_scaling: str = "norm",
        # Optional: explicit reg_embedding, else computed from PCA(X)
        reg_embedding: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> None:
        self._cne_module = _require_dependency(
            "cne",
            None,
            "DREAMS-CNE not available; `pip install git+https://github.com/berenslab/DREAMS-CNE@tp` and its deps",
        )
        self.n_components = int(max(1, n_components))
        self.k = int(max(1, k))
        self.negative_samples = int(max(1, negative_samples))
        self.n_epochs = int(max(1, n_epochs))
        self.batch_size = int(max(32, batch_size))
        self.learning_rate = float(learning_rate)
        self.parametric = bool(parametric)
        self.device = str(device)
        self.regularizer = bool(regularizer)
        self.reg_lambda = float(reg_lambda)
        self.reg_scaling = str(reg_scaling)
        self.reg_embedding = reg_embedding  # may be None; computed at fit
        self.seed = int(seed)
        self._model = None  # type: ignore

    def supports_transform(self) -> bool:
        # Only the parametric variant can transform new samples efficiently
        return bool(self.parametric)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DreamsCNEReducer":
        X = np.asarray(X, dtype=np.float32)
        # Compute default reg embedding if needed (PCA to target dims, scaled by std of first component)
        reg_emb = self.reg_embedding
        if self.regularizer and reg_emb is None:
            try:
                # Use PCA with n_components matching embedding size
                from sklearn.decomposition import (
                    PCA as _SKPCA_local,  # local import to avoid hard dep if unused
                )
                pca = _SKPCA_local(n_components=self.n_components)
                reg_emb = pca.fit_transform(X)
                if reg_emb.shape[1] >= 1:
                    s = float(np.std(reg_emb[:, 0]))
                    if s > 0:
                        reg_emb = reg_emb / s
            except Exception:
                # Fallback: first n_components columns or zeros if insufficient dims
                reg_emb = X[:, : self.n_components]
        # Build CNE embedder
        kwargs: Dict[str, Any] = dict(
            negative_samples=self.negative_samples,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
            regularizer=self.regularizer,
            reg_lambda=self.reg_lambda,
            reg_scaling=self.reg_scaling,
        )
        if self.regularizer and reg_emb is not None:
            kwargs["reg_embedding"] = np.asarray(reg_emb, dtype=np.float32)
        self._embedder = self._cne_module.CNE(
            k=self.k,
            parametric=self.parametric,
            decoder=False,
            embd_dim=self.n_components,
            seed=self.seed,
            **kwargs,
        )
        # Fit model; we don't need graph override here
        self._embedder.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_embedder") or self._embedder is None:
            raise RuntimeError("DREAMS-CNE reducer not fitted")
        # Parametric transform returns new embeddings; non-param returns train embeddings only (we forbid by supports_transform)
        emb = self._embedder.transform(np.asarray(X, dtype=np.float32))
        # If decoder mode were used, transform may return tuple; we don’t use decoder here
        if isinstance(emb, tuple):
            emb = emb[0]
        return np.asarray(emb, dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def info(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_components": int(self.n_components),
            "k": int(self.k),
            "negative_samples": int(self.negative_samples),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "parametric": bool(self.parametric),
            "regularizer": bool(self.regularizer),
            "reg_lambda": float(self.reg_lambda),
            "reg_scaling": str(self.reg_scaling),
        }

def create_reducer(method: Optional[str], params: Optional[Dict[str, Any]] = None) -> Tuple[DimReducer, Dict[str, Any]]:
    """Factory: create a dimensionality reducer from a method string and params.

    method: one of None/'none', 'pca', 'svd', 'umap', 'isomap', 'tsne'.
    params: dict of keyword args relevant to the method.

    Returns: (reducer_instance, effective_params)
    """
    if not method or str(method).lower() in ("none", "null", "false"):
        return NoneReducer(), {"method": "none"}
    m = str(method).lower().strip()
    p = dict(params or {})
    if m == "pca":
        n = int(p.get("n_components", p.get("components", 0) or 0))
        if n <= 0:
            raise ValueError("PCA requires a positive n_components")
        r = PCAReducer(n)
        return r, r.info()
    if m == "svd":
        n = int(p.get("n_components", p.get("components", 0) or 0))
        if n <= 0:
            raise ValueError("SVD requires a positive n_components")
        r = SVDReducer(n)
        return r, r.info()
    if m == "spca":
        n = int(p.get("n_components", 2))
        alpha = float(p.get("alpha", 1.0))
        r = SparsePCAReducer(n_components=n, alpha=alpha)
        return r, r.info()
    if m == "kpca":
        n = int(p.get("n_components", 2))
        kernel = str(p.get("kernel", "rbf"))
        gamma = p.get("gamma", None)
        gamma = None if gamma in (None, "none", "null", "") else float(gamma)
        degree = int(p.get("degree", 3))
        coef0 = float(p.get("coef0", 1.0))
        r = KPCAReducer(n_components=n, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        return r, r.info()
    if m == "isomap":
        n = int(p.get("n_components", 2))
        k = int(p.get("n_neighbors", 5))
        r = IsomapReducer(n_components=n, n_neighbors=k)
        return r, r.info()
    if m == "laplacian":
        n = int(p.get("n_components", 2))
        k = int(p.get("n_neighbors", 10))
        r = LaplacianReducer(n_components=n, n_neighbors=k)
        return r, r.info()
    if m == "umap":
        n = int(p.get("n_components", 2))
        k = int(p.get("n_neighbors", 15))
        md = float(p.get("min_dist", 0.1))
        r = UMAPReducer(n_components=n, n_neighbors=k, min_dist=md)
        return r, r.info()
    if m == "diffusion":
        n = int(p.get("n_components", 2))
        alpha = float(p.get("alpha", 0.5))
        eps = p.get("epsilon", None)
        eps = None if eps in (None, "none", "null", "") else float(eps)
        k = p.get("k", None)
        k = None if k in (None, "none", "null", "") else int(k)
        r = DiffusionMapsReducer(n_components=n, alpha=alpha, epsilon=eps, k=k)
        return r, r.info()
    if m == "dreams_cne":
        # Map common params with sane defaults
        n = int(p.get("n_components", 2))
        k = int(p.get("k", 15))
        neg = int(p.get("negative_samples", 500))
        epochs = int(p.get("n_epochs", 250))
        bs = int(p.get("batch_size", 4096))
        lr = float(p.get("learning_rate", 1e-3))
        parametric = bool(p.get("parametric", True))
        device = str(p.get("device", "auto"))
        regularizer = bool(p.get("regularizer", True))
        reg_lambda = float(p.get("reg_lambda", 5e-4))
        reg_scaling = str(p.get("reg_scaling", "norm"))
        r = DreamsCNEReducer(
            n_components=n,
            k=k,
            negative_samples=neg,
            n_epochs=epochs,
            batch_size=bs,
            learning_rate=lr,
            parametric=parametric,
            device=device,
            regularizer=regularizer,
            reg_lambda=reg_lambda,
            reg_scaling=reg_scaling,
        )
        return r, r.info()
    if m == "dreams_cne_fast":
        # Tuned for speed on moderate index sizes
        n = int(p.get("n_components", 2))
        r = DreamsCNEReducer(
            n_components=n,
            k=int(p.get("k", 10)),
            negative_samples=int(p.get("negative_samples", 200)),
            n_epochs=int(p.get("n_epochs", 60)),
            batch_size=int(p.get("batch_size", 1024)),
            learning_rate=float(p.get("learning_rate", 5e-3)),
            parametric=bool(p.get("parametric", True)),
            device=str(p.get("device", "auto")),
            regularizer=bool(p.get("regularizer", True)),
            reg_lambda=float(p.get("reg_lambda", 5e-4)),
            reg_scaling=str(p.get("reg_scaling", "norm")),
        )
        return r, r.info()
    if m == "tsne":
        n = int(p.get("n_components", 2))
        perplexity = float(p.get("perplexity", 30.0))
        lr = float(p.get("learning_rate", 200.0))
        iters = int(p.get("n_iter", 1000))
        r = TSNEReducer(n_components=n, perplexity=perplexity, learning_rate=lr, n_iter=iters)
        return r, r.info()
    if m == "lda":
        # LDA is supervised and requires class labels (y) to fit;
        # pattern_search does not provide labels, so we error out here with guidance.
        _require_dependency(
            "sklearn.discriminant_analysis",
            "LinearDiscriminantAnalysis",
            "scikit-learn not available; cannot use LDA",
        )
        raise RuntimeError("LDA is supervised and requires labels; not supported for unsupervised pattern search")
    if m == "deep_diffusion_maps":
        # Placeholder for research implementations
        raise RuntimeError("Deep Diffusion Maps not available. Provide an implementation or plugin.")
    if m == "dreams":
        raise RuntimeError("DREAMS (Dimensionality Reduction Enhanced Across Multiple Scales) not available. Provide an implementation or plugin.")
    if m == "pcc":
        raise RuntimeError("PCC (Preserving Clusters and Correlations) not available. Provide an implementation or plugin.")
    raise ValueError(f"Unknown dimensionality reduction method: {method}")


def list_dimred_methods() -> Dict[str, Dict[str, Any]]:
    """Return available dimension reduction methods and availability flags."""
    sklearn_available = _dependency_available("sklearn")
    cne_available = _dependency_available("cne")
    out: Dict[str, Dict[str, Any]] = {
        "none": {"available": True, "description": "No reduction; pass-through."},
        "pca": {"available": sklearn_available, "description": "Principal Component Analysis (sklearn)."},
        "svd": {"available": sklearn_available, "description": "Truncated SVD (sklearn)."},
        "spca": {"available": sklearn_available, "description": "Sparse PCA (sklearn)."},
        "kpca": {"available": sklearn_available, "description": "Kernel PCA (sklearn)."},
        "isomap": {"available": sklearn_available, "description": "Isomap manifold learning (sklearn)."},
        "laplacian": {"available": sklearn_available, "description": "Laplacian Eigenmaps / Spectral Embedding (sklearn)."},
        "umap": {"available": _dependency_available("umap"), "description": "UMAP dimensionality reduction (umap-learn)."},
        "diffusion": {"available": _dependency_available("pydiffmap"), "description": "Diffusion Maps (pydiffmap)."},
        "tsne": {"available": sklearn_available, "description": "t-SNE (sklearn); no transform for new samples."},
        "lda": {"available": sklearn_available, "description": "Linear Discriminant Analysis (supervised; requires labels)."},
        "dreams_cne": {"available": cne_available, "description": "DREAMS-CNE (parametric supports transform; heavy Torch training)."},
        "dreams_cne_fast": {"available": cne_available, "description": "DREAMS-CNE with faster defaults (smaller k/epochs/batch)."},
        "deep_diffusion_maps": {"available": False, "description": "Deep Diffusion Maps (research; plugin required)."},
        "dreams": {"available": False, "description": "DREAMS (Across Multiple Scales) (research; plugin required)."},
        "pcc": {"available": False, "description": "Preserving Clusters and Correlations (research; plugin required)."},
    }
    return out
