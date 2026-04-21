"""Tests for mtdata.forecast.methods.pretrained – coverage for lines 96-268, 296-492, 525-702."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub heavy third-party packages before importing the module under test
# ---------------------------------------------------------------------------

def _make_module(name: str, attrs: dict | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    return mod


# --- torch stub -----------------------------------------------------------
_torch = _make_module("torch")
_torch.float32 = "float32"
_torch.device = MagicMock(side_effect=lambda x: x)
_torch_cuda = _make_module("torch.cuda")
_torch_cuda.is_available = MagicMock(return_value=False)
_torch.cuda = _torch_cuda
_torch.load = MagicMock(return_value={"hyper_parameters": {"model_kwargs": {}}})

class _FakeTensor:
    def __init__(self, data=None, dtype=None):
        self._data = np.array(data) if data is not None else np.zeros(1)
    def unsqueeze(self, dim):
        return self
    def to(self, device):
        return self
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self._data
    @property
    def shape(self):
        return self._data.shape

_torch.tensor = lambda data, dtype=None: _FakeTensor(data, dtype)
_torch_serial = _make_module("torch.serialization")
_torch_serial.add_safe_globals = None
_torch.serialization = _torch_serial

# --- chronos stub ----------------------------------------------------------
_chronos = _make_module("chronos")

class _FakePipeline:
    @classmethod
    def from_pretrained(cls, model_name, device_map=None):
        return cls()

    def predict_quantiles(self, ctx, prediction_length=10, quantile_levels=None, **kw):
        h = prediction_length
        n_q = len(quantile_levels) if quantile_levels else 1
        q_tensor = _FakeTensor(np.random.rand(1, h, n_q))
        m_tensor = _FakeTensor(np.random.rand(1, h))
        return q_tensor, m_tensor

    def predict(self, ctx, prediction_length=10, **kw):
        return _FakeTensor(np.random.rand(1, prediction_length))

_chronos.ChronosBoltPipeline = _FakePipeline
_chronos.ChronosPipeline = _FakePipeline

# --- timesfm stub ----------------------------------------------------------
_timesfm = _make_module("timesfm")

class _FakeForecastConfig:
    def __init__(self, **kw):
        pass

_timesfm.ForecastConfig = _FakeForecastConfig

class _FakeTimesFMTorch:
    def __init__(self, **kw):
        pass
    def load_checkpoint(self):
        pass
    def compile(self, cfg):
        pass
    def forecast(self, inputs, horizon):
        arr = np.ones((1, horizon))
        return arr, None

_timesfm_torch = _make_module("timesfm.torch")
_timesfm_torch.TimesFM_2p5_200M_torch = _FakeTimesFMTorch
_timesfm_2p5 = _make_module("timesfm.timesfm_2p5")
_timesfm_2p5_torch = _make_module("timesfm.timesfm_2p5.timesfm_2p5_torch")
_timesfm_configs = _make_module("timesfm.configs")
_timesfm_configs.ForecastConfig = _FakeForecastConfig

# --- lag_llama stubs -------------------------------------------------------
_lag_llama = _make_module("lag_llama")
_lag_llama_gluon = _make_module("lag_llama.gluon")
_lag_llama_gluon_est = _make_module("lag_llama.gluon.estimator")

class _FakeLagLlamaEstimator:
    def __init__(self, **kw):
        self._kw = kw
    def create_lightning_module(self):
        return MagicMock()
    def create_transformation(self):
        return MagicMock()
    def create_predictor(self, transformation, module):
        return MagicMock()

_lag_llama_gluon_est.LagLlamaEstimator = _FakeLagLlamaEstimator

# --- huggingface_hub stub --------------------------------------------------
_hf_hub = _make_module("huggingface_hub")
_hf_hub.hf_hub_download = MagicMock(return_value="/tmp/fake.ckpt")

# --- gluonts stubs ---------------------------------------------------------
_gluonts = _make_module("gluonts")
_gluonts_eval = _make_module("gluonts.evaluation")

class _FakeForecastObj:
    def __init__(self, h=10):
        self.mean = np.ones(h)
        self.samples = np.ones((100, h))
    def quantile(self, q):
        return np.ones(10) * q

_gluonts_eval.make_evaluation_predictions = MagicMock(
    side_effect=lambda **kw: (iter([_FakeForecastObj()]), iter([]))
)
_gluonts_dataset = _make_module("gluonts.dataset")
_gluonts_dataset_common = _make_module("gluonts.dataset.common")
_gluonts_dataset_common.ListDataset = MagicMock(return_value=[{"target": np.ones(50), "start": pd.Timestamp("2000-01-01")}])

# --- Register all stubs in sys.modules ------------------------------------
# Stubs are installed at module level for the initial import, then
# re-installed before each test (via autouse fixture) to survive cleanup
# by other test modules that share stub keys (e.g., test_gluonts_coverage).
_TORCH_STUBS = {
    "torch": _torch,
    "torch.cuda": _torch_cuda,
    "torch.serialization": _torch_serial,
}
_NON_TORCH_STUBS = {
    "chronos": _chronos,
    "timesfm": _timesfm,
    "timesfm.torch": _timesfm_torch,
    "timesfm.timesfm_2p5": _timesfm_2p5,
    "timesfm.timesfm_2p5.timesfm_2p5_torch": _timesfm_2p5_torch,
    "timesfm.configs": _timesfm_configs,
    "lag_llama": _lag_llama,
    "lag_llama.gluon": _lag_llama_gluon,
    "lag_llama.gluon.estimator": _lag_llama_gluon_est,
    "huggingface_hub": _hf_hub,
    "gluonts": _gluonts,
    "gluonts.evaluation": _gluonts_eval,
    "gluonts.dataset": _gluonts_dataset,
    "gluonts.dataset.common": _gluonts_dataset_common,
}

# Install stubs for the module-level import of pretrained
_originals = {}
for name, mod in _NON_TORCH_STUBS.items():
    _originals[name] = sys.modules.get(name)
    sys.modules[name] = mod

# Now import the module under test
from mtdata.forecast.interface import ForecastResult
import mtdata.forecast.methods.pretrained as pretrained_module
from mtdata.forecast.methods.pretrained import (
    ChronosBoltMethod,
    LagLlamaMethod,
    PretrainedMethod,
    TimesFMMethod,
    _resolve_chronos_model_defaults,
    _resolve_chronos_device_map,
    _stringify_exception_chain,
)

# ---------------------------------------------------------------------------
# Per-test fixtures to ensure stubs are present even if other modules
# cleaned them up between tests.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_stubs():
    """Re-install non-torch stubs before every test & clean up after."""
    saved = {}
    for name, mod in _NON_TORCH_STUBS.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    yield
    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


@pytest.fixture()
def _with_torch_stubs():
    """Temporarily inject torch stubs for a single test, then clean up."""
    saved = {}
    for name, mod in _TORCH_STUBS.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    yield
    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig
    try:
        from scipy._lib.array_api_compat.common._helpers import _issubclass_fast
        _issubclass_fast.cache_clear()
    except Exception:
        pass


# ===========================================================================
# Helpers
# ===========================================================================

def _series(n: int = 100) -> pd.Series:
    return pd.Series(np.random.rand(n) * 100 + 50, name="price")


# ===========================================================================
# _resolve_chronos_device_map
# ===========================================================================

class TestResolveChronosDeviceMap:
    def test_empty_no_cuda(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = False
        assert _resolve_chronos_device_map(None, torch_mod) == "cpu"

    def test_empty_with_cuda(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        assert _resolve_chronos_device_map("", torch_mod) == "cuda:0"

    def test_explicit_cpu(self):
        assert _resolve_chronos_device_map("cpu", MagicMock()) == "cpu"

    def test_explicit_cuda1(self):
        assert _resolve_chronos_device_map("cuda:1", MagicMock()) == "cuda:1"

    def test_auto_single_gpu(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        torch_mod.cuda.device_count.return_value = 1
        assert _resolve_chronos_device_map("auto", torch_mod) == "cuda:0"

    def test_auto_multi_gpu(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        torch_mod.cuda.device_count.return_value = 4
        assert _resolve_chronos_device_map("auto", torch_mod) == "cuda:0"

    def test_auto_no_cuda(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = False
        assert _resolve_chronos_device_map("auto", torch_mod) == "cpu"

    def test_none_cuda_exception(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.side_effect = RuntimeError("no cuda")
        assert _resolve_chronos_device_map(None, torch_mod) == "cpu"

    def test_whitespace_treated_as_empty(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = False
        assert _resolve_chronos_device_map("   ", torch_mod) == "cpu"


# ===========================================================================
# PretrainedMethod base class
# ===========================================================================

class TestPretrainedMethodBase:
    def test_category(self):
        m = ChronosBoltMethod()
        assert m.category == "pretrained"

    def test_supports_features(self):
        m = ChronosBoltMethod()
        feats = m.supports_features
        assert feats["price"] is True
        assert feats["ci"] is True


# ===========================================================================
# ChronosBoltMethod (lines 96-268)
# ===========================================================================

@pytest.mark.usefixtures("_with_torch_stubs")
class TestChronosBoltMethod:
    def setup_method(self):
        self.method = ChronosBoltMethod()

    def test_name(self):
        assert self.method.name == "chronos_bolt"

    def test_required_packages(self):
        assert "chronos" in self.method.required_packages
        assert "torch" in self.method.required_packages

    def test_forecast_basic(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={})
        assert isinstance(res, ForecastResult)
        assert res.forecast is not None

    def test_forecast_with_model_name(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"model_name": "amazon/chronos-bolt-small"})
        assert res.params_used["model_name"] == "amazon/chronos-bolt-small"

    def test_forecast_with_context_length(self):
        res = self.method.forecast(_series(200), horizon=10, seasonality=1, params={"context_length": 50})
        assert res is not None

    def test_forecast_zero_context_length(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"context_length": 0})
        assert res.forecast is not None

    def test_forecast_with_quantiles(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"quantiles": [0.1, 0.5, 0.9]})
        assert "quantiles" in res.metadata

    def test_forecast_device_map_cpu(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"device_map": "cpu"})
        assert res.params_used["device_map"] == "cpu"

    def test_forecast_short_series(self):
        res = self.method.forecast(_series(5), horizon=3, seasonality=1, params={})
        assert res.forecast is not None

    def test_forecast_long_horizon(self):
        res = self.method.forecast(_series(), horizon=50, seasonality=1, params={})
        assert res.forecast is not None

    def test_forecast_with_exog(self):
        s = _series(50)
        exog_hist = np.random.rand(50, 2)
        exog_fut = np.random.rand(10, 2)
        res = self.method.forecast(s, horizon=10, seasonality=1, params={}, exog_used=exog_hist, exog_future=exog_fut)
        assert res.forecast is not None

    def test_forecast_exog_in_params(self):
        s = _series(50)
        res = self.method.forecast(s, horizon=10, seasonality=1, params={
            "exog_used": np.random.rand(50, 2),
            "exog_future": np.random.rand(10, 2),
        })
        assert res.forecast is not None

    def test_forecast_none_params(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params=None)
        assert res.forecast is not None

    def test_default_model_name(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"method_name": "chronos_bolt"})
        assert "chronos-bolt-base" in res.params_used["model_name"]
        assert res.params_used.get("pipeline") == "ChronosBoltPipeline"

    def test_chronos2_default_model_name_prefers_t5(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"method_name": "chronos2"})
        assert res.params_used["model_name"] == "amazon/chronos-t5-small"
        assert res.params_used.get("pipeline") == "ChronosPipeline"

    def test_prepare_forecast_call_injects_method_name(self):
        context = types.SimpleNamespace(method="chronos2")
        params, kwargs = self.method.prepare_forecast_call({}, {}, context)
        assert params["method_name"] == "chronos2"
        assert kwargs == {}

    def test_falls_back_when_chronos2_pipeline_init_fails(self):
        class _BrokenChronos2Pipeline:
            @classmethod
            def from_pretrained(cls, model_name, device_map=None):
                raise AttributeError("module 'chronos.chronos2' has no attribute 'ChronosBoltModelForForecasting'")

        saved = getattr(_chronos, "Chronos2Pipeline", None)
        _chronos.Chronos2Pipeline = _BrokenChronos2Pipeline
        try:
            with pytest.raises(RuntimeError, match="failed to initialize any compatible Chronos pipeline"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={"model_name": "amazon/chronos-2", "method_name": "chronos2"})
        finally:
            if saved is None:
                delattr(_chronos, "Chronos2Pipeline")
            else:
                _chronos.Chronos2Pipeline = saved


def test_resolve_chronos_model_defaults_uses_t5_for_chronos2():
    model_name, order = _resolve_chronos_model_defaults("chronos2", {})
    assert model_name == "amazon/chronos-t5-small"
    assert order == ("ChronosPipeline",)


def test_resolve_chronos_model_defaults_uses_bolt_for_chronos_bolt():
    model_name, order = _resolve_chronos_model_defaults("chronos_bolt", {})
    assert model_name == "amazon/chronos-bolt-base"
    assert order == ("ChronosBoltPipeline",)

    def test_predict_quantiles_fallback_to_predict(self):
        """When predict_quantiles returns None, falls back to predict."""
        orig = _FakePipeline.predict_quantiles
        _FakePipeline.predict_quantiles = lambda self, *a, **k: (None, None)
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakePipeline.predict_quantiles = orig

    def test_from_pretrained_type_error_fallback(self):
        """When from_pretrained raises TypeError on device_map kwarg, retries without."""
        call_count = [0]
        orig = _FakePipeline.from_pretrained

        @classmethod
        def _new_from_pretrained(cls, model_name, device_map=None):
            call_count[0] += 1
            if call_count[0] == 1 and device_map is not None:
                raise TypeError("unexpected kwarg device_map")
            return orig.__func__(cls, model_name)

        _FakePipeline.from_pretrained = _new_from_pretrained
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakePipeline.from_pretrained = orig

    def test_predict_quantiles_type_error_retry(self):
        """predict_quantiles TypeError triggers retry without extra kwargs."""
        call_count = [0]
        orig_pq = _FakePipeline.predict_quantiles

        def _new_pq(self_pipe, ctx, prediction_length=10, quantile_levels=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1 and kw:
                raise TypeError("unexpected kwargs")
            return orig_pq(self_pipe, ctx, prediction_length=prediction_length, quantile_levels=quantile_levels)

        _FakePipeline.predict_quantiles = _new_pq
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"quantiles": [0.5]})
            assert res.forecast is not None
        finally:
            _FakePipeline.predict_quantiles = orig_pq

    def test_no_pipeline_found(self):
        """Error when chronos module has no pipeline classes."""
        saved = {}
        for attr in ("Chronos2Pipeline", "ChronosBoltPipeline", "ChronosPipeline"):
            saved[attr] = getattr(_chronos, attr, None)
            if hasattr(_chronos, attr):
                delattr(_chronos, attr)
        try:
            with pytest.raises(RuntimeError, match="no supported pipeline found"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            for attr, val in saved.items():
                if val is not None:
                    setattr(_chronos, attr, val)

    def test_quantile_shape_q_axis_0(self):
        """Quantile tensor shape (n_q, h) -> q_axis=0."""
        def _pq(self_pipe, ctx, prediction_length=10, quantile_levels=None, **kw):
            n_q = len(quantile_levels) if quantile_levels else 1
            return _FakeTensor(np.random.rand(n_q, prediction_length)), _FakeTensor(np.random.rand(1, prediction_length))
        orig = _FakePipeline.predict_quantiles
        _FakePipeline.predict_quantiles = _pq
        try:
            res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"quantiles": [0.1, 0.9]})
            assert res.metadata["quantiles"]
        finally:
            _FakePipeline.predict_quantiles = orig

    def test_forecast_pads_short_chronos_outputs_to_requested_horizon(self):
        orig = _FakePipeline.predict_quantiles

        def _pq(self_pipe, ctx, prediction_length=10, quantile_levels=None, **kw):
            short_h = max(1, int(prediction_length) - 2)
            n_q = len(quantile_levels) if quantile_levels else 1
            return _FakeTensor(np.ones((1, short_h, n_q))), _FakeTensor(np.arange(short_h, dtype=float).reshape(1, short_h))

        _FakePipeline.predict_quantiles = _pq
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"quantiles": [0.5]})
            assert len(res.forecast) == 5
            assert res.forecast.tolist() == [0.0, 1.0, 2.0, 2.0, 2.0]
        finally:
            _FakePipeline.predict_quantiles = orig

    def test_import_error_raises(self):
        """When chronos not importable, RuntimeError is raised."""
        saved = sys.modules.get("chronos")
        sys.modules["chronos"] = None  # makes __import__ fail
        try:
            m = ChronosBoltMethod()
            with pytest.raises(RuntimeError):
                m.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            sys.modules["chronos"] = saved


# ===========================================================================
# TimesFMMethod (lines 296-492)
# ===========================================================================

@pytest.mark.usefixtures("_with_torch_stubs")
class TestTimesFMMethod:
    def setup_method(self):
        self.method = TimesFMMethod()

    def test_name(self):
        assert self.method.name == "timesfm"

    def test_required_packages(self):
        assert "timesfm" in self.method.required_packages

    def test_forecast_basic(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={})
        assert isinstance(res, ForecastResult)
        assert res.forecast is not None
        assert len(res.forecast) == 10

    def test_forecast_with_context_length(self):
        res = self.method.forecast(_series(200), horizon=10, seasonality=1, params={"context_length": 64})
        assert res.forecast is not None

    def test_forecast_with_quantiles_dict(self):
        """Quantiles returned as dict from forecast call."""
        def _forecast_dict(self_mdl, inputs, horizon):
            return np.ones((1, horizon)), {"0.5": np.ones(horizon)}
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _forecast_dict
        try:
            res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"quantiles": [0.5]})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_with_quantiles_array(self):
        """Quantiles returned as 3-D array."""
        def _forecast_arr(self_mdl, inputs, horizon):
            return np.ones((1, horizon)), np.random.rand(1, horizon, 9)
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _forecast_arr
        try:
            res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"quantiles": [0.1, 0.5, 0.9]})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_model_class_override(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"model_class": "TimesFM_2p5_200M_torch"})
        assert res.forecast is not None

    def test_forecast_config_kwargs(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"context_length": 128})
        assert res.forecast is not None

    def test_forecast_none_point_raises(self):
        """When forecast returns None point forecast, RuntimeError raised."""
        def _forecast_none(self_mdl, inputs, horizon):
            return None, None
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _forecast_none
        try:
            with pytest.raises(RuntimeError, match="no point forecast"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_2d_point(self):
        """2-D point forecast array handled correctly."""
        def _forecast_2d(self_mdl, inputs, horizon):
            return np.ones((1, horizon)) * 42, None
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _forecast_2d
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            np.testing.assert_allclose(res.forecast, 42.0)
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_constructor_type_error(self):
        """Constructor TypeError falls back with device kwarg."""
        call_count = [0]
        orig_init = _FakeTimesFMTorch.__init__

        def _new_init(self_mdl, **kw):
            call_count[0] += 1
            if call_count[0] == 1 and not kw:
                raise TypeError("needs device")
            orig_init(self_mdl, **kw)

        _FakeTimesFMTorch.__init__ = _new_init
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.__init__ = orig_init

    def test_resolve_forecast_config_missing(self):
        """When ForecastConfig missing, raises RuntimeError."""
        saved = _timesfm.ForecastConfig
        del _timesfm.ForecastConfig
        _timesfm_configs.ForecastConfig = None
        try:
            with pytest.raises(RuntimeError, match="ForecastConfig"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _timesfm.ForecastConfig = saved
            _timesfm_configs.ForecastConfig = _FakeForecastConfig

    def test_import_error_raises(self):
        saved = sys.modules.get("timesfm")
        sys.modules["timesfm"] = None
        try:
            m = TimesFMMethod()
            with pytest.raises(RuntimeError):
                m.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            sys.modules["timesfm"] = saved

    def test_short_series(self):
        res = self.method.forecast(_series(3), horizon=2, seasonality=1, params={})
        assert len(res.forecast) == 2

    def test_forecast_call_signature_fallbacks(self):
        """_call_forecast tries multiple signatures."""
        call_count = [0]
        def _tricky_forecast(self_mdl, inputs, horizon):
            call_count[0] += 1
            return np.ones(horizon), None
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _tricky_forecast
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_returns_dict(self):
        """Model forecast returns dict instead of tuple."""
        def _forecast_dict(self_mdl, inputs, horizon):
            return {"point_forecast": np.ones(horizon), "quantiles": None}
        # The internal _call_forecast tries tuple, then dict, then raw
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = lambda self, **kw: {"point_forecast": np.ones(kw.get("horizon", 5))}
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.forecast = orig


# ===========================================================================
# LagLlamaMethod (lines 525-702)
# ===========================================================================

@pytest.mark.usefixtures("_with_torch_stubs")
class TestLagLlamaMethod:
    def setup_method(self):
        self.method = LagLlamaMethod()

    def test_name(self):
        assert self.method.name == "lag_llama"

    def test_required_packages(self):
        pkgs = self.method.required_packages
        assert "lag-llama" in pkgs
        assert "gluonts" in pkgs

    def test_forecast_with_ckpt(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"ckpt_path": "/tmp/fake.ckpt"})
        assert isinstance(res, ForecastResult)
        assert res.forecast is not None

    def test_forecast_auto_download(self):
        """Uses huggingface_hub to download when ckpt_path missing."""
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={})
        assert res.forecast is not None

    def test_forecast_custom_hf_repo(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={
            "hf_repo": "my-org/my-model",
            "hf_filename": "model.ckpt",
        })
        assert res.forecast is not None

    def test_forecast_with_quantiles(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={
            "ckpt_path": "/tmp/fake.ckpt",
            "quantiles": [0.1, 0.5, 0.9],
        })
        assert "quantiles" in res.metadata

    def test_forecast_context_length(self):
        res = self.method.forecast(_series(200), horizon=10, seasonality=1, params={
            "ckpt_path": "/tmp/fake.ckpt",
            "context_length": 32,
        })
        assert res.params_used["context_length"] == 32

    def test_forecast_num_samples(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={
            "ckpt_path": "/tmp/fake.ckpt",
            "num_samples": 50,
        })
        assert res.params_used["num_samples"] == 50

    def test_forecast_rope_scaling(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={
            "ckpt_path": "/tmp/fake.ckpt",
            "use_rope_scaling": True,
            "context_length": 64,
        })
        assert res.params_used["use_rope_scaling"] is True

    def test_forecast_device_explicit(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={
            "ckpt_path": "/tmp/fake.ckpt",
            "device": "cpu",
        })
        assert "cpu" in str(res.params_used["device"])

    def test_forecast_freq(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={
            "ckpt_path": "/tmp/fake.ckpt",
            "freq": "D",
        })
        assert res.params_used["freq"] == "D"

    def test_checkpoint_alias(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={
            "checkpoint": "/tmp/fake.ckpt",
        })
        assert res.forecast is not None

    def test_model_path_alias(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={
            "model_path": "/tmp/fake.ckpt",
        })
        assert res.forecast is not None

    def test_hf_download_failure(self):
        """When hf_hub_download raises, RuntimeError propagated."""
        _hf_hub.hf_hub_download.side_effect = OSError("download failed")
        try:
            with pytest.raises(RuntimeError, match="auto-download"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _hf_hub.hf_hub_download.side_effect = None
            _hf_hub.hf_hub_download.return_value = "/tmp/fake.ckpt"

    def test_short_series_context(self):
        res = self.method.forecast(_series(5), horizon=3, seasonality=1, params={"ckpt_path": "/tmp/fake.ckpt", "context_length": 100})
        assert res.forecast is not None

    def test_zero_context_uses_full(self):
        res = self.method.forecast(_series(20), horizon=5, seasonality=1, params={"ckpt_path": "/tmp/fake.ckpt", "context_length": 0})
        assert res.forecast is not None

    def test_quantiles_params_used(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={
            "ckpt_path": "/tmp/fake.ckpt",
            "quantiles": [0.25, 0.75],
        })
        assert "quantiles" in res.params_used


@pytest.mark.parametrize(
    "name",
    ["forecast_chronos_bolt", "forecast_timesfm", "forecast_lag_llama"],
)
def test_pretrained_wrapper_functions_removed(name):
    with pytest.raises(AttributeError):
        getattr(pretrained_module, name)


@pytest.mark.usefixtures("_with_torch_stubs")
class TestPretrainedErrorContext:
    def test_timesfm_forecast_preserves_root_cause(self, monkeypatch):
        def _boom(self, inputs, horizon):
            raise ValueError("predict failed")

        monkeypatch.setattr(_FakeTimesFMTorch, "forecast", _boom)

        method = TimesFMMethod()
        with pytest.raises(RuntimeError, match="timesfm error: predict failed") as exc_info:
            method.forecast(_series(50), 5, 0, {})

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "predict failed"

    def test_lag_llama_forecast_preserves_root_cause(self):
        original = sys.modules["gluonts.evaluation"].make_evaluation_predictions
        sys.modules["gluonts.evaluation"].make_evaluation_predictions = MagicMock(
            side_effect=ValueError("predict failed")
        )
        try:
            method = LagLlamaMethod()
            with pytest.raises(RuntimeError, match="lag_llama inference error: predict failed") as exc_info:
                method.forecast(_series(50), 5, 0, {"ckpt_path": "/tmp/fake.ckpt"})
        finally:
            sys.modules["gluonts.evaluation"].make_evaluation_predictions = original

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "predict failed"


def test_stringify_exception_chain_joins_nested_causes():
    try:
        try:
            raise ValueError("root cause")
        except ValueError as inner:
            raise RuntimeError("outer cause") from inner
    except RuntimeError as ex:
        assert _stringify_exception_chain(ex) == "outer cause | caused by: root cause"
