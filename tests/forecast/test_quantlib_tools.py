from __future__ import annotations

import types

from mtdata.forecast import quantlib_tools as qtools


def _make_fake_quantlib():
    class _Settings:
        _instance = None

        def __init__(self):
            self.evaluationDate = None

        @classmethod
        def instance(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    class _Date:
        @staticmethod
        def todaysDate():
            return _Date()

        def __add__(self, _days):
            return _Date()

    class _UnitedStates:
        NYSE = "NYSE"

        def __init__(self, _market=None):
            pass

    class _Option:
        Call = "call"
        Put = "put"

    class _Barrier:
        UpIn = "up_in"
        UpOut = "up_out"
        DownIn = "down_in"
        DownOut = "down_out"

    class _SimpleQuote:
        def __init__(self, value):
            self.value = float(value)

    class _QuoteHandle:
        def __init__(self, quote):
            self.quote = quote

    class _FlatForward:
        def __init__(self, _today, rate, _day_count):
            self.rate = float(rate)

    class _YieldTermStructureHandle:
        def __init__(self, curve):
            self.curve = curve

    class _BlackConstantVol:
        def __init__(self, _today, _calendar, vol, _day_count):
            self.vol = float(vol)

    class _BlackVolTermStructureHandle:
        def __init__(self, vol):
            self.vol = float(vol.vol)

    class _BlackScholesMertonProcess:
        def __init__(self, spot_h, _div_ts, _rf_ts, vol_ts):
            self.spot = float(spot_h.quote.value)
            self.vol = float(vol_ts.vol)

    class _AnalyticBarrierEngine:
        def __init__(self, process):
            self.process = process

    class _PlainVanillaPayoff:
        def __init__(self, option_type, strike):
            self.option_type = option_type
            self.strike = float(strike)

    class _EuropeanExercise:
        def __init__(self, maturity):
            self.maturity = maturity

    class _BarrierOption:
        def __init__(self, barrier_type, barrier, rebate, payoff, exercise):
            self.barrier_type = barrier_type
            self.barrier = float(barrier)
            self.rebate = float(rebate)
            self.payoff = payoff
            self.exercise = exercise
            self._engine = None

        def setPricingEngine(self, engine):
            self._engine = engine

        def NPV(self):
            if self._engine is None:
                return 0.0
            return max(0.0, 0.01 * self._engine.process.spot + 0.1 * self._engine.process.vol)

    class _HestonProcess:
        def __init__(self, *_args, **_kwargs):
            pass

    class _HestonModel:
        def __init__(self, _process):
            self._kappa = 2.0
            self._theta = 0.04
            self._sigma = 0.30
            self._rho = -0.5
            self._v0 = 0.04

        def calibrate(self, _helpers, _method, _end_criteria):
            return None

        def kappa(self):
            return self._kappa

        def theta(self):
            return self._theta

        def sigma(self):
            return self._sigma

        def rho(self):
            return self._rho

        def v0(self):
            return self._v0

    class _AnalyticHestonEngine:
        def __init__(self, _model):
            pass

    class _Period:
        def __init__(self, length, unit):
            self.length = int(length)
            self.unit = unit

    class _HestonModelHelper:
        def __init__(self, _maturity, _calendar, spot, strike, iv_handle, _rf_ts, _div_ts, _err_type):
            self.spot = float(spot)
            self.strike = float(strike)
            self.iv = float(iv_handle.quote.value)

        def setPricingEngine(self, _engine):
            return None

        def calibrationError(self):
            return abs(self.strike - self.spot) / max(self.spot, 1.0)

    class _BlackCalibrationHelper:
        ImpliedVolError = "iv_err"

    fake = types.SimpleNamespace(
        Date=_Date,
        Settings=_Settings,
        Actual365Fixed=lambda: object(),
        UnitedStates=_UnitedStates,
        Option=_Option,
        Barrier=_Barrier,
        PlainVanillaPayoff=_PlainVanillaPayoff,
        EuropeanExercise=_EuropeanExercise,
        BarrierOption=_BarrierOption,
        QuoteHandle=_QuoteHandle,
        SimpleQuote=_SimpleQuote,
        YieldTermStructureHandle=_YieldTermStructureHandle,
        FlatForward=_FlatForward,
        BlackVolTermStructureHandle=_BlackVolTermStructureHandle,
        BlackConstantVol=_BlackConstantVol,
        BlackScholesMertonProcess=_BlackScholesMertonProcess,
        AnalyticBarrierEngine=_AnalyticBarrierEngine,
        HestonProcess=_HestonProcess,
        HestonModel=_HestonModel,
        AnalyticHestonEngine=_AnalyticHestonEngine,
        Period=_Period,
        Days="days",
        HestonModelHelper=_HestonModelHelper,
        BlackCalibrationHelper=_BlackCalibrationHelper,
        LevenbergMarquardt=lambda: object(),
        EndCriteria=lambda *_args: object(),
    )
    return fake


def test_price_barrier_option_quantlib_with_fake_backend(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    out = qtools.price_barrier_option_quantlib(
        spot=100.0,
        strike=100.0,
        barrier=120.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        risk_free_rate=0.02,
        dividend_yield=0.0,
        volatility=0.2,
        rebate=0.0,
    )
    assert out["success"] is True
    assert out["price"] > 0.0
    assert out["params_used"]["option_type"] == "call"
    assert out["params_used"]["barrier_type"] == "up_out"


def test_calibrate_heston_quantlib_from_options_with_fake_backend(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            "options": [
                {"strike": 90.0, "implied_volatility": 0.35, "side": "call"},
                {"strike": 95.0, "implied_volatility": 0.30, "side": "call"},
                {"strike": 100.0, "implied_volatility": 0.28, "side": "call"},
                {"strike": 105.0, "implied_volatility": 0.29, "side": "call"},
                {"strike": 110.0, "implied_volatility": 0.33, "side": "call"},
                {"strike": 115.0, "implied_volatility": 0.37, "side": "call"},
            ],
        },
    )
    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
        option_type="call",
        risk_free_rate=0.03,
        dividend_yield=0.01,
        min_open_interest=0,
        min_volume=0,
        max_contracts=5,
    )
    assert out["success"] is True
    assert out["symbol"] == "AAPL"
    assert out["contracts_used"] == 5
    assert set(out["params"].keys()) == {"kappa", "theta", "sigma", "rho", "v0"}
    assert out["calibration_error_rmse"] is not None
