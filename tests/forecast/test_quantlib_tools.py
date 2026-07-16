from __future__ import annotations

import datetime as _dt
import types

from mtdata.forecast import quantlib_tools as qtools


def _make_fake_quantlib():  # noqa: C901
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
        def __init__(self, day=None, month=None, year=None):
            if day is None or month is None or year is None:
                self.ordinal = _dt.date(2026, 1, 1).toordinal()
            else:
                self.ordinal = _dt.date(int(year), int(month), int(day)).toordinal()

        @classmethod
        def from_ordinal(cls, ordinal):
            instance = cls.__new__(cls)
            instance.ordinal = int(ordinal)
            return instance

        @staticmethod
        def todaysDate():
            return _Date.from_ordinal(_dt.date(2026, 1, 1).toordinal())

        def __add__(self, _days):
            return _Date.from_ordinal(self.ordinal + int(_days))

        def __sub__(self, other):
            return int(self.ordinal - other.ordinal)

        def year(self):
            return _dt.date.fromordinal(self.ordinal).year

        def month(self):
            return _dt.date.fromordinal(self.ordinal).month

        def dayOfMonth(self):
            return _dt.date.fromordinal(self.ordinal).day

    class _UnitedStates:
        NYSE = "NYSE"

        def __init__(self, market=None):
            self.market = market

        def advance(self, date, days, _unit):
            return date + int(days)

        def businessDaysBetween(self, start, end):
            return max(0, (end - start) - 4)

    class _NullCalendar:
        def advance(self, date, days, _unit):
            return date + int(days)

        def businessDaysBetween(self, start, end):
            return max(0, end - start)

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
        created = []

        def __init__(self, _maturity, _calendar, spot, strike, iv_handle, _rf_ts, _div_ts, _err_type, option_type="call"):
            self.spot = float(spot)
            self.strike = float(strike)
            self.iv = float(iv_handle.quote.value)
            self.option_type = option_type
            type(self).created.append(self)

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
        NullCalendar=_NullCalendar,
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
    assert out["greeks_status"] == "complete"
    assert out["greeks_spot_step"] == 0.01


def test_price_barrier_option_quantlib_uses_safe_step_near_barrier(monkeypatch):
    fake = _make_fake_quantlib()
    base_option = fake.BarrierOption

    class _StrictBarrierOption(base_option):
        def NPV(self):
            spot = self._engine.process.spot
            if self.barrier_type.startswith("up_") and spot >= self.barrier:
                raise RuntimeError("barrier touched")
            if self.barrier_type.startswith("down_") and spot <= self.barrier:
                raise RuntimeError("barrier touched")
            return super().NPV()

    fake.BarrierOption = _StrictBarrierOption
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)

    out = qtools.price_barrier_option_quantlib(
        spot=1.168,
        strike=1.15,
        barrier=1.17,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
    )

    assert out["success"] is True
    assert out["greeks_status"] == "complete"
    assert out["greeks_method"] == "central_difference"
    assert out["greeks_spot_step"] < 1.17 - 1.168
    assert out["delta"] is not None


def test_price_barrier_option_quantlib_exposes_calendar_overrides(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())

    out = qtools.price_barrier_option_quantlib(
        spot=100.0,
        strike=100.0,
        barrier=120.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        calendar="NullCalendar",
        maturity_basis="business_days",
        valuation_date="2026-07-03",
    )

    assert out["success"] is True
    assert out["pricing_assumptions"]["calendar"] == "NullCalendar"
    assert out["pricing_assumptions"]["maturity_basis"] == "business_days"
    assert out["params_used"]["calendar"] == "NullCalendar"
    assert out["params_used"]["maturity_basis"] == "business_days"
    assert out["valuation_date"] == "2026-07-03"
    assert out["maturity_date"] == "2026-08-02"
    assert out["time_to_maturity_years"] == 30 / 365
    assert out["params_used"]["valuation_date"] == "2026-07-03"


def test_price_barrier_option_quantlib_validates_touched_down_barrier():
    out = qtools.price_barrier_option_quantlib(
        spot=100.0,
        strike=100.0,
        barrier=100.0,
        maturity_days=30,
        option_type="call",
        barrier_type="down_in",
        risk_free_rate=0.02,
        dividend_yield=0.0,
        volatility=0.2,
        rebate=0.0,
    )

    assert out["error_code"] == "invalid_barrier_geometry"
    assert out["error"] == "For a down barrier option, barrier must be below spot."
    assert out["params_used"]["spot"] == 100.0
    assert out["params_used"]["barrier_type"] == "down_in"


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
        valuation_date="2026-12-01",
    )
    assert out["success"] is True
    assert out["symbol"] == "AAPL"
    assert out["valuation_date"] == "2026-12-01"
    assert out["days_to_expiry"] == 18
    assert out["contracts_used"] == 5
    assert set(out["params"].keys()) == {"kappa", "theta", "sigma", "rho", "v0"}
    assert out["calibration_error_rmse"] is not None


def test_calibrate_heston_uses_contract_option_side(monkeypatch):
    fake = _make_fake_quantlib()
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            "options": [
                {"strike": strike, "implied_volatility": 0.25, "side": side}
                for strike, side in zip((90, 95, 100, 105, 110), ("put", "call", "put", "call", "put"))
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL", expiration="2026-12-19", option_type="both", valuation_date="2026-12-01"
    )

    assert out["success"] is True
    assert {helper.option_type for helper in fake.HestonModelHelper.created} == {
        fake.Option.Call,
        fake.Option.Put,
    }


def test_calibrate_heston_quantlib_uses_calendar_override_for_business_days(monkeypatch):
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
        valuation_date="2026-12-01",
        calendar="NullCalendar",
        maturity_basis="business_days",
    )

    assert out["success"] is True
    assert out["days_to_expiry"] == 18
    assert out["pricing_assumptions"]["calendar"] == "NullCalendar"
    assert out["pricing_assumptions"]["maturity_basis"] == "business_days"


def test_calibrate_heston_rejects_invalid_valuation_date(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            "options": [
                {"strike": strike, "implied_volatility": 0.25, "side": "call"}
                for strike in (90, 95, 100, 105, 110)
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        valuation_date="12/01/2026",
    )

    assert out == {
        "error": "Invalid valuation_date: 12/01/2026. Use YYYY-MM-DD."
    }
