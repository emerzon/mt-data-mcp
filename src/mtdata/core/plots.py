from typing import Any, Dict, Optional, List

from .server import mcp, _auto_connect_wrapper


@mcp.tool()
@_auto_connect_wrapper
def backtest_plot_generate(
    symbol: str,
    timeframe: str = "H1",
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    methods: Optional[List[str]] = None,
    slippage_bps: float = 0.0,
    trade_threshold: float = 0.0,
    out_dir: str = "backtests",
    filename_base: Optional[str] = None,
    target: str = "price",
    denoise_method: Optional[str] = None,
    denoise_span: Optional[int] = None,
    plot_delta: bool = False,
) -> Dict[str, Any]:
    """Run a rolling backtest and save summary PNG plots.

    Returns a dict with file paths:
      - summary_image: RMSE ranking chart per method
      - equity_image: equity curve for best method
      - anchors_image: small multiples of forecast vs actual for best method
    """
    try:
        from ..utils.backtest_plot import generate_backtest_plots
        # Build denoise spec from method/span if provided
        denoise = None
        if denoise_method:
            denoise = {"method": str(denoise_method), "params": {}}
            if denoise_span is not None:
                denoise["params"]["span"] = int(denoise_span)

        res = generate_backtest_plots(
            symbol=symbol,
            timeframe=timeframe,
            horizon=int(horizon),
            steps=int(steps),
            spacing=int(spacing),
            methods=methods,
            out_dir=out_dir,
            filename_base=filename_base,
            slippage_bps=float(slippage_bps),
            trade_threshold=float(trade_threshold),
            target=str(target),
            denoise=denoise,
            plot_delta=bool(plot_delta),
        )
        return res
    except Exception as e:
        return {"error": str(e)}
