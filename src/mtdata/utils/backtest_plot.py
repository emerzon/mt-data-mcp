"""
Backtest plotting helper.

Generates summary charts from forecast_backtest results:
- Bar chart of Avg RMSE per method (plus Directional Accuracy / Win Rate labels when available)
- Cumulative equity curve for the best method (by avg_rmse)
- Small multiples of per-anchor Actual vs Forecast for the best method

Run as a module:
  python -m src.mtdata.utils.backtest_plot --symbol EURUSD --timeframe H1 \
      --horizon 12 --steps 5 --spacing 20 --out-dir backtests

Or via CLI tool once wired (see core/plots.py):
  python cli.py backtest_plot_generate EURUSD --timeframe H1 --horizon 12 --steps 5 --spacing 20
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os

import numpy as np


def _import_matplotlib():
    import importlib
    mpl = importlib.import_module("matplotlib")
    import matplotlib.pyplot as plt
    return mpl, plt


def _best_method_from_backtest(bt: Dict[str, Any]) -> Optional[str]:
    results = bt.get("results") if isinstance(bt, dict) else None
    if not isinstance(results, dict):
        return None
    best: Tuple[str, float] | None = None
    for m, res in results.items():
        if not isinstance(res, dict) or not res.get("success"):
            continue
        try:
            rmse = float(res.get("avg_rmse", float("inf")))
        except Exception:
            rmse = float("inf")
        if best is None or rmse < best[1]:
            best = (m, rmse)
    return best[0] if best else None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fmt_pct(x: Optional[float]) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "n/a"
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "n/a"


def generate_backtest_plots(
    *,
    symbol: str,
    timeframe: str = "H1",
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    methods: Optional[List[str]] = None,
    denoise: Optional[Dict[str, Any]] = None,
    slippage_bps: float = 0.0,
    trade_threshold: float = 0.0,
    out_dir: str | os.PathLike = "backtests",
    filename_base: Optional[str] = None,
    target: str = "price",
    plot_delta: bool = False,
) -> Dict[str, Any]:
    """Run a rolling backtest and emit summary PNGs.

    Returns a dict with file paths.
    """
    # Defer heavy imports
    _, plt = _import_matplotlib()
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        pass

    # Ensure MT5 session and call the raw implementation to avoid MCP text-minifying
    from ..utils.mt5 import mt5_connection
    if not mt5_connection._ensure_connection():
        return {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}

    from ..forecast.backtest import forecast_backtest as _forecast_backtest_impl

    bt = _forecast_backtest_impl(
        symbol=symbol,
        timeframe=timeframe,
        horizon=int(horizon),
        steps=int(steps),
        spacing=int(spacing),
        methods=methods,
        denoise=denoise,
        target=target,
        slippage_bps=float(slippage_bps),
        trade_threshold=float(trade_threshold),
    )
    if not isinstance(bt, dict) or bt.get("error"):
        return {"error": bt.get("error") if isinstance(bt, dict) else "Backtest failed"}

    results = bt.get("results", {})
    if not results:
        return {"error": "No results"}

    out_path = Path(out_dir)
    _ensure_dir(out_path)
    base = filename_base or f"{symbol}_{timeframe}_h{horizon}_s{steps}_p{spacing}"
    explicit_png_name = None
    if isinstance(filename_base, str) and filename_base.lower().endswith('.png'):
        explicit_png_name = filename_base

    # 1) Summary bar chart (avg_rmse + labels for DA/win rate)
    methods_sorted = []
    for m, r in results.items():
        if not isinstance(r, dict) or not r.get("success"):
            continue
        methods_sorted.append((m, float(r.get("avg_rmse", float("inf"))), r))
    if not methods_sorted:
        return {"error": "No successful method results"}
    methods_sorted.sort(key=lambda x: x[1])

    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=144)
    labels = [m for m, _, _ in methods_sorted]
    values = [rmse for _, rmse, _ in methods_sorted]
    bars = ax1.bar(labels, values, color="#4C78A8")
    ax1.set_title(f"Backtest Avg RMSE by Method — {symbol} {timeframe} (h={horizon}, steps={steps})")
    ax1.set_ylabel("Avg RMSE")
    ax1.set_xlabel("Method")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    # Annotate DA and Win Rate
    for bar, (_, _, res) in zip(bars, methods_sorted):
        da = res.get("avg_directional_accuracy")
        wr = (res.get("metrics") or {}).get("win_rate")
        txt = []
        if da is not None:
            txt.append(f"DA {_fmt_pct(da)}")
        if wr is not None:
            txt.append(f"WR {_fmt_pct(wr)}")
        if txt:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), "\n".join(txt),
                     ha='center', va='bottom', fontsize=8)
    fig1.tight_layout()
    file_summary = out_path / f"{base}_summary.png"
    fig1.savefig(file_summary)
    plt.close(fig1)

    # 2) Equity curve for best method
    best_method = _best_method_from_backtest(bt)
    if best_method is None:
        best_method = labels[0]
    best = results.get(best_method) or {}
    details = best.get("details") or []
    rets = []
    anchors = []
    for d in details:
        anchors.append(d.get("anchor"))
        r = d.get("trade_return")
        rets.append(float(r) if r is not None else 0.0)
    equity = np.cumprod(1.0 + np.asarray(rets, dtype=float)) if rets else np.array([1.0])

    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=144)
    ax2.plot(range(len(equity)), equity, marker='o', color="#F58518")
    ax2.set_title(f"Equity Curve — {best_method} ({symbol} {timeframe}, h={horizon})")
    ax2.set_ylabel("Equity (1=flat)")
    ax2.set_xlabel("Backtest step")
    ax2.grid(True, alpha=0.3)
    # Annotate end stats
    if rets:
        total = equity[-1] - 1.0
        wr = float(np.mean(np.array(rets) > 0.0)) if rets else 0.0
        ax2.text(0.02, 0.95, f"Total {_fmt_pct(total)}\nWin {_fmt_pct(wr)}",
                 transform=ax2.transAxes, ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, lw=0.0))
    fig2.tight_layout()
    file_equity = out_path / f"{base}_equity_{best_method}.png"
    fig2.savefig(file_equity)
    plt.close(fig2)

    # 3) Small multiples: Actual vs Forecast for best method
    # Build pairs for each anchor
    pairs: List[Tuple[str, List[float], List[float]]] = []
    for d in details:
        fc = d.get("forecast") or []
        act = d.get("actual") or []
        if fc and act:
            m = min(len(fc), len(act))
            fcs = fc[:m]
            acts = act[:m]
            if plot_delta and target == 'price':
                try:
                    entry = float(d.get('entry_price'))
                    fcs = [float(v) - entry for v in fcs]
                    acts = [float(v) - entry for v in acts]
                except Exception:
                    pass
            pairs.append((str(d.get("anchor")), fcs, acts))
    if pairs:
        cols = min(3, max(1, int(np.ceil(np.sqrt(len(pairs))))))
        rows = int(np.ceil(len(pairs) / cols))
        fig3, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.8*rows), dpi=144, squeeze=False)
        for i, (anchor, fc, act) in enumerate(pairs):
            r = i // cols; c = i % cols
            ax = axes[r][c]
            ax.plot(act, label='Actual', color='#4C78A8', lw=1.5)
            ax.plot(fc, label='Forecast', color='#E45756', lw=1.2)
            ax.set_title(anchor, fontsize=9)
            ax.grid(True, alpha=0.3)
            if r == rows - 1:
                ax.set_xlabel("Step")
            if c == 0:
                ax.set_ylabel("ΔPrice" if (plot_delta and target == 'price') else ("Return" if target == 'return' else "Price"))
        # Hide empty axes
        for j in range(len(pairs), rows*cols):
            r = j // cols; c = j % cols
            axes[r][c].axis('off')
        # Shared legend
        handles, labels_ = axes[0][0].get_legend_handles_labels()
        fig3.legend(handles, labels_, loc='upper center', ncol=2)
        fig3.suptitle(f"Best Method {best_method}: Forecast vs Actual — {symbol} {timeframe}", y=0.995)
        fig3.tight_layout(rect=(0, 0, 1, 0.96))
        file_pairs = out_path / f"{base}_anchors_{best_method}.png"
        fig3.savefig(file_pairs)
        plt.close(fig3)
    else:
        file_pairs = None

    # 4) Combined sheet similar to example 'download.png'
    try:
        import matplotlib.pyplot as plt2  # reuse state safely
        from matplotlib.gridspec import GridSpec
        # Rebuild core elements to draw into a single canvas
        fig = plt2.figure(figsize=(14, 8), dpi=140)
        gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.4), height_ratios=(1.0, 1.0), wspace=0.25, hspace=0.25)
        # TL: RMSE bars
        ax_tl = fig.add_subplot(gs[0, 0])
        ax_tl.bar(labels, values, color="#4C78A8")
        ax_tl.set_title("Avg RMSE by Method")
        ax_tl.set_ylabel("Avg RMSE")
        ax_tl.grid(axis="y", alpha=0.3)
        for x, (_, _, res) in enumerate(methods_sorted):
            da = res.get("avg_directional_accuracy")
            wr = (res.get("metrics") or {}).get("win_rate")
            txt = []
            if da is not None:
                txt.append(f"DA {_fmt_pct(da)}")
            if wr is not None:
                txt.append(f"WR {_fmt_pct(wr)}")
            if txt:
                ax_tl.text(x, values[x], "\n".join(txt), ha='center', va='bottom', fontsize=8)
        # BL: Equity
        ax_bl = fig.add_subplot(gs[1, 0])
        ax_bl.plot(range(len(equity)), equity, marker='o', color="#F58518")
        ax_bl.set_title(f"Equity — {best_method}")
        ax_bl.set_xlabel("Backtest step")
        ax_bl.set_ylabel("Equity")
        ax_bl.grid(True, alpha=0.3)
        if rets:
            total = equity[-1] - 1.0
            wr = float(np.mean(np.array(rets) > 0.0)) if rets else 0.0
            ax_bl.text(0.02, 0.95, f"Total {_fmt_pct(total)}\nWin {_fmt_pct(wr)}", transform=ax_bl.transAxes,
                       ha='left', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, lw=0.0))
        # Right: small multiples
        ax_right = fig.add_subplot(gs[:, 1])
        # Create a nested grid inside the right panel
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        pairs_to_plot = pairs[:9] if pairs else []
        if pairs_to_plot:
            ncols = min(3, max(1, int(np.ceil(np.sqrt(len(pairs_to_plot))))))
            nrows = int(np.ceil(len(pairs_to_plot) / ncols))
            sub_gs = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[:, 1], wspace=0.25, hspace=0.35)
            sub_axes = []
            for i in range(nrows * ncols):
                r = i // ncols; c = i % ncols
                sub_axes.append(fig.add_subplot(sub_gs[r, c]))
            for i, (anchor, fc, act) in enumerate(pairs_to_plot):
                ax = sub_axes[i]
                ax.plot(act, label='Actual', color='#4C78A8', lw=1.3)
                ax.plot(fc, label='Forecast', color='#E45756', lw=1.1)
                ax.set_title(anchor, fontsize=9)
                ax.grid(True, alpha=0.25)
                if i // ncols == nrows - 1:
                    ax.set_xlabel("Step")
                if i % ncols == 0:
                    ax.set_ylabel("ΔPrice" if (plot_delta and target == 'price') else ("Return" if target == 'return' else "Price"))
            for j in range(len(pairs_to_plot), len(sub_axes)):
                sub_axes[j].axis('off')
            handles, labels_ = sub_axes[0].get_legend_handles_labels()
            fig.legend(handles, labels_, loc='upper center', ncol=2)
        else:
            ax_right.text(0.5, 0.5, "No anchor plots available", ha='center', va='center')
            ax_right.axis('off')
        fig.suptitle(f"Backtest Summary — {symbol} {timeframe} (h={horizon}, steps={steps}, spacing={spacing}, target={target}{', delta' if (plot_delta and target=='price') else ''})", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if explicit_png_name:
            file_combined = out_path / explicit_png_name
        else:
            file_combined = out_path / f"{base}_download.png"
        fig.savefig(file_combined)
        plt2.close(fig)
    except Exception:
        file_combined = None

    return {
        "success": True,
        "summary_image": str(file_summary),
        "equity_image": str(file_equity),
        "anchors_image": str(file_pairs) if file_pairs else None,
        "best_method": best_method,
        "meta": {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": int(horizon),
            "steps": int(steps),
            "spacing": int(spacing),
            "target": target,
            "plot_delta": bool(plot_delta),
            "methods": list(results.keys()),
        },
        "combined_image": str(file_combined) if file_combined else None,
    }


def _parse_args(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser(description="Generate backtest plots")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="H1")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--spacing", type=int, default=20)
    p.add_argument("--methods", type=str, default=None, help="Space or comma separated method list")
    p.add_argument("--out-dir", default="backtests")
    p.add_argument("--filename-base", default=None)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--trade-threshold", type=float, default=0.0)
    p.add_argument("--target", choices=["price","return"], default="price")
    p.add_argument("--denoise", dest="denoise_method", default=None, help="Denoise method (e.g., ema, sma)")
    p.add_argument("--denoise-span", dest="denoise_span", type=int, default=None, help="Denoise span/window")
    p.add_argument("--plot-delta", action="store_true", help="For price target, plot ΔPrice vs anchor entry")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    methods: Optional[List[str]] = None
    if args.methods:
        txt = str(args.methods).strip()
        if "," in txt:
            methods = [s.strip() for s in txt.split(",") if s.strip()]
        else:
            methods = [s for s in txt.split() if s]
    # Build denoise spec if provided
    denoise_spec = None
    if args.denoise_method:
        denoise_spec = {"method": str(args.denoise_method), "params": {}}
        if args.denoise_span:
            denoise_spec["params"]["span"] = int(args.denoise_span)

    res = generate_backtest_plots(
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizon=int(args.horizon),
        steps=int(args.steps),
        spacing=int(args.spacing),
        methods=methods,
        out_dir=args.out_dir,
        filename_base=args.filename_base,
        slippage_bps=float(args.slippage_bps),
        trade_threshold=float(args.trade_threshold),
        target=str(args.target),
        denoise=denoise_spec,
        plot_delta=bool(args.plot_delta),
    )
    if isinstance(res, dict) and res.get("error"):
        print(f"Error: {res['error']}")
        return 1
    print("Generated:")
    for k in ("summary_image", "equity_image", "anchors_image"):
        v = res.get(k)
        if v:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
