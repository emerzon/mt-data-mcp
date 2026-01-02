# Documentation Index

This folder contains user-facing documentation for mtdata. Start with setup, then follow a guided workflow before exploring individual topics.

## Getting Started

| Document | Description |
|----------|-------------|
| [SETUP.md](SETUP.md) | Installation, MT5 connection, environment variables |
| [CLI.md](CLI.md) | Command conventions, output formats, help system |
| [GLOSSARY.md](GLOSSARY.md) | **Start here** — Explanations of all technical terms |

## Core Topics

| Document | Description |
|----------|-------------|
| [FORECAST.md](FORECAST.md) | Price forecasting methods and usage |
| [forecast/VOLATILITY.md](forecast/VOLATILITY.md) | Volatility estimation and forecasting |
| [forecast/REGIMES.md](forecast/REGIMES.md) | Market regime and change-point detection |
| [forecast/UNCERTAINTY.md](forecast/UNCERTAINTY.md) | Confidence and conformal intervals |
| [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) | TP/SL hit probability and optimization |
| [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) | Available indicators and how to use them |
| [DENOISING.md](DENOISING.md) | Smoothing filters and noise reduction |
| [forecast/PATTERN_SEARCH.md](forecast/PATTERN_SEARCH.md) | Pattern detection and similarity search |

## Tutorials

| Document | Description |
|----------|-------------|
| [SAMPLE-TRADE.md](SAMPLE-TRADE.md) | Beginner step-by-step trade analysis guide |
| [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) | Advanced workflow with regimes, HAR-RV, barriers |
| [EXAMPLE.md](EXAMPLE.md) | Complete end-to-end research loop |

## Troubleshooting

| Document | Description |
|----------|-------------|
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |

## Quick Reference

**List available commands:**
```bash
python cli.py --help
```

**Search for a command:**
```bash
python cli.py --help forecast
python cli.py --help barrier
```

**Get help for a specific command:**
```bash
python cli.py forecast_generate --help
python cli.py regime_detect --help
```
